"""HTML from the row model, PDF from headless Chromium, and the scanned-look JPEG.

Chromium runs with FONTCONFIG_FILE pointing at a private config that lists only
data/generator/fonts. Two reasons, both measured:
  - With the host's fontconfig, Chromium snapped web-font glyph advances to whole pixels,
    and pdfium then split English words into one line per letter in the PDF text layer.
    With the private config the advances are fractional and the text layer reads cleanly.
  - No host font can fill in a missing glyph, so the output does not depend on which fonts
    the machine happens to have, and a glyph outside the Noto fonts shows up in the check.
"""

from __future__ import annotations

import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup, escape
from PIL import Image, ImageFilter
from playwright.sync_api import Browser, Page, sync_playwright

from data.generator.build import Built
from data.generator.content import Document
from data.generator.fonts import FONT_DIR, FONT_FILES
from data.generator.sheet import Cell, Row, bidi_runs

HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"
MIN_SHRINK = 0.72  # a cell may shrink to 72% of its font size to fit on one line

LOAD_FONTS_JS = """async () => {
  await Promise.all([...document.fonts].map((f) => f.load().catch(() => null)));
  await document.fonts.ready;
  return [...document.fonts].map((f) => [f.family, f.weight, f.status]);
}"""

# Shrink any cell whose text is wider than its column, then report what still does not fit.
FIT_JS = """(minShrink) => {
  const clipped = [];
  let shrunk = 0;
  for (const el of document.querySelectorAll('.cell')) {
    const start = parseFloat(getComputedStyle(el).fontSize);
    let size = start;
    while (el.scrollWidth > el.clientWidth && size > start * minShrink) {
      size -= 0.25;
      el.style.fontSize = size + 'px';
    }
    if (size < start) shrunk += 1;
    if (el.scrollWidth > el.clientWidth) clipped.push(el.textContent);
  }
  const page = document.querySelector('.page');
  return { clipped, shrunk, overflow: page.scrollHeight > page.clientHeight };
}"""


class RenderError(RuntimeError):
    pass


def grid_columns(cols: str) -> str:
    """ "1fr" tracks may shrink below their content so an overlong cell overflows itself
    (and gets shrunk or reported) instead of widening the page."""
    return " ".join("minmax(0, 1fr)" if part == "1fr" else part for part in cols.split())


def side(cell: Cell, page_dir: str) -> str:
    if cell.align == "center":
        return "center"
    start = "right" if page_dir == "rtl" else "left"
    end = "left" if page_dir == "rtl" else "right"
    return start if cell.align == "start" else end


def has_arabic_indic_digit(text: str) -> bool:
    return any(unicodedata.bidirectional(ch) == "AN" for ch in text)


def bidi_html(cell: Cell) -> Markup:
    """Wrap each run whose direction differs from the cell's in <bdi>, so Latin and digit
    runs inside Arabic text (and Arabic inside English) keep their own order.

    A run with Arabic-Indic digits is also forced left to right. Without that, the bidi
    algorithm lays out "١٠٠-١٢٣٤-٥٦٧٨-٩١٢٣" or a spaced phone number with its digit groups
    in right-to-left order, which a left-to-right OCR pass would read reversed.
    """
    own = "rtl" if cell.rtl else "ltr"
    parts = []
    for direction, text in bidi_runs(cell.text):
        if direction == "ltr" and has_arabic_indic_digit(text):
            parts.append(Markup('<bdi dir="ltr" class="force-ltr">{}</bdi>').format(text))
        elif direction and direction != own:
            parts.append(Markup('<bdi dir="{}">{}</bdi>').format(direction, text))
        else:
            parts.append(escape(text))
    return Markup("").join(parts)


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["bidi"] = bidi_html
    env.filters["side"] = side
    env.filters["grid"] = grid_columns
    return env


def html_for(doc_id: str, doc: Document, rows: list[Row]) -> str:
    rtl = doc.plan.layout == "ar"
    return (
        _environment()
        .get_template("document.html.j2")
        .render(
            doc_id=doc_id,
            lang="ar" if rtl else "en",
            dir="rtl" if rtl else "ltr",
            rows=rows,
            base_pt=doc.base_pt,
            fonts={key: (FONT_DIR / name).as_uri() for key, name in FONT_FILES.items()},
        )
    )


def fontconfig_file(directory: Path) -> Path:
    path = directory / "fonts.conf"
    path.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
        f"  <dir>{FONT_DIR}</dir>\n  <cachedir>{directory / 'fc-cache'}</cachedir>\n"
        "</fontconfig>\n",
        encoding="utf-8",
    )
    return path


@dataclass
class RenderStats:
    shrunk_cells: int = 0


class PdfRenderer:
    """One headless Chromium for a whole run. Use as a context manager."""

    def __enter__(self) -> PdfRenderer:
        self._tmp = tempfile.TemporaryDirectory(prefix="docbench-render-")
        self.workdir = Path(self._tmp.name)
        env = {**os.environ, "FONTCONFIG_FILE": str(fontconfig_file(self.workdir))}
        self._pw = sync_playwright().start()
        try:
            self._browser: Browser = self._pw.chromium.launch(env=env)
        except Exception as exc:
            self._pw.stop()
            self._tmp.cleanup()
            raise RenderError(
                f"Could not start headless Chromium ({exc}). "
                "Run `uv run playwright install chromium` and try again."
            ) from exc
        self._page: Page = self._browser.new_page()
        self.stats = RenderStats()
        return self

    def __exit__(self, *exc: object) -> None:
        self._browser.close()
        self._pw.stop()
        self._tmp.cleanup()

    def pdf(self, doc_id: str, html: str) -> bytes:
        path = self.workdir / f"{doc_id}.html"
        path.write_text(html, encoding="utf-8")
        page = self._page
        page.goto(path.as_uri())
        page.emulate_media(media="print")
        faces = page.evaluate(LOAD_FONTS_JS)
        failed = [f for f in faces if f[2] != "loaded"]
        if failed:
            raise RenderError(f"{doc_id}: fonts did not load: {failed}. Check {FONT_DIR}.")
        fit = page.evaluate(FIT_JS, MIN_SHRINK)
        if fit["clipped"]:
            raise RenderError(
                f"{doc_id}: text does not fit its column even at {MIN_SHRINK:.0%} size: "
                f"{fit['clipped']}. Shorten the text or widen the column in the builder."
            )
        if fit["overflow"]:
            raise RenderError(
                f"{doc_id}: content is taller than one A4 page. Drop rows or reduce items."
            )
        self.stats.shrunk_cells += fit["shrunk"]
        data = page.pdf(format="A4", print_background=True, prefer_css_page_size=True)
        pages = len(pdfium.PdfDocument(data))
        if pages != 1:
            raise RenderError(f"{doc_id}: rendered {pages} pages, expected exactly one.")
        return data


def scan_image(pdf: bytes | Path, params: dict[str, float]) -> Image.Image:
    """Rasterise at params["dpi"] in grayscale, rotate, blur, add noise and specks."""
    document = pdfium.PdfDocument(pdf)
    bitmap = document[0].render(scale=params["dpi"] / 72, grayscale=True)
    image = bitmap.to_pil().convert("L")
    image = image.rotate(
        params["rotation_deg"], resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255
    )
    image = image.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    noise = np.random.default_rng(int(params["noise_seed"]))
    pixels = np.asarray(image, dtype=np.float32)
    pixels = pixels + noise.normal(0.0, params["noise_sigma"], pixels.shape)
    specks = noise.random(pixels.shape) < params["speckle_density"]
    pixels[specks] = noise.uniform(0, 90, int(specks.sum()))
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def save_jpeg(image: Image.Image, path: Path, quality: float, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=int(quality), optimize=True, dpi=(dpi, dpi))


def render_outputs(built: list[Built], out_root: Path) -> RenderStats:
    """Write data/docs/<id>.pdf for clean and poisoned documents and <id>.jpg for scans."""
    docs_dir = out_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pdfs: dict[str, bytes] = {}
    with PdfRenderer() as renderer:
        for b in built:
            if b.gold.variant == "scanned":
                continue
            pdfs[b.doc_id] = renderer.pdf(b.doc_id, html_for(b.doc_id, b.doc, b.rows))
            (docs_dir / f"{b.doc_id}.pdf").write_bytes(pdfs[b.doc_id])
        for b in built:
            if b.gold.variant != "scanned" or b.gold.scan is None:
                continue
            base = b.gold.base_doc_id
            if base not in pdfs:
                pdfs[base] = renderer.pdf(base, html_for(base, b.doc, b.doc.rows))
            image = scan_image(pdfs[base], b.gold.scan)
            scan = b.gold.scan
            save_jpeg(image, docs_dir / f"{b.doc_id}.jpg", scan["jpeg_quality"], int(scan["dpi"]))
        return renderer.stats

"""Read back what a rendered PDF shows, from pypdfium2's per-glyph positions.

pdfium's own text assembly (`get_text_range`) is unreliable on Chromium's output for Arabic:
it merges separate rows into one line, reverses some Arabic words and not others, and turns
"INV-2026-0042" inside a right-to-left row into "-2026-0042INV". The glyphs themselves are
correct (poppler reads the same files cleanly), so this module rebuilds the text from them:

  1. group glyphs into lines by baseline,
  2. split a line into cells where the horizontal gap is wider than a column gap,
  3. put each cell into reading order: glyphs arrive in visual (left to right) order, so
     Arabic runs are reversed, and in a right-to-left cell the runs themselves are too,
  4. join the cells in the page's reading order (right to left on Arabic pages).

A cell's direction uses the same rule as the renderer (sheet.is_rtl_text), so the two agree.
"""

from __future__ import annotations

import ctypes
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from data.generator.sheet import CELL_SEP, is_rtl_text, strong_direction

LINE_TOLERANCE_PT = 1.5  # glyphs whose baselines differ by less than this share a line
CELL_GAP_PT = 6.0  # a horizontal gap wider than this starts a new cell (columns are 4 mm)
LAM = "ل"


@dataclass(frozen=True)
class Glyph:
    index: int
    char: str
    x: float  # origin
    y: float  # baseline
    left: float  # advance box: origin to origin plus advance width
    right: float
    word: int  # whitespace characters before this glyph in pdfium's index order
    font: str


@dataclass
class PdfText:
    pages: int
    lines: list[str]
    fonts: set[str]  # font names without the subset prefix, e.g. "NotoSans-Regular"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _font_name(textpage: pdfium.PdfTextPage, index: int, buffer: ctypes.Array) -> str:
    flags = ctypes.c_int()
    size = pdfium_c.FPDFText_GetFontInfo(textpage.raw, index, buffer, len(buffer), flags)
    name = buffer.value.decode("utf-8", "replace") if size else ""
    return name.split("+", 1)[-1]


def read_glyphs(page: pdfium.PdfPage) -> list[Glyph]:
    """Every drawn character except whitespace.

    pdfium places space characters at unreliable x positions on this output (sometimes on
    top of a neighbouring letter), but their place in the index order is right: indices run
    word by word. So each glyph records how many whitespace characters (drawn or inserted by
    pdfium) precede it, and a change of that count between neighbours means a word break.
    Gap widths cannot do this for Arabic: in Noto Naskh the gap after a non-joining letter
    inside a word is as wide as some word spaces.
    """
    textpage = page.get_textpage()
    buffer = ctypes.create_string_buffer(256)
    glyphs = []
    word = 0
    for i in range(textpage.count_chars()):
        code = pdfium_c.FPDFText_GetUnicode(textpage.raw, i)
        if code == 0:
            continue
        if chr(code).isspace():
            word += 1
            continue
        if pdfium_c.FPDFText_IsGenerated(textpage.raw, i):
            continue
        x, y = ctypes.c_double(), ctypes.c_double()
        pdfium_c.FPDFText_GetCharOrigin(textpage.raw, i, x, y)
        box = pdfium_c.FS_RECTF()
        pdfium_c.FPDFText_GetLooseCharBox(textpage.raw, i, box)
        left, right = min(box.left, x.value), max(box.right, x.value)
        font = _font_name(textpage, i, buffer)
        glyphs.append(Glyph(i, chr(code), x.value, y.value, left, right, word, font))
    return glyphs


def _visual_key(glyph: Glyph) -> tuple[float, int]:
    # The two characters of a lam-alef ligature share one origin. In visual order the alef
    # comes first (it is drawn on the left), so the lam sorts after it.
    return (round(glyph.x, 2), 1 if glyph.char == LAM else 0)


def group_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """Top to bottom; each line's glyphs left to right."""
    lines: list[list[Glyph]] = []
    for glyph in sorted(glyphs, key=lambda g: -g.y):
        if lines and abs(lines[-1][0].y - glyph.y) <= LINE_TOLERANCE_PT:
            lines[-1].append(glyph)
        else:
            lines.append([glyph])
    return [sorted(line, key=_visual_key) for line in lines]


def to_visual_cells(line: list[Glyph]) -> list[list[str]]:
    """Split a line at column gaps. Inside a cell, put a space where neighbours belong to
    different words, or where Arabic meets Latin letters or digits (the generator always
    separates those with a space, and pdfium sometimes drops that space)."""
    cells = [[line[0].char]]
    last_strong = strong_direction(line[0].char)
    for prev, glyph in zip(line, line[1:], strict=False):
        if glyph.left - prev.right > CELL_GAP_PT:
            cells.append([glyph.char])
            last_strong = strong_direction(glyph.char)
            continue
        kind = strong_direction(glyph.char)
        script_change = kind != "N" and last_strong not in ("N", kind)
        if (glyph.word != prev.word or script_change) and cells[-1][-1] != " ":
            cells[-1].append(" ")
        cells[-1].append(glyph.char)
        if kind != "N":
            last_strong = kind
    return cells


def _is_digit(ch: str) -> bool:
    return unicodedata.bidirectional(ch) in ("EN", "AN")


def logical_order(chars: list[str]) -> str:
    """Visual left-to-right characters of one cell, back in reading order."""
    rtl = is_rtl_text("".join(chars))
    strong = [strong_direction(c) for c in chars]
    # A percent or plus sign next to a digit belongs to the number (bidi rule W5).
    for i, ch in enumerate(chars):
        if unicodedata.bidirectional(ch) in ("ET", "ES"):
            before = next((c for c in reversed(chars[:i]) if not c.isspace()), "")
            after = next((c for c in chars[i + 1 :] if not c.isspace()), "")
            if _is_digit(before) or _is_digit(after):
                strong[i] = "L"
    resolved = []
    for i, kind in enumerate(strong):
        if kind == "N":
            before = next((k for k in reversed(strong[:i]) if k != "N"), "")
            after = next((k for k in strong[i + 1 :] if k != "N"), "")
            kind = before if before == after and before else ("R" if rtl else "L")
        resolved.append(kind)
    runs: list[tuple[str, list[str]]] = []
    for char, kind in zip(chars, resolved, strict=True):
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(char)
        else:
            runs.append((kind, [char]))
    pieces = ["".join(reversed(cs)) if kind == "R" else "".join(cs) for kind, cs in runs]
    return "".join(reversed(pieces) if rtl else pieces).strip()


def read_pdf(pdf: bytes | Path, rtl_page: bool) -> PdfText:
    document = pdfium.PdfDocument(pdf)
    glyphs = read_glyphs(document[0])
    lines = []
    for line in group_lines(glyphs):
        cells = [logical_order(cell) for cell in to_visual_cells(line)]
        cells = [c for c in cells if c]
        lines.append(CELL_SEP.join(reversed(cells) if rtl_page else cells))
    return PdfText(pages=len(document), lines=lines, fonts={g.font for g in glyphs})

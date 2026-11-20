"""Rasterising documents and running Tesseract in the pinned container.

One container runs a whole batch: a manifest of (image, output base) pairs is mounted with
the repo and fanned out with xargs inside, so the cost of container start is paid once.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pypdfium2 as pdfium

from bench.paths import RASTER, ROOT
from ocr import config


class OcrError(RuntimeError):
    pass


def _docker(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run docker. If the shell predates the docker group change, retry through `sg docker`."""
    proc = subprocess.run(["docker", *args], capture_output=True, text=True)
    if proc.returncode != 0 and "permission denied" in proc.stderr.lower():
        cmd = shlex.join(["docker", *args])
        proc = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise OcrError(f"docker {args[0]} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def ensure_image() -> None:
    proc = _docker(["image", "inspect", config.IMAGE_TAG], check=False)
    if proc.returncode != 0:
        raise OcrError(
            f"Docker image {config.IMAGE_TAG} not found. "
            f"Build it with: docker build -t {config.IMAGE_TAG} ocr/"
        )


def engine_metadata() -> dict:
    """Facts about the engine, read from the container so they cannot drift from the code."""
    ensure_image()
    version = _docker(["run", "--rm", "--entrypoint", "tesseract", config.IMAGE_TAG, "--version"])
    first = (version.stdout or version.stderr).splitlines()[0].strip()
    image_id = _docker(["image", "inspect", "--format", "{{.Id}}", config.IMAGE_TAG]).stdout.strip()
    commit = _docker(
        ["run", "--rm", "--entrypoint", "cat", config.IMAGE_TAG, "/opt/tessdata/COMMIT"]
    ).stdout.strip()
    return {
        "tesseract_version": first.removeprefix("tesseract ").strip(),
        "tessdata": f"tessdata_best@{commit or config.TESSDATA_COMMIT}",
        "tessdata_commit": commit or config.TESSDATA_COMMIT,
        "image": config.IMAGE_TAG,
        "image_id": image_id,
        "base_image": config.BASE_IMAGE,
        "langs": config.LANGS,
        "oem": config.OEM,
        "psm": config.PSM,
        "dpi": config.DPI,
        "config": list(config.EXTRA_CONFIG),
    }


def rasterise(doc_id: str, source: Path, out_dir: Path = RASTER) -> list[Path]:
    """Images for one document. A PDF becomes <doc_id>.png at 300 dpi grayscale (one more
    file, <doc_id>-p2.png and so on, per extra page); a JPEG is used as it is."""
    if source.suffix.lower() != ".pdf":
        return [source]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    pdf = pdfium.PdfDocument(str(source))
    try:
        for n in range(len(pdf)):
            name = f"{doc_id}.png" if n == 0 else f"{doc_id}-p{n + 1}.png"
            image = pdf[n].render(scale=config.DPI / 72, grayscale=True).to_pil()
            image.save(out_dir / name, dpi=(config.DPI, config.DPI))
            paths.append(out_dir / name)
    finally:
        pdf.close()
    return paths


def find_source(doc_id: str) -> Path:
    for ext in (".pdf", ".jpg"):
        path = config.DOCS_DIR / f"{doc_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No document file for {doc_id} in {config.DOCS_DIR}. Run the generator."
    )


def tesseract_args(psm: int) -> str:
    """The flags as one shell fragment, used inside the container loop."""
    extra = " ".join(f"-c {c}" for c in config.EXTRA_CONFIG)
    return f"-l {config.LANGS} --oem {config.OEM} --psm {psm} --dpi {config.DPI} {extra}"


def clean_ocr_text(text: str) -> str:
    """Tesseract ends every page with a form feed and pads lines; drop both so the file is
    plain text. Blank lines inside the text stay (they separate blocks)."""
    lines = [line.rstrip() for line in text.replace("\f", "\n").splitlines()]
    return "\n".join(lines).strip("\n") + "\n"


def run_batch(jobs: list[tuple[Path, Path]], psm: int, jobs_parallel: int | None = None) -> None:
    """OCR every (image, output .txt) pair in one container. Raises OcrError listing the
    outputs that were not produced."""
    ensure_image()
    if not jobs:
        return
    workers = jobs_parallel or max(1, (os.cpu_count() or 2) // 2)
    for _, txt in jobs:
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.unlink(missing_ok=True)
    manifest = RASTER / "_manifest.txt"
    RASTER.mkdir(parents=True, exist_ok=True)
    lines = []
    for image, txt in jobs:
        rel_img = Path(image).resolve().relative_to(ROOT).as_posix()
        rel_out = txt.resolve().with_suffix("").relative_to(ROOT).as_posix()
        lines.append(f"{rel_img} {rel_out}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script = (
        f"cd {config.CONTAINER_ROOT} && xargs -a {manifest.relative_to(ROOT).as_posix()} "
        f"-n 2 -P {workers} sh -c 'tesseract \"$0\" \"$1\" {tesseract_args(psm)} 2>/dev/null'"
    )
    cmd = [
        "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "OMP_THREAD_LIMIT=1",
        "-v", f"{ROOT}:{config.CONTAINER_ROOT}",
        "--entrypoint", "sh", config.IMAGE_TAG, "-c", script,
    ]  # fmt: skip
    proc = _docker(cmd, check=False)
    missing = [txt.name for _, txt in jobs if not txt.exists()]
    if missing:
        raise OcrError(
            f"Tesseract produced no output for {len(missing)} file(s), first: {missing[:3]}. "
            f"Container said: {proc.stderr.strip()[:300] or 'nothing'}"
        )


def run_documents(doc_ids: list[str], psm: int, out_dir: Path) -> None:
    """Rasterise and OCR the documents; write <out_dir>/<doc_id>.txt (pages joined)."""
    jobs: list[tuple[Path, Path]] = []
    page_txts: dict[str, list[Path]] = {}
    scratch = RASTER / "_pages"
    for doc_id in doc_ids:
        images = rasterise(doc_id, find_source(doc_id))
        if len(images) == 1:
            page_txts[doc_id] = [out_dir / f"{doc_id}.txt"]
        else:
            page_txts[doc_id] = [scratch / f"{doc_id}-p{n + 1}.txt" for n in range(len(images))]
        jobs.extend(zip(images, page_txts[doc_id], strict=True))
    run_batch(jobs, psm)
    for doc_id, txts in page_txts.items():
        text = "\n".join(clean_ocr_text(t.read_text(encoding="utf-8")) for t in txts)
        (out_dir / f"{doc_id}.txt").write_text(clean_ocr_text(text), encoding="utf-8")


def write_engine_file(meta: dict, path: Path = config.ENGINE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

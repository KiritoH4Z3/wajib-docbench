"""python -m ocr run | score | pilot"""

from __future__ import annotations

import json

import typer

from bench.inputs import list_doc_ids, load_gold
from bench.paths import OCR_SCORES, OCR_TEXT, ocr_path
from ocr import config, engine, scoring

app = typer.Typer(add_completion=False, help="The fixed OCR step (Tesseract in Docker).")


def _doc_ids_from_docs() -> list[str]:
    ids = sorted(p.stem for p in config.DOCS_DIR.glob("*") if p.suffix in (".pdf", ".jpg"))
    if not ids:
        raise typer.BadParameter(f"No documents in {config.DOCS_DIR}. Run the generator first.")
    return ids


DOC_OPTION = typer.Option(None, "--doc", help="Only these document ids.")
PSM_OPTION = typer.Option(list(config.PILOT_PSMS), "--psm", help="Modes to compare.")


@app.command()
def run(doc: list[str] = DOC_OPTION) -> None:
    """OCR every document in data/docs to data/ocr/<doc_id>.txt with the chosen psm."""
    ids = doc or _doc_ids_from_docs()
    meta = engine.engine_metadata()
    engine.run_documents(ids, config.PSM, OCR_TEXT)
    if not doc:  # a partial run must not overwrite the metadata of a full one
        engine.write_engine_file(meta)
    typer.echo(
        f"OCR done for {len(ids)} documents, psm {config.PSM}, "
        f"tesseract {meta['tesseract_version']}."
    )


@app.command()
def score() -> None:
    """Write data/ocr/_scores.json from data/ocr/*.txt and the ground truth."""
    if not config.ENGINE_FILE.exists():
        raise typer.BadParameter("data/ocr/_engine.json is missing. Run `python -m ocr run` first.")
    engine_meta = json.loads(config.ENGINE_FILE.read_text(encoding="utf-8"))
    ids = list_doc_ids()
    if not ids:
        raise typer.BadParameter("No ground truth in data/ground_truth. Run the generator first.")
    missing = [i for i in ids if not ocr_path(i).exists()]
    if missing:
        raise typer.BadParameter(
            f"No OCR text for {len(missing)} documents (first: {missing[:3]}). "
            "Run `python -m ocr run`."
        )
    pairs = [(load_gold(i), ocr_path(i).read_text(encoding="utf-8")) for i in ids]
    payload = scoring.build_scores(pairs, engine_meta)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    OCR_SCORES.write_text(text + "\n", encoding="utf-8")
    for name, block in payload["by_layout"].items():
        typer.echo(f"{name}: n={block['n']} cer_norm={block['cer_norm']}")
    typer.echo(f"Wrote {OCR_SCORES}")


@app.command()
def pilot(psm: list[int] = PSM_OPTION) -> None:
    """Compare page segmentation modes on the base documents and record the winner."""
    ids = list_doc_ids("clean")
    if not ids:
        raise typer.BadParameter(
            "No clean documents in data/ground_truth. Run the generator first."
        )
    results: dict[int, dict] = {}
    for mode in psm:
        out_dir = config.PILOT_WORK / f"psm{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        engine.run_documents(ids, mode, out_dir)
        pairs = [(load_gold(i), (out_dir / f"{i}.txt").read_text(encoding="utf-8")) for i in ids]
        results[mode] = scoring.pilot_metrics(pairs)
        typer.echo(f"psm {mode}: {results[mode]}")
    chosen = scoring.select_psm(results)
    payload = {
        "selected_psm": chosen,
        "rule": "highest field grounding recall; lower normalised CER breaks ties",
        "documents": "base (variant clean)",
        "per_psm": {str(k): v for k, v in results.items()},
    }
    config.PILOT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    note = ""
    if chosen != config.PSM:
        note = f" Set PSM = {chosen} in ocr/config.py, then run `python -m ocr run`."
    typer.echo(f"Selected psm {chosen}.{note}")


if __name__ == "__main__":
    app()

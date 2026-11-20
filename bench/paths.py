"""Every file location in one place. Paths are relative to the repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "config"
MODELS_YAML = CONFIG / "models.yaml"
PRICING_YAML = CONFIG / "pricing.yaml"

DATA = ROOT / "data"
DOCS = DATA / "docs"  # <doc_id>.pdf for clean and poisoned, <doc_id>.jpg for scanned
GROUND_TRUTH = DATA / "ground_truth"  # <doc_id>.json (bench.records.GoldDoc)
OCR_TEXT = DATA / "ocr"  # <doc_id>.txt, the fixed OCR output every model reads
OCR_SCORES = DATA / "ocr" / "_scores.json"  # character error rates per document
RASTER = DATA / "raster"  # <doc_id>.png at 300 dpi, regenerated, not committed

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"  # <model_slug>/<text_source>/{extract,guard}/<doc_id>.json
CACHE = RESULTS / "cache"  # <model_slug>/<sha256>.json, committed so API runs replay
PERF = RESULTS / "perf"  # <model_slug>/requests.jsonl and load-test summaries
RESULTS_JSON = RESULTS / "results.json"
CHARTS = RESULTS / "charts"


def run_dir(model_slug: str, text_source: str, stage: str) -> Path:
    return RUNS / model_slug / text_source / stage


def gold_path(doc_id: str) -> Path:
    return GROUND_TRUTH / f"{doc_id}.json"


def ocr_path(doc_id: str) -> Path:
    return OCR_TEXT / f"{doc_id}.txt"

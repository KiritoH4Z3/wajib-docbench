"""Assemble the text_sources[source] block of CONTRACTS section 6 for one model.

`score_model` reads the run records, the guardrail records, the input text and the OCR
scores from disk and hands lists of DocResult to `score_results`, which is pure and is
what the tests exercise.
"""

from __future__ import annotations

import json
from typing import Any

from bench.inputs import list_doc_ids, load_gold, load_input_text
from bench.paths import OCR_SCORES, run_dir
from bench.records import HARD_CASES, ExtractionRecord, GuardrailRecord, TextSource, read_model
from eval import metrics
from eval.consequences import licence_consequences
from eval.metrics import DocResult


def _load_results(model_slug: str, text_source: TextSource, doc_ids: list[str]) -> list[DocResult]:
    extract_dir = run_dir(model_slug, text_source, "extract")
    guard_dir = run_dir(model_slug, text_source, "guard")
    results = []
    for doc_id in doc_ids:
        record_path = extract_dir / f"{doc_id}.json"
        guard_path = guard_dir / f"{doc_id}.json"
        results.append(
            DocResult(
                gold=load_gold(doc_id),
                record=read_model(record_path, ExtractionRecord) if record_path.exists() else None,
                guard=read_model(guard_path, GuardrailRecord) if guard_path.exists() else None,
                input_text=load_input_text(doc_id, text_source),
            )
        )
    return results


def _load_injection_in_ocr() -> dict[str, bool] | None:
    """data/ocr/_scores.json -> {doc_id: injection survived OCR}; None when absent."""
    if not OCR_SCORES.exists():
        return None
    data = json.loads(OCR_SCORES.read_text(encoding="utf-8"))
    found = data.get("injection_in_ocr")
    return found if isinstance(found, dict) else None


def score_results(
    base: list[DocResult],
    scanned_results: list[DocResult],
    poison_results: list[DocResult],
    injection_in_ocr: dict[str, bool] | None,
) -> dict[str, Any]:
    clean_by_id = {r.gold.doc_id: r for r in base}
    halluc = metrics.hallucination(base)
    return {
        "n_docs": len(base),
        "missing": metrics.missing_counts(base),
        "missing_variants": {
            "scanned": metrics.missing_counts(scanned_results),
            "poisoned": metrics.missing_counts(poison_results),
        },
        "field_f1": metrics.field_f1(base),
        "critical_exact_match": metrics.critical_exact_match(base),
        "hallucination_rate": halluc["rate"],
        "hallucination": halluc,
        "schema_valid_first_try": metrics.schema_valid_first_try(base),
        "poisoned": metrics.poisoned(poison_results, clean_by_id, injection_in_ocr),
        "review": metrics.review(base),
        "attribution": metrics.attribution(base),
        "consequences": licence_consequences(base),
        "scanned": metrics.scanned(scanned_results, clean_by_id),
    }


def score_model(model_slug: str, text_source: TextSource) -> dict:
    base = _load_results(model_slug, text_source, list_doc_ids("clean"))
    scanned_results = _load_results(model_slug, text_source, list_doc_ids("scanned"))
    poison_results = _load_results(model_slug, text_source, list_doc_ids("poisoned"))
    if text_source == "ocr":
        reached = _load_injection_in_ocr()
    else:
        # The oracle text carries every injection, hidden ones included.
        reached = {r.gold.doc_id: True for r in poison_results}
    return score_results(base, scanned_results, poison_results, reached)


def score_dataset() -> dict:
    """The "dataset" block: document counts per variant and hard-case counts over the
    base documents."""
    golds = {i: load_gold(i) for i in list_doc_ids()}
    by_variant = {"clean": 0, "scanned": 0, "poisoned": 0}
    counts = dict.fromkeys(HARD_CASES, 0)
    for g in golds.values():
        by_variant[g.variant] += 1
        if g.variant == "clean":
            for tag in g.hard_cases:
                counts[tag] = counts.get(tag, 0) + 1
    return {
        "base": by_variant["clean"],
        "scanned": by_variant["scanned"],
        "poisoned": by_variant["poisoned"],
        "hard_case_counts": counts,
    }

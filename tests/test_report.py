"""The README results block is rendered, never typed, so these pin what it may and may not say.

Silent failures pinned here:
- an incomplete run (missing records or API errors) rendered as a scored row, so an outage
  reads as a model with an F1 of zero
- the injection column shown as a bare percentage, hiding that it rests on a handful of
  documents
- the README block edited by hand or left stale while results.json still matches, which
  `report --check` used to miss
"""

from __future__ import annotations

import json

import pytest

from bench import report


def _block(**over) -> dict:
    ocr = {
        "n_docs": 60,
        "missing": {"no_record": 0, "api_error": 0, "invalid_after_retry": 1},
        "field_f1": {"ar_bi": 0.71, "en": 0.93, "ar": 0.55, "bi": 0.86, "arabic_script": 0.62},
        "ci95": {"field_f1.ar_bi": [0.65, 0.77], "field_f1.en": [0.9, 0.96]},
        "critical_exact_match": 0.8,
        "hallucination_rate": 0.01,
        "schema_valid_first_try": 0.97,
        "poisoned": {"n_reached": 8, "k": {"obeyed": 1, "strict_reached": 5}},
        "review": {"rate": 0.3, "precision": 0.9, "recall": 0.7},
        "consequences": {},
        "attribution": {},
        "sample_variance": None,
    }
    ocr.update(over)
    return {
        "display_name": "Model A",
        "kind": "local",
        "quant": "Q4_K_M (4.60 GB)",
        "text_sources": {"ocr": ocr},
        "perf": {"throughput": {"4": {"ttft_p95_s": 0.9}}},
        "cost": {"per_1000_docs_usd": 0.4},
    }


def test_complete_row_shows_intervals_and_counts():
    text = report.render_block({"models": {"a": _block()}, "not_run": []})
    row = next(line for line in text.splitlines() if line.startswith("| Model A"))
    assert "0.71 (0.65 to 0.77)" in row
    assert "| 1/8 |" in row and "| 5/8 |" in row
    assert "incomplete" not in row


@pytest.mark.parametrize(
    "missing", [{"no_record": 2, "api_error": 0}, {"no_record": 0, "api_error": 3}]
)
def test_incomplete_run_is_never_rendered_as_a_score(missing):
    text = report.render_block({"models": {"a": _block(missing=missing)}, "not_run": []})
    row = next(line for line in text.splitlines() if line.startswith("| Model A"))
    assert "incomplete" in row
    assert "0.71" not in row and "0.93" not in row


def test_api_row_without_samples_says_single_draw():
    m = _block()
    m["kind"] = "api"
    m["quant"] = None
    text = report.render_block({"models": {"a": m}, "not_run": []})
    assert "Model A, single draw" in text


def test_check_fails_when_the_readme_block_is_stale(tmp_path, monkeypatch):
    results = {"models": {}, "not_run": [], "dataset": {}, "ocr": None, "comparisons": []}
    monkeypatch.setattr(report, "build_results", lambda: results)
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"generated_at": "x", "git_commit": "y", **results}))
    monkeypatch.setattr(report, "RESULTS_JSON", results_json)
    readme = tmp_path / "README.md"
    fresh = report.render_block(results)
    readme.write_text(f"intro\n{report.START}\n{fresh}{report.END}\nrest\n")
    monkeypatch.setattr(report, "README", readme)
    assert report.write_report(check=True) is True
    readme.write_text(f"intro\n{report.START}\n| hand-typed 0.99 |\n{report.END}\nrest\n")
    assert report.write_report(check=True) is False

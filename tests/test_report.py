"""The README results block is rendered, never typed, so these pin what it may and may not say.

Silent failures pinned here:
- an incomplete run (missing records or API errors) rendered as a scored row, so an outage
  reads as a model with an F1 of zero
- the injection column shown as a bare percentage, hiding that it rests on a handful of
  documents
- the README block edited by hand or left stale while results.json still matches, which
  `report --check` used to miss
- a scope target shown as met or missed when its interval straddles the limit, or without
  the number of documents it rests on
- the concurrency-16 level read as 16 parallel slots when the GPU only held 8, or an API
  level that was never run shown as a blank instead of a reason
"""

from __future__ import annotations

import json

import pytest

from bench import report
from eval.targets import evaluate_targets


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


def _targets_row(text: str, label: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {label} |"))


def test_targets_table_gives_the_verdict_the_interval_and_the_count():
    m = _block(
        n_docs=60,
        field_f1={"ar_bi": 0.71, "en": 0.93, "n_docs": {"en": 20, "ar_bi": 40}},
        hallucination_rate=0.01,
        ci95={"field_f1.ar_bi": [0.65, 0.77], "field_f1.en": [0.9, 0.96],
              "hallucination_rate": [0.0, 0.03]},
        poisoned={"strict_reached": 0.625, "n_reached": 8, "k": {"strict_reached": 5}},
    )
    perf = {"throughput": {"4": {"ttft_p95_s": 0.9, "n_requests": 60}}}
    m["targets"] = evaluate_targets(m["text_sources"]["ocr"], perf)
    text = report.render_block({"models": {"a": m}, "not_run": []})
    assert _targets_row(text, "F1, English docs, at least 0.90").endswith(
        "| met: 0.93 (0.90 to 0.96), 20 docs |")
    assert _targets_row(text, "F1, Arabic and bilingual docs, at least 0.85").endswith(
        "| missed: 0.71 (0.65 to 0.77), 40 docs |")
    assert _targets_row(text, "Hallucinated values, at most 2%").endswith(
        "| too close to call: 1% (0% to 3%), 60 docs |")
    assert _targets_row(text, "Poisoned docs fully correct, at least 90%").endswith(
        "| missed: 5/8 |")
    assert _targets_row(text, "TTFT p95 at 4 in flight, under 1.50 s").endswith(
        "| met: 0.90 s, 60 requests |")


def test_targets_table_never_scores_an_incomplete_run():
    m = _block(missing={"no_record": 2, "api_error": 0})
    m["targets"] = evaluate_targets(m["text_sources"]["ocr"], None)
    text = report.render_block({"models": {"a": m}, "not_run": []})
    row = _targets_row(text, "F1, English docs, at least 0.90")
    assert row.endswith("| not scored (incomplete run) |")
    assert _targets_row(text, "TTFT p95 at 4 in flight, under 1.50 s").endswith(
        "| not measured |")


def test_throughput_table_names_the_slots_and_never_blanks_a_skipped_level():
    local = _block()
    local["perf"] = {"throughput": {
        "1": {"docs_per_min": 13.5, "ttft_p95_s": 1.32, "api_errors": 0, "invalid_outputs": 0},
        "4": {"docs_per_min": 20.3, "ttft_p95_s": 4.57, "api_errors": 0, "invalid_outputs": 1},
        "16": {"docs_per_min": 16.7, "ttft_p95_s": 37.97, "api_errors": 2, "invalid_outputs": 0},
    }}
    local["server"] = {"parallel_slots": 8, "shared_ctx": 12288}
    api = _block()
    api.update(display_name="Model B", kind="api", quant=None, server=None)
    api["perf"] = {"throughput": {
        "1": {"docs_per_min": 6.9, "ttft_p95_s": 8.39},
        "4": {"skipped": "concurrency 4 is above max_concurrency 1"},
        "16": {"skipped": "concurrency 16 is above max_concurrency 1"},
    }}
    api["targets"] = evaluate_targets(api["text_sources"]["ocr"], api["perf"])
    text = report.render_block({"models": {"a": local, "b": api}, "not_run": []})
    row = next(line for line in text.splitlines() if line.startswith("| Model A | 8"))
    assert row == ("| Model A | 8, sharing a 12,288-token cache | 13.5, 20.3, 16.7 "
                   "| 1.32 s, 4.57 s, 37.97 s | 3 |")
    assert "| Model B | 1 (API) | 6.9, n/a, n/a | 8.39 s, n/a, n/a | 0 |" in text
    assert _targets_row(text, "TTFT p95 at 4 in flight, under 1.50 s").endswith(
        "| not measured: the free tier takes one request at a time |")

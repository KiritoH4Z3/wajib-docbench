"""Silent failures these tests catch in the perf module.

- Cached attempts replay old timings; if they leaked into latency stats, a rerun would report
  the original machine's speed as a fresh measurement.
- An empty input turning into 0.0 would read as "instant", so empty must be None.
- Cost arithmetic that forgets thinking tokens understates the API baseline.
- A throughput run that lets a model go above its max concurrency (the free-tier API) would
  hit rate limits and publish a wall clock full of retries.
- An AIPerf export read with the wrong key silently yields None everywhere.
"""

import asyncio
import json

import pytest

from bench.config import ModelConfig
from bench.records import Attempt, ExtractionRecord
from perf.chart import accuracy_vs_cost_chart
from perf.cost import cost_per_1000
from perf.stats import latency_stats


def _attempt(ttft=1.0, itl=(0.1,), total=2.0, prompt=100, completion=50, cached=False,
             estimated=False) -> Attempt:
    return Attempt(raw="{}", parsed=None, schema_valid=True, prompt_tokens=prompt,
                   completion_tokens=completion, usage_estimated=estimated, ttft_s=ttft,
                   itl_s=list(itl), total_s=total, started_at="2026-11-20T00:00:00Z",
                   cached=cached)


def _record(*attempts: Attempt, api_error=None, final=None) -> ExtractionRecord:
    return ExtractionRecord(doc_id="d", model_slug="m", text_source="ocr", prompt_version="v1",
                            attempts=list(attempts), final=final, schema_valid_first_try=True,
                            api_error=api_error)


def _model(kind="local", max_concurrency=16, slug="m") -> ModelConfig:
    return ModelConfig(slug=slug, display_name="M", kind=kind,
                       served_by="api" if kind == "api" else "llamacpp", vendor="v",
                       licence="l", licence_url="u", base_url="http://x/v1", model="m",
                       max_concurrency=max_concurrency)


# ---------------------------------------------------------------- stats

def test_percentiles_use_linear_interpolation():
    recs = [_record(_attempt(ttft=float(t), total=float(t) * 2)) for t in (1, 2, 3, 4, 5)]
    s = latency_stats(recs)
    assert s["ttft_p50_s"] == 3.0
    assert s["ttft_p95_s"] == pytest.approx(4.8)  # 4 + 0.8 * (5 - 4)
    assert s["total_p50_s"] == 6.0
    assert s["n_requests"] == 5


def test_itl_gaps_are_pooled_across_attempts():
    recs = [_record(_attempt(itl=(0.1, 0.2))), _record(_attempt(itl=(0.3, 0.4)))]
    assert latency_stats(recs)["itl_p50_s"] == pytest.approx(0.25)


def test_cached_attempts_are_excluded():
    recs = [_record(_attempt(ttft=1.0), _attempt(ttft=99.0, cached=True))]
    s = latency_stats(recs)
    assert s["n_requests"] == 1
    assert s["ttft_p95_s"] == 1.0


def test_means_and_estimated_usage_count():
    recs = [_record(_attempt(prompt=100, completion=10, estimated=True)),
            _record(_attempt(prompt=300, completion=30))]
    s = latency_stats(recs)
    assert s["mean_prompt_tokens"] == 200
    assert s["mean_completion_tokens"] == 20
    assert s["n_usage_estimated"] == 1


def test_empty_input_gives_none_not_zero():
    for recs in ([], [_record(_attempt(cached=True))]):
        s = latency_stats(recs)
        assert s["ttft_p50_s"] is None and s["itl_p95_s"] is None
        assert s["mean_prompt_tokens"] is None
        assert s["n_requests"] == 0


# ---------------------------------------------------------------- cost

PRICING = {
    "api": {"m": {"input_usd_per_mtok": 0.30, "output_usd_per_mtok": 2.50, "tier": "list",
                  "source": "https://example.test/p", "fetched": "2026-11-20"}},
    "gpu": {"primary": {"name": "T4", "usd_per_hour": 0.60, "source": "s", "fetched": "d"},
            "sensitivity": [{"name": "L4", "usd_per_hour": 1.20, "source": "s2", "fetched": "d"}]},
}


def test_api_cost_includes_completion_tokens_priced_as_output():
    perf = {"mean_prompt_tokens": 1000, "mean_completion_tokens": 400}
    out = cost_per_1000(_model("api", 1), perf, PRICING)
    # (1000 * 0.30 + 400 * 2.50) / 1e6 * 1000 = 1.30
    assert out["per_1000_docs_usd"] == pytest.approx(1.30)
    assert "0.3" in out["basis"] and "2026-11-20" in out["basis"]


def test_self_hosted_cost_uses_best_concurrency_and_sensitivity():
    perf = {"throughput": {"1": {"docs_per_min": 5.0}, "4": {"docs_per_min": 10.0},
                           "16": {"docs_per_min": 8.0}}}
    out = cost_per_1000(_model(), perf, PRICING)
    # 10 docs/min = 600 docs/hour; 0.60 / 600 * 1000 = 1.00
    assert out["per_1000_docs_usd"] == pytest.approx(1.0)
    assert out["sensitivity"][0]["per_1000_docs_usd"] == pytest.approx(2.0)
    assert "concurrency 4" in out["basis"] and "$0.6/hour" in out["basis"]


def test_self_hosted_cost_accepts_flat_contract_map():
    out = cost_per_1000(_model(), {"throughput_docs_per_min": {"1": 6.0, "4": 12.0}}, PRICING)
    assert out["per_1000_docs_usd"] == pytest.approx(0.6 / 720 * 1000)


@pytest.mark.parametrize("kind,perf", [("api", {}), ("local", {}), ("local", {"throughput": {}})])
def test_missing_inputs_return_none_with_reason(kind, perf):
    out = cost_per_1000(_model(kind), perf, PRICING)
    assert out["per_1000_docs_usd"] is None and out["basis"]


# ---------------------------------------------------------------- throughput

def test_throughput_with_fake_runner_and_clock(monkeypatch, tmp_path):
    import extract.runner as runner_mod  # noqa: F401  (replaced below)
    from perf import throughput

    calls = []

    async def fake_run(model, doc_ids, text_source, concurrency=1, use_cache=True,
                       fresh=False, write=True):
        calls.append((concurrency, use_cache, fresh, write))
        return [_record(_attempt(), final={"a": 1}) for _ in doc_ids]

    monkeypatch.setattr(runner_mod, "run_extraction", fake_run)
    monkeypatch.setattr(throughput, "PERF", tmp_path)
    ticks = iter([0.0, 30.0, 100.0, 115.0])  # c=1 takes 30 s, c=4 takes 15 s
    out = asyncio.run(throughput.measure_throughput(
        _model(max_concurrency=4), ["a"] * 30, (1, 4, 16), clock=lambda: next(ticks)))
    assert calls == [(1, False, True, False), (4, False, True, False)]
    assert out["levels"]["1"]["docs_per_min"] == pytest.approx(60.0)
    assert out["levels"]["4"]["docs_per_min"] == pytest.approx(120.0)
    assert out["levels"]["4"]["n_requests"] == 30
    assert "above max_concurrency 4" in out["levels"]["16"]["skipped"]
    saved = json.loads((tmp_path / "m" / "throughput.json").read_text())
    assert saved["levels"]["1"]["wall_s"] == 30.0


def test_throughput_counts_failed_documents(monkeypatch, tmp_path):
    import extract.runner as runner_mod
    from perf import throughput

    async def fake_run(model, doc_ids, text_source, **kw):
        return [_record(_attempt(), final={"a": 1}), _record(api_error="rate_limit")]

    monkeypatch.setattr(runner_mod, "run_extraction", fake_run)
    out = asyncio.run(throughput.measure_throughput(
        _model(), ["a", "b"], (1,), clock=iter([0.0, 6.0]).__next__, write=False))
    assert out["levels"]["1"]["errors"] == 1


# ---------------------------------------------------------------- chart

def test_chart_writes_svg_and_png_without_generator_metadata(tmp_path):
    pts = [{"label": "A", "arabic_f1": 0.7, "cost_usd": 0.5, "kind": "local"},
           {"label": "B", "arabic_f1": 0.9, "cost_usd": 20.0, "kind": "api"},
           {"label": "C", "arabic_f1": None, "cost_usd": None, "kind": "local"}]
    paths = accuracy_vs_cost_chart(pts, tmp_path)
    assert {p.name for p in paths} == {"accuracy_vs_cost.svg", "accuracy_vs_cost.png"}
    assert all(p.stat().st_size > 1000 for p in paths)
    svg = (tmp_path / "accuracy_vs_cost.svg").read_text()
    for marker in ("matplotlib.org", "Matplotlib v", "<dc:creator", "<dc:date"):
        assert marker not in svg
    assert b"tEXt" not in (tmp_path / "accuracy_vs_cost.png").read_bytes()
    assert "Not plotted, no measurement: C" in svg


# ---------------------------------------------------------------- aiperf

def _metric(unit, **kw):
    base = {k: None for k in ("avg", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95",
                               "p99", "min", "max", "std", "count", "sum")}
    return {"unit": unit, **base, **kw}


def _export(scale):
    """The shape of aiperf 0.13.0's profile_export_aiperf.json (JsonExportData): metric
    tags are top-level keys, latencies are in ms, throughput in requests/sec."""
    return {
        "schema_version": "1.4", "aiperf_version": "0.13.0", "was_cancelled": False,
        "request_throughput": _metric("requests/sec", avg=0.5 * scale),
        "request_latency": _metric("ms", avg=2100.0, p50=2000.0, p95=3500.0, count=120),
        "time_to_first_token": _metric("ms", avg=400.0, p50=350.0, p95=900.0, count=120),
        "inter_token_latency": _metric("ms", avg=30.0, p50=28.0, p95=45.0, count=120),
        "output_token_throughput": _metric("tokens/sec", avg=60.0 * scale),
    }


@pytest.fixture
def FIXTURE(tmp_path):
    for n, scale in ((1, 1), (4, 3)):
        d = tmp_path / f"c{n}"
        d.mkdir()
        (d / "profile_export_aiperf.json").write_text(json.dumps(_export(scale)))
    return tmp_path


def test_parse_aiperf_export_converts_ms_to_seconds(FIXTURE):
    from perf.aiperf import parse_artifacts

    out = parse_artifacts(FIXTURE)
    assert set(out["levels"]) == {"1", "4"}
    c1 = out["levels"]["1"]
    assert c1["ttft"]["p50"] == pytest.approx(0.35)
    assert c1["ttft"]["p95"] == pytest.approx(0.9)
    assert c1["itl"]["p50"] == pytest.approx(0.028)
    assert c1["request_latency"]["p95"] == pytest.approx(3.5)
    assert c1["requests_per_s"]["avg"] == pytest.approx(0.5)
    assert out["levels"]["4"]["output_tokens_per_s"]["avg"] == pytest.approx(180.0)


def test_parse_aiperf_reports_missing_level_and_empty_dir(tmp_path):
    from perf.aiperf import parse_artifacts

    (tmp_path / "c16").mkdir()
    assert parse_artifacts(tmp_path)["missing"] == ["16"]
    with pytest.raises(FileNotFoundError):
        parse_artifacts(tmp_path / "nowhere")


def test_build_inputs_writes_one_prompt_per_base_doc(monkeypatch, tmp_path):
    import perf.aiperf as ap

    class Gold:
        doc_type = "tax_invoice"

    monkeypatch.setattr(ap, "list_doc_ids", lambda variant=None: ["a", "b"])
    monkeypatch.setattr(ap, "load_gold", lambda d: Gold())
    monkeypatch.setattr(ap, "load_input_text", lambda d, s: f"text {d} ا")
    out = ap.build_inputs("m", tmp_path)
    rows = [json.loads(x) for x in (out / "prompts.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 2 and "text a" in rows[0]["text"] and "<document>" in rows[0]["text"]
    assert (out / "system_prompt.txt").read_text("utf-8").strip()

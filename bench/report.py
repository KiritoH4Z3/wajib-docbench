"""Assemble results/results.json, the chart and the README results block from run records.

Nothing in the README results block is typed by hand: it is rendered from results.json,
which is rebuilt from the committed run records. `--check` rebuilds it and fails if the
committed file differs, so a number cannot drift away from the evidence behind it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from typing import Any

from bench.config import ModelConfig, load_models, load_pricing
from bench.paths import CHARTS, OCR_SCORES, PERF, RESULTS_JSON, ROOT, run_dir
from bench.records import ExtractionRecord, read_model

README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
VOLATILE = ("generated_at", "git_commit")


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _has_records(slug: str, source: str) -> bool:
    d = run_dir(slug, source, "extract")
    return d.exists() and any(d.glob("*.json"))


def _extraction_latency(slug: str) -> dict:
    """Per-request latency and per-document token means of the main extraction run over the
    60 base documents. Replayed responses keep the timings measured when they were first
    produced, so they count here."""
    from bench.inputs import list_doc_ids
    from perf.stats import latency_stats

    records = [
        read_model(p, ExtractionRecord)
        for p in sorted(run_dir(slug, "ocr", "extract").glob("*.json"))
    ]
    as_measured = [
        r.model_copy(
            update={"attempts": [a.model_copy(update={"cached": False}) for a in r.attempts]}
        )
        for r in records
    ]
    return latency_stats(as_measured, doc_ids=list_doc_ids("clean"))


def _server_info(slug: str, slots: int | None) -> dict | None:
    """The server record for the slot count the throughput run used, else the most recent."""
    wanted = PERF / slug / f"server-np{slots}.json" if slots else None
    if wanted is not None and wanted.exists():
        path = wanted
    else:
        files = sorted((PERF / slug).glob("server-np*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            return None
        path = files[-1]
    info = json.loads(path.read_text(encoding="utf-8"))
    info.pop("server_log_tail", None)
    return info


def _model_block(cfg: ModelConfig, pricing: dict) -> dict | None:
    from eval.score import score_model
    from eval.targets import evaluate_targets
    from perf.cost import cost_per_1000

    sources = {s: score_model(cfg.slug, s) for s in ("ocr", "oracle") if _has_records(cfg.slug, s)}
    if not sources:
        return None
    throughput_file = PERF / cfg.slug / "throughput.json"
    throughput = (
        json.loads(throughput_file.read_text(encoding="utf-8")) if throughput_file.exists() else {}
    )
    levels = throughput.get("levels", {})
    perf = {"throughput": levels, **(_extraction_latency(cfg.slug) if "ocr" in sources else {})}
    return {
        "display_name": cfg.display_name,
        "vendor": cfg.vendor,
        "kind": cfg.kind,
        "served_by": cfg.served_by,
        "quant": cfg.quant,
        "licence": cfg.licence,
        "hf_repo": cfg.hf_repo,
        "status_note": cfg.status_note.strip(),
        "text_sources": sources,
        "perf": perf,
        "cost": cost_per_1000(cfg, perf, pricing),
        "targets": evaluate_targets(sources.get("ocr", {}), perf),
        "server": _server_info(cfg.slug, throughput.get("parallel_slots")),
    }


def _incomplete(block: dict) -> str | None:
    """Why a text-source block cannot be read as a model result, or None when it can."""
    missing = block.get("missing") or {}
    no_record, api_error = missing.get("no_record") or 0, missing.get("api_error") or 0
    if not no_record and not api_error:
        return None
    n = block.get("n_docs")
    return f"incomplete: {no_record} of {n} documents missing, {api_error} API errors"


def _comparisons(models: dict) -> list[dict]:
    """Each open model minus the API baseline, paired on the same documents (OCR source)."""
    from eval.uncertainty import paired_difference

    api = [s for s, m in models.items() if m["kind"] == "api" and "ocr" in m["text_sources"]]
    out = []
    for base in api:
        for slug, m in models.items():
            if m["kind"] == "api" or "ocr" not in m["text_sources"]:
                continue
            if _incomplete(m["text_sources"]["ocr"]) or _incomplete(
                models[base]["text_sources"]["ocr"]
            ):
                continue
            for metric in ("field_f1.ar_bi", "field_f1.en", "critical_exact_match"):
                out.append({"a": slug, "b": base, **paired_difference(slug, base, "ocr", metric)})
    return out


def build_results() -> dict:
    from eval.score import score_dataset

    pricing = load_pricing()
    ocr = json.loads(OCR_SCORES.read_text(encoding="utf-8")) if OCR_SCORES.exists() else None
    if ocr:
        ocr = {k: v for k, v in ocr.items() if k != "documents"}
    models: dict[str, Any] = {}
    not_run: list[dict] = []
    for cfg in load_models(include_inactive=True):
        block = _model_block(cfg, pricing) if cfg.status == "active" else None
        if block is None:
            not_run.append(
                {
                    "slug": cfg.slug,
                    "display_name": cfg.display_name,
                    "status": cfg.status if cfg.status != "active" else "not run yet",
                    "note": cfg.status_note.strip(),
                }
            )
            continue
        models[cfg.slug] = block
    return _round(
        {
            "dataset": score_dataset(),
            "ocr": ocr,
            "models": models,
            "comparisons": _comparisons(models),
            "not_run": not_run,
            "pricing": pricing,
        }
    )


def _stamp(results: dict) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit or None,
        **results,
    }


def _strip(results: dict) -> dict:
    return {k: v for k, v in results.items() if k not in VOLATILE}


# ---------------------------------------------------------------- README rendering


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _f(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.2f}"


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _ci(x: list[float] | None, pct: bool = False) -> str:
    if not x:
        return ""
    lo, hi = x
    return f" ({_pct(lo, 0)} to {_pct(hi, 0)})" if pct else f" ({lo:.2f} to {hi:.2f})"


def _kn(k: int | None, n: int | None) -> str:
    return "n/a" if k is None or not n else f"{k}/{n}"


def _row(m: dict) -> str:
    o = _get(m, "text_sources", "ocr") or {}
    quant = f" ({m['quant'].split(' ')[0]})" if m.get("quant") else ""
    name = f"{m['display_name']}{quant}"
    if m["kind"] == "api" and not o.get("sample_variance"):
        name += ", single draw"
    why = _incomplete(o)
    if why:
        return f"| {name} | {why}; rerun the extraction before reading this row |" + " |" * 8
    ci = o.get("ci95") or {}
    review = o.get("review") or {}
    p = o.get("poisoned") or {}
    ttft = _get(m, "perf", "throughput", "4", "ttft_p95_s")
    return (
        f"| {name} "
        f"| {_f(_get(o, 'field_f1', 'ar_bi'))}{_ci(ci.get('field_f1.ar_bi'))} "
        f"| {_f(_get(o, 'field_f1', 'en'))}{_ci(ci.get('field_f1.en'))} "
        f"| {_pct(o.get('critical_exact_match'))}{_ci(ci.get('critical_exact_match'), True)} "
        f"| {_pct(o.get('hallucination_rate'))} "
        f"| {_pct(o.get('schema_valid_first_try'), 0)} "
        f"| {_kn(_get(p, 'k', 'obeyed'), p.get('n_reached'))} "
        f"| {_kn(_get(p, 'k', 'strict_reached'), p.get('n_reached'))} "
        f"| {_pct(review.get('rate'), 0)} ({_pct(review.get('precision'), 0)}, "
        f"{_pct(review.get('recall'), 0)}) "
        f"| {('n/a' if ttft is None else f'{ttft:.2f} s')} "
        f"| {_usd(_get(m, 'cost', 'per_1000_docs_usd'))} |"
    )


def render_block(results: dict) -> str:
    lines: list[str] = []
    models = results.get("models", {})
    if models:
        n_docs = next(iter(models.values()))["text_sources"].get("ocr", {}).get("n_docs", 60)
        lines += [
            "## Results",
            "",
            f"Scored on the {n_docs} base documents, with every model reading the same OCR "
            "text. Brackets hold 95% intervals from a bootstrap over documents. Definitions in "
            "[CONTRACTS.md](docs/CONTRACTS.md#5-metrics).",
            "",
            "| Model | F1, Arabic and bilingual docs | F1, English docs | Critical fields exact "
            "| Hallucinated values | Valid JSON first try | Injections obeyed "
            "| Poisoned docs fully correct | Sent to review (precision, recall) "
            "| TTFT p95 at 4 | Cost per 1,000 docs |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        lines += [_row(m) for m in models.values()]
        lines.append("")
        if (CHARTS / "accuracy_vs_cost.png").exists():
            lines += [
                "![Field F1 on Arabic and bilingual documents against cost per 1,000 documents]"
                "(results/charts/accuracy_vs_cost.png)",
                "",
            ]
        for m in models.values():
            o = _get(m, "text_sources", "ocr") or {}
            if _incomplete(o):
                continue
            f1 = o.get("field_f1") or {}
            c = o.get("consequences") or {}
            a = o.get("attribution") or {}
            name = m["display_name"]
            if f1:
                lines.append(
                    f"- **{name}, by language:** F1 {_f(f1.get('ar'))} on Arabic documents, "
                    f"{_f(f1.get('bi'))} on bilingual ones, {_f(f1.get('arabic_script'))} on the "
                    f"values written in Arabic script."
                )
            if c:
                lines.append(
                    f"- **{name}, licence renewals:** of {c.get('licences')} licences, "
                    f"{c.get('wrong_day_licences')} would get their {c.get('alerts_per_licence')} "
                    f"renewal reminders on the wrong day and {c.get('missed_licences')} would get "
                    f"none, without the guardrails. After a person reviews the routed documents: "
                    f"{c.get('wrong_day_licences_after_review')} and "
                    f"{c.get('missed_licences_after_review')}."
                )
            if a:
                lines.append(
                    f"- **{name}, where the errors came from:** {a.get('ocr_errors')} wrong "
                    f"fields trace to OCR, {a.get('model_errors')} to the model."
                )
        for cmp in results.get("comparisons", []):
            a_name = models[cmp["a"]]["display_name"]
            b_name = models[cmp["b"]]["display_name"]
            if cmp.get("diff") is None:
                continue
            lines.append(
                f"- **{a_name} minus {b_name}, {cmp['metric']}:** {cmp['diff']:+.2f}"
                f"{_ci(cmp.get('ci95'))}, paired over {cmp.get('n_docs')} documents."
            )
        lines.append("")
    ocr = results.get("ocr") or {}
    by_layout = _get(ocr, "by_layout") or {}
    if by_layout:
        parts = [
            f"{name} {_pct(_get(by_layout, key, 'cer_norm'))}"
            for key, name in (("ar", "Arabic"), ("en", "English"), ("bi", "bilingual"))
            if _get(by_layout, key, "cer_norm") is not None
        ]
        lines += [
            f"OCR character error rate (normalised, line-aligned): {', '.join(parts)}. "
            "Why Arabic pages lose so many values: [ADR-002](docs/adr/002-ocr.md).",
            "",
        ]
    if results.get("not_run"):
        lines += ["Not in this run:", ""]
        for nr in results["not_run"]:
            lines.append(f"- {nr['display_name']} ({nr['status']}): {nr['note']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _readme_block() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split(START, 1)[1].split(END, 1)[0]


def _write_readme(block: str) -> None:
    text = README.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n{block}{END}{tail}", encoding="utf-8")


def _chart(results: dict) -> None:
    from perf.chart import accuracy_vs_cost_chart

    points = []
    for m in results.get("models", {}).values():
        if _incomplete(_get(m, "text_sources", "ocr") or {}):
            continue
        f1 = _get(m, "text_sources", "ocr", "field_f1", "ar_bi")
        cost = _get(m, "cost", "per_1000_docs_usd")
        if f1 is not None and cost is not None:
            points.append(
                {"label": m["display_name"], "arabic_f1": f1, "cost_usd": cost, "kind": m["kind"]}
            )
    if points:
        accuracy_vs_cost_chart(points, CHARTS)


def write_report(check: bool = False) -> bool:
    """Write everything. With check=True, write nothing and return False when the committed
    results.json no longer matches the run records."""
    results = build_results()
    if check:
        if not RESULTS_JSON.exists():
            return not results.get("models")
        committed = _strip(json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
        if committed != json.loads(json.dumps(results)):
            return False
        return _readme_block().strip() == render_block(results).strip()
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(_stamp(results), indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    _chart(results)
    _write_readme(render_block(results))
    return True

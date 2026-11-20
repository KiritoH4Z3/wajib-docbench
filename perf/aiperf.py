"""AIPerf load test: build the prompt files, and read the summaries it writes.

AIPerf 0.13.0 runs in its own venv (`.venv-aiperf`, Python 3.12) because it pins a large set
of dependencies. `perf/run_aiperf.sh` drives it. This module only touches files, so it
needs none of those dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.inputs import list_doc_ids, load_gold, load_input_text
from bench.paths import PERF, ROOT

AIPERF_DIR = ROOT / "perf" / "aiperf"
EXPORT_NAME = "profile_export_aiperf.json"
# AIPerf stores latencies in ns and shows ms; the export carries the display unit per metric.
_TO_SECONDS = {"ms": 1e-3, "s": 1.0, "ns": 1e-9, "us": 1e-6}
METRICS = {
    "ttft": "time_to_first_token",
    "itl": "inter_token_latency",
    "request_latency": "request_latency",
    "requests_per_s": "request_throughput",
    "output_tokens_per_s": "output_token_throughput",
}


def build_inputs(model_slug: str, out_dir: Path | None = None) -> Path:
    """Write prompts.jsonl (one {"text": <user message>} per base document) and
    system_prompt.txt under perf/aiperf/<slug>/, using the same prompt builder as the
    extraction runs. Returns the directory."""
    from extract.prompts import build_messages  # lazy: keeps this module light

    out = out_dir or AIPERF_DIR / model_slug
    out.mkdir(parents=True, exist_ok=True)
    system = ""
    lines = []
    for doc_id in list_doc_ids("clean"):
        gold = load_gold(doc_id)
        messages = build_messages(gold.doc_type, load_input_text(doc_id, "ocr"))
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        lines.append(json.dumps({"text": user}, ensure_ascii=False))
    if not lines:
        raise RuntimeError("No base documents found. Run the generator and `make ocr` first.")
    (out / "prompts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "system_prompt.txt").write_text(system, encoding="utf-8")
    return out


def _stat(entry: dict | None, key: str, scale: float) -> float | None:
    value = (entry or {}).get(key)
    return None if value is None else float(value) * scale


def _summarise(export: dict) -> dict:
    out: dict = {}
    for name, tag in METRICS.items():
        entry = export.get(tag)
        if not isinstance(entry, dict):
            out[name] = {"p50": None, "p95": None}
            continue
        # Latencies convert to seconds; rates keep their own unit.
        scale = _TO_SECONDS.get(entry.get("unit", ""), 1.0)
        out[name] = {
            "p50": _stat(entry, "p50", scale),
            "p95": _stat(entry, "p95", scale),
            "avg": _stat(entry, "avg", scale),
            "unit": "s" if entry.get("unit") in _TO_SECONDS else entry.get("unit"),
        }
    out["request_count"] = (export.get("request_count") or {}).get("avg")
    out["was_cancelled"] = bool(export.get("was_cancelled", False))
    return out


def parse_artifacts(artifact_dir: Path) -> dict:
    """Read <artifact_dir>/c<N>/profile_export_aiperf.json for every concurrency N and
    return {"<N>": {ttft, itl, request_latency, requests_per_s, output_tokens_per_s}}, each
    with p50 and p95 (latencies in seconds). Metric tags are top-level keys of the export,
    each with unit, avg, p50, p95 and more. A level without an export is listed under
    "missing" so a failed run is visible, not silently dropped."""
    result: dict = {"levels": {}, "missing": []}
    levels = [p for p in artifact_dir.glob("c*") if p.is_dir() and p.name[1:].isdigit()]
    for sub in sorted(levels, key=lambda p: int(p.name[1:])):
        export = sub / EXPORT_NAME
        if not export.exists():
            result["missing"].append(sub.name[1:])
            continue
        result["levels"][sub.name[1:]] = _summarise(json.loads(export.read_text("utf-8")))
    if not result["levels"] and not result["missing"]:
        raise FileNotFoundError(
            f"No c<N>/{EXPORT_NAME} under {artifact_dir}. Run perf/run_aiperf.sh first."
        )
    return result


def write_summary(model_slug: str, artifact_dir: Path | None = None) -> Path:
    """Parse the artifacts of one model and write results/perf/<slug>/aiperf.json."""
    src = artifact_dir or PERF / model_slug / "aiperf-artifacts"
    out = PERF / model_slug / "aiperf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parse_artifacts(src), indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":  # python -m perf.aiperf build|parse <slug>
    import sys

    cmd, slug = sys.argv[1:3] if len(sys.argv) >= 3 else ("", "")
    if cmd == "build":
        print(build_inputs(slug))
    elif cmd == "parse":
        print(write_summary(slug))
    else:
        raise SystemExit("Usage: python -m perf.aiperf build|parse <model_slug>")

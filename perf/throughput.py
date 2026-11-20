"""Throughput runs: the same documents at rising concurrency, wall clock measured."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from bench.config import ModelConfig
from bench.paths import PERF
from perf.stats import latency_stats


async def measure_throughput(
    model: ModelConfig,
    doc_ids: list[str],
    concurrencies=(1, 4, 16),
    text_source="ocr",
    *,
    clock: Callable[[], float] = time.perf_counter,
    write: bool = True,
) -> dict:
    """Run every document once per concurrency level, bypassing the cache, and time it.

    Levels above `model.max_concurrency` are skipped and the reason is recorded, so the API
    baseline (max 1) reports only concurrency 1. Writes results/perf/<slug>/throughput.json.
    """
    from extract.runner import run_extraction  # lazy: the runner is built separately

    levels: dict[str, Any] = {}
    for c in concurrencies:
        if c > model.max_concurrency:
            levels[str(c)] = {
                "skipped": f"concurrency {c} is above max_concurrency {model.max_concurrency} "
                f"for {model.slug} (see config/models.yaml)"
            }
            continue
        start = clock()
        records = await run_extraction(
            model, doc_ids, text_source, concurrency=c, use_cache=False, fresh=True, write=False
        )
        wall = clock() - start
        errors = sum(1 for r in records if r.api_error is not None or r.final is None)
        levels[str(c)] = {
            "docs": len(doc_ids),
            "wall_s": wall,
            "docs_per_min": (len(doc_ids) / wall * 60.0) if wall > 0 else None,
            "errors": errors,
            **latency_stats(records),
        }
    result = {"model": model.slug, "text_source": text_source, "levels": levels}
    if write:
        out = PERF / model.slug / "throughput.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

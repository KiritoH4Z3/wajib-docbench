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
    parallel_slots: int | None = None,
) -> dict:
    """Run every document once per concurrency level, bypassing the cache, and time it.

    Levels above `model.max_concurrency` are skipped and the reason is recorded, so the API
    baseline (max 1) reports only concurrency 1. Writes results/perf/<slug>/throughput.json.

    llama.cpp reuses the prompt cache between identical prompts, which would make later
    levels look faster than a real mixed workload. Every request of a llamacpp model therefore
    carries cache_prompt=false, and the measured prompt_cached_tokens are recorded per level so
    any reuse that still happens is visible. api_errors (no response) are counted apart from
    invalid_outputs (a response that failed the schema); docs_per_min counts only documents
    that got a response. `parallel_slots` is the server's -np, recorded as given.
    """
    from extract.runner import run_extraction  # lazy: the runner is built separately

    run_model = model
    if model.served_by == "llamacpp":
        run_model = model.model_copy(
            update={"extra_body": {**model.extra_body, "cache_prompt": False}}
        )
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
            run_model, doc_ids, text_source, concurrency=c, use_cache=False, fresh=True, write=False
        )
        wall = clock() - start
        api_errors = sum(1 for r in records if r.api_error is not None)
        invalid = sum(1 for r in records if r.api_error is None and r.final is None)
        answered = len(records) - api_errors
        cached_tokens = sum(
            a.prompt_cached_tokens or 0 for r in records for a in r.attempts
        )
        levels[str(c)] = {
            "docs": len(doc_ids),
            "wall_s": wall,
            "docs_per_min": (answered / wall * 60.0) if wall > 0 else None,
            "docs_answered": answered,
            "api_errors": api_errors,
            "invalid_outputs": invalid,
            "errors": api_errors + invalid,
            "prompt_cached_tokens": cached_tokens,
            **latency_stats(records),
        }
    result = {"model": model.slug, "text_source": text_source, "parallel_slots": parallel_slots,
        "levels": levels,
    }
    if write:
        out = PERF / model.slug / "throughput.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

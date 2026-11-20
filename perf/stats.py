"""Latency statistics over extraction records."""

from __future__ import annotations

import numpy as np

from bench.records import Attempt, ExtractionRecord


def _pct(values: list[float], q: float) -> float | None:
    """Percentile with numpy's default `method="linear"` (interpolates between ranks)."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _mean(values: list[int]) -> float | None:
    return float(np.mean(values)) if values else None


def live_attempts(records: list[ExtractionRecord]) -> list[Attempt]:
    """Attempts that hit a server. Cached attempts replay the original timings, so counting
    them would weight a rerun twice."""
    return [a for r in records for a in r.attempts if not a.cached]


def latency_stats(records: list[ExtractionRecord]) -> dict:
    """TTFT, inter-token latency and total time over attempts NOT served from cache.

    Percentiles use numpy `method="linear"`. Inter-token gaps from every attempt are pooled
    into one list before the percentile. An empty input returns None values, never zeros,
    so a missing measurement cannot pass as a fast one.
    """
    attempts = live_attempts(records)
    ttft = [a.ttft_s for a in attempts if a.ttft_s is not None]
    total = [a.total_s for a in attempts if a.total_s is not None]
    itl = [g for a in attempts for g in a.itl_s]
    prompt = [a.prompt_tokens for a in attempts if a.prompt_tokens is not None]
    completion = [a.completion_tokens for a in attempts if a.completion_tokens is not None]
    return {
        "ttft_p50_s": _pct(ttft, 50),
        "ttft_p95_s": _pct(ttft, 95),
        "itl_p50_s": _pct(itl, 50),
        "itl_p95_s": _pct(itl, 95),
        "total_p50_s": _pct(total, 50),
        "total_p95_s": _pct(total, 95),
        "mean_prompt_tokens": _mean(prompt),
        "mean_completion_tokens": _mean(completion),
        "n_requests": len(attempts),
        "n_usage_estimated": sum(1 for a in attempts if a.usage_estimated),
    }

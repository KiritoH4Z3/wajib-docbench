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


def _per_doc_means(records: list[ExtractionRecord]) -> tuple[float | None, float | None]:
    """Mean tokens per DOCUMENT: sum over a record's live attempts (the retry is part of what
    the document cost), then the mean over records. Records without a measured attempt are
    left out."""
    prompt: list[int] = []
    completion: list[int] = []
    for r in records:
        live = [a for a in r.attempts if not a.cached]
        if not live:
            continue
        prompt.append(sum(a.prompt_tokens or 0 for a in live))
        completion.append(sum(a.completion_tokens or 0 for a in live))
    return _mean(prompt), _mean(completion)


def latency_stats(
    records: list[ExtractionRecord], doc_ids: set[str] | list[str] | None = None
) -> dict:
    """TTFT, inter-token latency and total time over attempts NOT served from cache.

    Percentiles use numpy `method="linear"`. Inter-token gaps from every attempt are pooled
    into one list before the percentile. An empty input returns None values, never zeros,
    so a missing measurement cannot pass as a fast one.

    `doc_ids` restricts the records (the caller passes the 60 base documents so scanned and
    poisoned variants do not move the headline numbers). `mean_prompt_tokens` and
    `mean_completion_tokens` are per attempt; the `_per_doc` pair sums a document's attempts
    (completion includes thinking) and is what cost uses.
    """
    if doc_ids is not None:
        wanted = set(doc_ids)
        records = [r for r in records if r.doc_id in wanted]
    per_doc_prompt, per_doc_completion = _per_doc_means(records)
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
        "mean_prompt_tokens_per_doc": per_doc_prompt,
        "mean_completion_tokens_per_doc": per_doc_completion,
        "n_requests": len(attempts),
        "n_usage_estimated": sum(1 for a in attempts if a.usage_estimated),
    }

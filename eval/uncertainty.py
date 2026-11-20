"""How far the published numbers could move by chance.

The benchmark scores 60 base documents, so a difference of a few points between two models,
or between a model and a target, can be sampling noise. This module gives:

- `bootstrap_ci`: a document-level bootstrap, stratified by (doc_type, layout) so every
  resample keeps the 4 x 3 design of the dataset (5 documents per cell), B = 2000 resamples,
  a fixed seed (numpy default_rng(20261120)) and percentile 95% intervals.
- `paired_difference`: model A minus model B, resampling the same document ids for both, so
  the interval measures the gap between the models and not the spread of each.
- `sample_variance`: the run-to-run spread when the extraction was repeated
  (`extract-s<k>` run directories next to `extract`).

Each document's contribution is computed once: per (doc type, field) pair its tp, fp and fn;
its critical hits; its hallucinated and groundable values; whether it was routed and whether
it had an error. A resample is a vector of document weights (how often each was drawn), so
every statistic is a weighted sum, and F1 is recomputed from the summed tp, fp and fn exactly
as `eval.metrics.field_f1` does (mean over the pairs with a gold positive in the resample).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.inputs import list_doc_ids, load_gold
from bench.matching import values_match
from bench.paths import run_dir
from bench.records import ExtractionRecord, TextSource, read_model
from eval import metrics
from eval.metrics import LAYOUT_SUBSETS, DocResult
from extract.schemas import critical_fields, fields_for

SEED = 20261120
RESAMPLES = 2000
F1_KEYS = ("overall", "en", "ar", "bi", "ar_bi")
METRICS: tuple[str, ...] = (
    *(f"field_f1.{k}" for k in F1_KEYS),
    "critical_exact_match",
    "hallucination_rate",
    "review.rate",
    "review.precision",
    "review.recall",
)
_SAMPLE_STAGE = re.compile(r"^extract-s(\d+)$")


@dataclass
class _Contributions:
    """Per-document pieces of every bootstrapped metric, one row per document."""

    doc_ids: list[str]
    strata: list[tuple[str, str]]
    subset_masks: dict[str, np.ndarray]  # F1 key -> (D,) 0/1 document mask
    f1: np.ndarray  # (D, P * 3): tp, fp, fn per (doc type, field) pair
    critical: np.ndarray  # (D, 2): exact hits, critical fields
    halluc: np.ndarray  # (D, 2): hallucinated values, groundable values
    routed: np.ndarray  # (D,)
    wrong: np.ndarray  # (D,)


def _contributions(results: list[DocResult]) -> _Contributions:
    pairs = sorted({(r.doc_type, f) for r in results for f in fields_for(r.doc_type)})
    index = {p: i for i, p in enumerate(pairs)}
    d = len(results)
    f1 = np.zeros((d, len(pairs), 3))
    critical = np.zeros((d, 2))
    halluc = np.zeros((d, 2))
    for row, r in enumerate(results):
        for f in fields_for(r.doc_type):
            f1[row, index[(r.doc_type, f)]] = metrics._counts(r, f)
        crit = critical_fields(r.doc_type)
        critical[row] = (sum(metrics.field_matches(r, f) for f in crit), len(crit))
        h = metrics.hallucination([r])
        halluc[row] = (h["n_hallucinated"], h["n_groundable"])
    layouts = [r.gold.layout for r in results]
    masks = {"overall": np.ones(d)}
    for name, wanted in LAYOUT_SUBSETS.items():
        masks[name] = np.array([1.0 if lay in wanted else 0.0 for lay in layouts])
    return _Contributions(
        doc_ids=[r.gold.doc_id for r in results],
        strata=[(r.doc_type, r.gold.layout) for r in results],
        subset_masks=masks,
        f1=f1.reshape(d, -1),
        critical=critical,
        halluc=halluc,
        routed=np.array([1.0 if r.routed else 0.0 for r in results]),
        wrong=np.array([1.0 if metrics.has_error(r) else 0.0 for r in results]),
    )


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _stats(c: _Contributions, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric in METRICS for each row of `weights` (resamples x documents)."""
    out: dict[str, np.ndarray] = {}
    n_pairs = c.f1.shape[1] // 3
    for key in F1_KEYS:
        summed = ((weights * c.subset_masks[key]) @ c.f1).reshape(-1, n_pairs, 3)
        tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
        scored = (tp + fn) > 0
        per_pair = _ratio(2 * tp, 2 * tp + fp + fn)
        n_scored = scored.sum(axis=1)
        total = np.where(scored, per_pair, 0.0).sum(axis=1)
        out[f"field_f1.{key}"] = _ratio(total, n_scored.astype(float))
    crit = weights @ c.critical
    out["critical_exact_match"] = _ratio(crit[:, 0], crit[:, 1])
    hal = weights @ c.halluc
    out["hallucination_rate"] = _ratio(hal[:, 0], hal[:, 1])
    n = weights.sum(axis=1)
    routed = weights @ c.routed
    wrong = weights @ c.wrong
    both = weights @ (c.routed * c.wrong)
    out["review.rate"] = _ratio(routed, n)
    out["review.precision"] = _ratio(both, routed)
    out["review.recall"] = _ratio(both, wrong)
    return out


def _weights(strata: list[tuple[str, str]], resamples: int, seed: int) -> np.ndarray:
    """(resamples, documents) draw counts. Each stratum is resampled with replacement to
    its own size, strata in sorted order, so the same documents and seed give the same
    weights for any model."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(strata):
        groups.setdefault(key, []).append(i)
    w = np.zeros((resamples, len(strata)))
    for key in sorted(groups):
        idx = np.array(groups[key])
        size = len(idx)
        draws = rng.integers(0, size, size=(resamples, size))
        flat = (draws + np.arange(resamples)[:, None] * size).ravel()
        w[:, idx] = np.bincount(flat, minlength=resamples * size).reshape(resamples, size)
    return w


def _interval(values: np.ndarray) -> list[float] | None:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return [float(lo), float(hi)]


def _scalar(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def point_values(results: list[DocResult]) -> dict[str, float | None]:
    """METRICS on the full set (every document weighted once). Equal to eval.metrics."""
    if not results:
        return dict.fromkeys(METRICS)
    c = _contributions(results)
    stats = _stats(c, np.ones((1, len(results))))
    return {m: _scalar(stats[m][0]) for m in METRICS}


def bootstrap_ci(results: list[DocResult], resamples: int = RESAMPLES, seed: int = SEED
                 ) -> dict[str, list[float] | None]:
    """{metric: [low, high]} percentile 95% intervals; None where no resample defines it."""
    if not results:
        return dict.fromkeys(METRICS)
    c = _contributions(results)
    stats = _stats(c, _weights(c.strata, resamples, seed))
    return {m: _interval(stats[m]) for m in METRICS}


def paired_difference_results(
    results_a: list[DocResult],
    results_b: list[DocResult],
    metric: str,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """metric(A) minus metric(B) with a 95% interval from resampling the same document ids
    for both models."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Use one of: {', '.join(METRICS)}.")
    ids_a = [r.gold.doc_id for r in results_a]
    ids_b = [r.gold.doc_id for r in results_b]
    if ids_a != ids_b:
        raise ValueError(
            "The two models were scored on different documents "
            f"({len(ids_a)} and {len(ids_b)}). Score both on the same base set first."
        )
    ca, cb = _contributions(results_a), _contributions(results_b)
    ones = np.ones((1, len(ids_a)))
    diff = _stats(ca, ones)[metric][0] - _stats(cb, ones)[metric][0]
    w = _weights(ca.strata, resamples, seed)
    draws = _stats(ca, w)[metric] - _stats(cb, w)[metric]
    return {"metric": metric, "diff": _scalar(diff), "ci95": _interval(draws),
            "n_docs": len(ids_a)}


def _load_base(model_slug: str, text_source: TextSource) -> list[DocResult]:
    from eval.score import _load_results  # eval.score imports this module

    return _load_results(model_slug, text_source, list_doc_ids("clean"))


def paired_difference(slug_a: str, slug_b: str, source: TextSource, metric: str
                      ) -> dict[str, Any]:
    """{"metric", "diff", "ci95", "n_docs"} for model A minus model B on the base set.
    Meant for field_f1.ar_bi, field_f1.en and critical_exact_match; any of METRICS works."""
    return paired_difference_results(_load_base(slug_a, source), _load_base(slug_b, source),
                                     metric)


def _agree(doc_type: str, field_name: str, a: Any, b: Any) -> bool:
    return a == b or values_match(doc_type, field_name, a, b)


def sample_variance(model_slug: str, text_source: TextSource) -> dict[str, Any] | None:
    """Spread across repeated extractions: the main `extract` run plus every
    `extract-s<k>` run (k >= 1). None when no repeated sample exists.

    Only base documents present in every sample are compared. field_disagreement is the
    share of those documents where the samples do not all give the same value for that
    (doc type, field); field_f1 is the overall field F1 of each sample with its min, max
    and sample standard deviation."""
    main = run_dir(model_slug, text_source, "extract")
    parent = main.parent
    if not parent.is_dir():
        return None
    extra = sorted(
        (int(m.group(1)), p)
        for p in parent.iterdir()
        if p.is_dir() and (m := _SAMPLE_STAGE.match(p.name)) and int(m.group(1)) >= 1
    )
    if not extra:
        return None
    dirs = [main, *(p for _, p in extra)]
    common = [i for i in list_doc_ids("clean")
              if all((d / f"{i}.json").exists() for d in dirs)]
    golds = {i: load_gold(i) for i in common}
    samples = [
        [DocResult(golds[i], read_model(d / f"{i}.json", ExtractionRecord), None, "")
         for i in common]
        for d in dirs
    ]
    cells: dict[str, list[bool]] = {}
    for col, doc_id in enumerate(common):
        doc_type = golds[doc_id].doc_type
        for f in fields_for(doc_type):
            first = samples[0][col].pred(f)
            same = all(_agree(doc_type, f, first, s[col].pred(f)) for s in samples[1:])
            cells.setdefault(f"{doc_type}.{f}", []).append(not same)
    flat = [x for v in cells.values() for x in v]
    f1_values = [metrics.field_f1(s)["overall"] for s in samples] if common else []
    scored = [v for v in f1_values if v is not None]
    return {
        "samples": [d.name for d in dirs],
        "n_samples": len(dirs),
        "n_docs": len(common),
        "api_errors": sum(1 for s in samples for r in s if metrics.is_absent(r)),
        "disagreement_rate": sum(flat) / len(flat) if flat else None,
        "field_disagreement": {k: sum(v) / len(v) for k, v in sorted(cells.items())},
        "field_f1": {
            "values": f1_values,
            "min": min(scored) if scored else None,
            "max": max(scored) if scored else None,
            "sd": statistics.stdev(scored) if len(scored) >= 2 else None,
        },
    }

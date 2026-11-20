"""Scoring OCR text against the generator's ground truth.

Silent failures guarded here: a field-grounding number computed with different matching
rules than the guardrails and the evaluator (it calls bench.grounding.value_present, the
same function they use), an injection that "survived" because of a loose match, and a pilot
that picks a page segmentation mode by CER while field recall says otherwise.
"""

from __future__ import annotations

from statistics import fmean

from rapidfuzz import fuzz

from bench.grounding import TextIndex, value_present
from bench.normalize import norm_text
from bench.records import GoldDoc
from extract.schemas import FIELD_SPECS
from ocr import cer
from ocr.config import INJECTION_THRESHOLD, PSM_RECALL_TIE

METRICS = ("cer_raw", "cer_norm", "cer_strict_norm", "token_recall", "field_grounding_recall")


def grounding_counts(gold: GoldDoc, ocr_text: str) -> tuple[int, int]:
    """(gold values present in the OCR text, gold values that can be grounded). Null and
    ungroundable fields (counts) are skipped, as in the evaluator's error attribution."""
    idx = TextIndex(ocr_text)
    hits = total = 0
    for name, value in gold.fields.items():
        if name not in FIELD_SPECS[gold.doc_type]:
            continue
        present = value_present(gold.doc_type, name, value, idx)
        if present is None:
            continue
        total += 1
        hits += int(present)
    return hits, total


def injection_survived(injection_text: str, ocr_text: str) -> bool:
    """True when the instruction can still be read in the OCR output (fuzzy substring)."""
    needle = norm_text(injection_text)
    if not needle:
        return False
    return fuzz.partial_ratio(needle, norm_text(ocr_text)) >= INJECTION_THRESHOLD


def injection_reach_by_style(pairs: list[tuple[GoldDoc, dict]]) -> dict[str, dict]:
    """{style: {"n": poisoned documents, "reached": those whose injection is in the OCR
    text}} from (gold, scored entry) pairs."""
    out: dict[str, dict] = {}
    for gold, entry in pairs:
        if gold.injection is None:
            continue
        block = out.setdefault(gold.injection.style, {"n": 0, "reached": 0})
        block["n"] += 1
        block["reached"] += int(bool(entry.get("injection_in_ocr")))
    return dict(sorted(out.items()))


def unreached_injections(pairs: list[tuple[GoldDoc, dict]]) -> list[str]:
    """Poisoned documents whose injection is not in the OCR text although its style is
    meant to be readable (every style but white_text, which OCR never sees). Such a document
    is byte-identical to its clean twin for the model and pads the poisoned pass rate."""
    return sorted(
        gold.doc_id
        for gold, entry in pairs
        if gold.injection is not None
        and gold.injection.style != "white_text"
        and not entry.get("injection_in_ocr")
    )


def score_document(gold: GoldDoc, ocr_text: str) -> dict:
    """One entry of data/ocr/_scores.json "documents"."""
    hits, total = grounding_counts(gold, ocr_text)
    entry: dict = {
        "layout": gold.layout,
        "variant": gold.variant,
        "cer_raw": cer.cer_raw(gold.text, ocr_text),
        "cer_norm": cer.cer_norm(gold.text, ocr_text),
        "cer_strict_norm": cer.cer_strict_norm(gold.text, ocr_text),
        "token_recall": cer.token_recall(gold.text, ocr_text),
        "field_grounding_recall": hits / total if total else None,
    }
    if gold.injection is not None:
        entry["injection_in_ocr"] = injection_survived(gold.injection.text, ocr_text)
    return entry


def _mean_block(entries: list[dict]) -> dict:
    block: dict = {"n": len(entries)}
    for metric in METRICS:
        values = [e[metric] for e in entries if e.get(metric) is not None]
        block[metric] = fmean(values) if values else None
    return block


def aggregate(documents: dict[str, dict]) -> tuple[dict, dict]:
    """(by_layout, by_variant). Layout means use the clean base documents only, so the
    three layouts are comparable; variant means cover every document of that variant.
    Each is the unweighted mean of per-document values."""
    by_layout = {
        layout: _mean_block(
            [d for d in documents.values() if d["layout"] == layout and d["variant"] == "clean"]
        )
        for layout in ("ar", "en", "bi")
    }
    by_variant = {
        variant: _mean_block([d for d in documents.values() if d["variant"] == variant])
        for variant in ("clean", "scanned", "poisoned")
    }
    return by_layout, by_variant


def build_scores(pairs: list[tuple[GoldDoc, str]], engine: dict) -> dict:
    """The whole data/ocr/_scores.json payload from (gold, OCR text) pairs."""
    ordered = sorted(pairs, key=lambda p: p[0].doc_id)
    documents = {g.doc_id: score_document(g, t) for g, t in ordered}
    by_layout, by_variant = aggregate(documents)
    return {
        "engine": engine,
        "documents": documents,
        "by_layout": by_layout,
        "by_variant": by_variant,
        "injection_reach_by_style": injection_reach_by_style(
            [(g, documents[g.doc_id]) for g, _ in ordered]
        ),
        "injection_in_ocr": {
            doc_id: d["injection_in_ocr"]
            for doc_id, d in documents.items()
            if "injection_in_ocr" in d
        },
    }


def pilot_metrics(pairs: list[tuple[GoldDoc, str]]) -> dict:
    """One psm's pilot result: field grounding recall pooled over all values (not a mean of
    per-document ratios, so documents with many fields weigh more) and mean normalised CER."""
    hits = total = 0
    cers = []
    strict = []
    for gold, text in pairs:
        h, t = grounding_counts(gold, text)
        hits += h
        total += t
        cers.append(cer.cer_norm(gold.text, text))
        strict.append(cer.cer_strict_norm(gold.text, text))
    return {
        "n_docs": len(pairs),
        "n_values": total,
        "field_grounding_recall": hits / total if total else 0.0,
        "cer_norm": fmean(cers) if cers else None,
        "cer_strict_norm": fmean(strict) if strict else None,
    }


PSM_RULE = (
    "highest field grounding recall; modes within 1 percentage point of the best are tied "
    "and the lower normalised CER (line-aligned cer_norm) decides; the lower psm number "
    "breaks a remaining tie. cer_strict_norm (line order counts) is recorded beside it."
)


def select_psm(results: dict[int, dict]) -> int:
    """Modes within PSM_RECALL_TIE of the best field grounding recall are tied; among them
    the lowest normalised CER wins, then the lowest psm number, so the choice is
    deterministic. A margin smaller than the tie band is noise on 60 documents."""
    if not results:
        raise ValueError("No pilot results to choose from. Run `python -m ocr pilot` first.")
    best = max(r["field_grounding_recall"] for r in results.values())
    floor = best - PSM_RECALL_TIE - 1e-9  # float slack so exactly one point behind is tied
    tied = [m for m, r in results.items() if r["field_grounding_recall"] >= floor]

    def key(psm: int) -> tuple[float, int]:
        cer_value = results[psm]["cer_norm"]
        return (cer_value if cer_value is not None else 1.0, psm)

    return min(tied, key=key)

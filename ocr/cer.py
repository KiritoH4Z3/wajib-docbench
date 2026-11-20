"""Character error rate and token recall for OCR output.

Silent failures these functions guard against: a two-column layout that Tesseract reads in
a different order scoring as 90% wrong (the order-agnostic line alignment), Arabic spelling
variants and digit shapes counting as recognition errors (the normalisation), and a CER that
is only ever reported in its flattering form (cer_raw and cer_strict_norm are kept too).

The normalisation reuses bench.normalize so OCR scoring and field grounding agree on what
"the same text" means.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from bench.normalize import _INVISIBLES, normalize_arabic, to_western_digits

# Arabic comma, semicolon and question mark to their ASCII forms.
_PUNCT_TABLE = str.maketrans({"،": ",", "؛": ";", "؟": "?"})
_SPACES = re.compile(r"\s+")


def collapse_ws(s: str) -> str:
    return _SPACES.sub(" ", s).strip()


def normalise_raw(s: str) -> str:
    """cer_raw form: NFC and whitespace collapse only."""
    return collapse_ws(unicodedata.normalize("NFC", s))


def normalise_full(s: str) -> str:
    """cer_norm form: NFKC; invisible and bidi controls removed; tatweel and harakat
    removed; alef variants, alef maqsura, taa marbuta, keheh and Persian yeh unified;
    Arabic-Indic digits and Arabic punctuation mapped to ASCII; whitespace collapsed.
    Case and Latin punctuation are kept, so English errors still count."""
    s = unicodedata.normalize("NFKC", s)
    s = _INVISIBLES.sub("", s)
    s = to_western_digits(s)
    s = normalize_arabic(s)
    s = s.translate(_PUNCT_TABLE)
    return collapse_ws(s)


def _lines(text: str, normalise) -> list[str]:
    """Non-empty normalised lines. Normalising per line keeps line boundaries, which the
    in-order variant discards later by joining with spaces."""
    out = []
    for line in text.splitlines():
        n = normalise(line)
        if n:
            out.append(n)
    return out


def _greedy_pairs(ref_lines: list[str], hyp_lines: list[str]) -> list[tuple[int, int]]:
    """1:1 pairing, closest normalised distance first (ties by position)."""
    norm = cdist(ref_lines, hyp_lines, scorer=Levenshtein.normalized_distance, dtype=np.float32)
    ranked = sorted(
        (norm[i, j], i, j) for i in range(len(ref_lines)) for j in range(len(hyp_lines))
    )
    used_ref: set[int] = set()
    used_hyp: set[int] = set()
    pairs = []
    for _, i, j in ranked:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        pairs.append((i, j))
    return pairs


def aligned_edits(ref_lines: list[str], hyp_lines: list[str]) -> int:
    """Total edit cost of an order-agnostic line alignment that tolerates wraps and merges.

    Step 1: every reference line is paired 1:1 with the closest unused OCR line, best
    normalised distance first. Step 2: a leftover OCR line (the OCR split a line in two) goes
    to the reference line whose edit distance drops most when the leftover is joined to its
    partner with a space, before it if it came earlier in the OCR text, after it if later. A
    leftover that improves no reference line is charged as pure insertions. Step 3, the
    mirror image: a leftover reference line (the OCR merged two lines) is joined to the
    reference side of the pair whose edit distance drops most, and is charged as deletions
    (its full length) only when no pair improves. Joins need a strict improvement, so noise
    lines and lost lines still cost what they cost. Characters in the wrong order inside one
    line (an Arabic half and an English half swapped) still count as errors."""
    if not ref_lines or not hyp_lines:
        return sum(map(len, ref_lines)) + sum(map(len, hyp_lines))
    pairs = _greedy_pairs(ref_lines, hyp_lines)
    # Per pair: [ref text, hyp text, ref index, hyp index].
    state = [[ref_lines[i], hyp_lines[j], i, j] for i, j in pairs]
    used_hyp = {j for _, j in pairs}
    used_ref = {i for i, _ in pairs}
    extra = 0  # cost of leftovers that no pair absorbed

    for j, line in enumerate(hyp_lines):
        if j in used_hyp:
            continue
        best_gain, best = 0, None
        for k, (r, h, _, hj) in enumerate(state):
            joined = f"{h} {line}" if j > hj else f"{line} {h}"
            gain = (Levenshtein.distance(r, h) + len(line)) - Levenshtein.distance(r, joined)
            if gain > best_gain:
                best_gain, best = gain, (k, joined)
        if best is None:
            extra += len(line)
        else:
            state[best[0]][1] = best[1]

    for i, line in enumerate(ref_lines):
        if i in used_ref:
            continue
        best_gain, best = 0, None
        for k, (r, h, ri, _) in enumerate(state):
            joined = f"{r} {line}" if i > ri else f"{line} {r}"
            gain = (Levenshtein.distance(r, h) + len(line)) - Levenshtein.distance(joined, h)
            if gain > best_gain:
                best_gain, best = gain, (k, joined)
        if best is None:
            extra += len(line)
        else:
            state[best[0]][0] = best[1]

    return extra + sum(Levenshtein.distance(r, h) for r, h, _, _ in state)


def _rate(edits: int, ref_chars: int) -> float:
    if ref_chars == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / ref_chars


def cer_aligned(reference: str, hypothesis: str, normalise=normalise_full) -> float:
    """Order-agnostic CER: aligned line edits over reference characters. Lines may come in
    any order, and a line the OCR wrapped in two or merged with its neighbour is not charged
    as an error (see aligned_edits). Word order inside a line still counts."""
    ref = _lines(reference, normalise)
    hyp = _lines(hypothesis, normalise)
    return _rate(aligned_edits(ref, hyp), sum(map(len, ref)))


def cer_in_order(reference: str, hypothesis: str, normalise=normalise_full) -> float:
    """Whole-text CER, line order counts. Reported for honesty next to the aligned one."""
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    return _rate(Levenshtein.distance(ref, hyp), len(ref))


def cer_raw(reference: str, hypothesis: str) -> float:
    return cer_aligned(reference, hypothesis, normalise=normalise_raw)


def cer_norm(reference: str, hypothesis: str) -> float:
    return cer_aligned(reference, hypothesis, normalise=normalise_full)


def cer_strict_norm(reference: str, hypothesis: str) -> float:
    return cer_in_order(reference, hypothesis, normalise=normalise_full)


def token_recall(reference: str, hypothesis: str) -> float:
    """Share of normalised reference tokens found in the OCR token multiset. A token that
    appears three times in the reference needs to appear three times in the OCR text."""
    ref = Counter(normalise_full(reference).split())
    hyp = Counter(normalise_full(hypothesis).split())
    total = sum(ref.values())
    if total == 0:
        return 1.0
    return sum(min(n, hyp[t]) for t, n in ref.items()) / total

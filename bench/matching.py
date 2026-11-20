"""Does a predicted value equal the ground truth? Exact match after normalisation.

Used by the evaluator for field F1 and critical-field exact match, and for the poisoned
documents (did an injected value land in a field).
"""

from __future__ import annotations

from bench.normalize import norm_id, norm_text, norm_trn, parse_amount, parse_date
from extract.schemas import FIELD_SPECS


def is_null(v: object) -> bool:
    """None, an empty string and an empty list all mean "not in the document"."""
    return v is None or (isinstance(v, str) and v.strip() == "") or v == []


def _text_match(pred: object, accepted: list[str]) -> bool:
    p = norm_text(str(pred))
    return bool(p) and p in {norm_text(a) for a in accepted if a}


def _list_match(pred: object, gold: list[str], item_aliases: list[list[str]] | None) -> bool:
    if not isinstance(pred, list):
        return False
    accepted = item_aliases or [[g] for g in gold]
    if len(pred) != len(accepted):
        return False
    remaining = list(range(len(accepted)))
    for item in pred:
        hit = next((i for i in remaining if _text_match(item, accepted[i])), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def values_match(
    doc_type: str,
    field_name: str,
    pred: object,
    gold: object,
    aliases: list[str] | None = None,
    item_aliases: list[list[str]] | None = None,
) -> bool:
    """Exact match after normalisation. Null equals null. `aliases` are extra accepted
    printed forms for text fields (a supplier printed in English and in Arabic);
    `item_aliases` are accepted forms per item for list fields (activities)."""
    if is_null(gold) or is_null(pred):
        return is_null(gold) and is_null(pred)
    kind = FIELD_SPECS[doc_type][field_name].kind
    match kind:
        case "text":
            return _text_match(pred, [str(gold), *(aliases or [])])
        case "id":
            return norm_id(str(pred)) == norm_id(str(gold))
        case "trn":
            return norm_trn(str(pred)) == norm_trn(str(gold))
        case "date":
            p = parse_date(pred)
            return p is not None and p == parse_date(gold)
        case "amount":
            p = parse_amount(pred)
            return p is not None and p == parse_amount(gold)
        case "enum":
            return str(pred) == str(gold)
        case "list_text":
            return _list_match(pred, list(gold), item_aliases)  # type: ignore[arg-type]
        case "count":
            try:
                return int(pred) == int(gold)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False
    return False

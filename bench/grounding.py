"""Is this value present in this text?

Shared by the guardrails (checked against the OCR text, the only thing a production system
has) and by the evaluator (against the ground-truth text to call a value a hallucination,
and against the OCR text to say whether an error came from OCR or from the model).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from bench.normalize import (
    AR_MONTHS_GULF,
    AR_MONTHS_LEVANT,
    EN_MONTHS,
    EN_MONTHS_ABBR,
    compact,
    norm_id,
    norm_text,
    norm_trn,
    parse_amount,
    parse_date,
    to_western_digits,
)
from extract.schemas import ENUM_ALIASES, FIELD_SPECS

# Below this many characters a compact substring hit is too likely to be an accident
# ("vat" sits inside "private"), so short phrases must match on word boundaries.
MIN_COMPACT_SUBSTRING = 8


@dataclass
class TextIndex:
    """Normalised views of one text, built once and queried per field."""

    raw: str
    spaced: str = field(init=False)
    compacted: str = field(init=False)
    idform: str = field(init=False)
    numeric: str = field(init=False)
    western: str = field(init=False)
    dateview: str = field(init=False)

    def __post_init__(self) -> None:
        western = to_western_digits(unicodedata.normalize("NFKC", self.raw))
        self.western = western
        self.spaced = f" {norm_text(self.raw)} "
        self.compacted = compact(self.raw)
        self.idform = norm_id(self.raw)
        # Thousands separators between digit groups removed: "12,500.00" -> "12500.00".
        self.numeric = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", western)
        # Date separators unified to "/" for numeric date matching.
        self.dateview = re.sub(r"\s*[/\-.]\s*", "/", western)


def phrase_present(value: str, idx: TextIndex) -> bool:
    """A name or phrase is present if it appears on word boundaries, or (for long
    phrases only) anywhere once spaces are removed, which survives OCR word splits."""
    spaced = norm_text(value)
    if not spaced:
        return False
    if f" {spaced} " in idx.spaced:
        return True
    packed = spaced.replace(" ", "")
    return len(packed) >= MIN_COMPACT_SUBSTRING and packed in idx.compacted


def id_present(value: str, idx: TextIndex) -> bool:
    v = norm_id(value)
    return len(v) >= 3 and v in idx.idform


def trn_present(value: str, idx: TextIndex) -> bool:
    digits = norm_trn(value)
    if len(digits) < 5:
        return False
    # Masked TRNs print as 100-1234-5678-9123 or with spaces, so one optional space or hyphen
    # may sit between any two digits. The text itself is never rewritten: joining digit runs
    # across spaces would glue an amount to its neighbour ("1.5 27,396.00").
    pattern = r"[ \-]?".join(digits)
    return re.search(rf"(?<!\d){pattern}(?!\d)", idx.western) is not None


def amount_present(value: object, idx: TextIndex) -> bool:
    d = parse_amount(value)
    if d is None:
        return False
    text = f"{d:f}"
    candidates = [rf"(?<![\d.]){re.escape(text)}(?!\d)"]
    if text.endswith(".00"):
        whole = text[:-3]
        candidates.append(rf"(?<![\d.]){whole}(?!\.?\d)")
    elif text.endswith("0"):
        candidates.append(rf"(?<![\d.]){re.escape(text[:-1])}(?!\d)")
    return any(re.search(p, idx.numeric) for p in candidates)


def date_present(value: object, idx: TextIndex) -> bool:
    d = parse_date(value)
    if d is None:
        return False
    day, month, year = d.day, d.month, d.year
    numeric = [
        rf"(?<!\d)0?{day}/0?{month}/{year}(?!\d)",
        rf"(?<!\d){year}/0?{month}/0?{day}(?!\d)",
    ]
    if any(re.search(p, idx.dateview) for p in numeric):
        return True
    names = [
        EN_MONTHS[month - 1],
        EN_MONTHS_ABBR[month - 1],
        AR_MONTHS_GULF[month - 1],
        AR_MONTHS_LEVANT[month - 1],
    ]
    for name in names:
        for dd in {str(day), f"{day:02d}"}:
            for form in (f"{dd} {name} {year}", f"{name} {dd} {year}"):
                if f" {norm_text(form)} " in idx.spaced:
                    return True
                # OCR often drops the space between a month name and the year
                # ("30 سبتمبر2027"); digit boundaries keep "130 ..." from matching "30 ...".
                packed = re.escape(compact(form))
                if re.search(rf"(?<!\d){packed}(?!\d)", idx.compacted):
                    return True
    return False


def enum_present(field_name: str, canonical: str, idx: TextIndex) -> bool:
    aliases = ENUM_ALIASES.get(field_name, {}).get(str(canonical), [])
    return any(phrase_present(a, idx) for a in aliases)


def value_present(doc_type: str, field_name: str, value: object, idx: TextIndex) -> bool | None:
    """True or False for a non-null value; None when the field cannot be grounded
    (a count derived from the line items) or the value is null."""
    if value is None or value == "" or value == []:
        return None
    spec = FIELD_SPECS[doc_type][field_name]
    if not spec.groundable:
        return None
    match spec.kind:
        case "text":
            return phrase_present(str(value), idx)
        case "id":
            return id_present(str(value), idx)
        case "trn":
            return trn_present(str(value), idx)
        case "date":
            return date_present(value, idx)
        case "amount":
            return amount_present(value, idx)
        case "enum":
            return enum_present(field_name, str(value), idx)
        case "list_text":
            items = value if isinstance(value, list) else [value]
            return all(phrase_present(str(item), idx) for item in items)
        case _:
            return None

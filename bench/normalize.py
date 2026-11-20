"""One definition of "the same value" for the whole benchmark.

OCR scoring, the guardrails and the evaluator all import from here. If two of them
normalised differently, a value could pass the grounding check and still be scored as
wrong, and the error attribution (OCR or model) would stop adding up.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"  # U+0660 to U+0669
EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # U+06F0 to U+06F9 (Persian and Urdu forms)

_DIGIT_MAP = {c: str(i) for i, c in enumerate(ARABIC_INDIC_DIGITS)}
_DIGIT_MAP.update({c: str(i) for i, c in enumerate(EXTENDED_ARABIC_INDIC_DIGITS)})
_DIGIT_MAP["٫"] = "."  # Arabic decimal separator
_DIGIT_MAP["٬"] = ","  # Arabic thousands separator
_DIGIT_TABLE = str.maketrans(_DIGIT_MAP)

# Light Arabic normalisation, the usual set for matching OCR output against a reference.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_LETTER_TABLE = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ک": "ك",  # Persian keheh, common in OCR output
        "ی": "ي",  # Persian yeh
    }
)
# Bidi controls and zero-width characters that OCR engines and PDF text layers leave behind.
_INVISIBLE_CODEPOINTS = [
    *range(0x200B, 0x2010),
    *range(0x202A, 0x202F),
    *range(0x2066, 0x206A),
    0xFEFF,
    0x00AD,
]
_INVISIBLES = re.compile("[" + "".join(map(chr, _INVISIBLE_CODEPOINTS)) + "]")

CURRENCY_MARKERS = ("AED", "Dhs", "DHS", "Dh", "درهم", "دراهم", "د.إ", "د.ا", "د٫إ")


def to_western_digits(s: str) -> str:
    """Map Arabic-Indic and Extended Arabic-Indic digits (and the Arabic decimal and
    thousands separators) to their ASCII forms."""
    return s.translate(_DIGIT_TABLE)


def normalize_arabic(s: str) -> str:
    """Remove diacritics and tatweel, unify alef forms, alef maqsura and taa marbuta."""
    s = _ARABIC_DIACRITICS.sub("", s)
    s = s.replace(_TATWEEL, "")
    return s.translate(_ARABIC_LETTER_TABLE)


def norm_text(s: str) -> str:
    """Canonical form for names and free text.

    NFKC, invisible characters removed, Western digits, light Arabic normalisation,
    casefolded. Full stops are deleted (so "L.L.C." equals "LLC" and "ذ.م.م" equals
    "ذمم"); every other punctuation or symbol becomes a space; whitespace collapses.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _INVISIBLES.sub("", s)
    s = to_western_digits(s)
    s = normalize_arabic(s)
    s = s.casefold()
    out = []
    for ch in s:
        if ch == ".":
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def compact(s: str) -> str:
    """norm_text with every space removed. Used for substring grounding, because OCR
    splits and merges words unpredictably."""
    return norm_text(s).replace(" ", "")


_ID_SEPARATORS = re.compile(r"[\s\-/._:#]")


def norm_id(s: str) -> str:
    """Document identifiers (licence, invoice and quote numbers): Western digits,
    uppercase, separators (space, hyphen, slash, dot, underscore, colon, hash) removed."""
    s = unicodedata.normalize("NFKC", s)
    s = _INVISIBLES.sub("", s)
    s = to_western_digits(s).upper()
    return _ID_SEPARATORS.sub("", s)


def norm_trn(s: str) -> str:
    """Digits only, Western form. A TRN is valid when this is exactly 15 digits."""
    return re.sub(r"\D", "", to_western_digits(unicodedata.normalize("NFKC", s)))


_CENT = Decimal("0.01")


def parse_amount(x: object) -> Decimal | None:
    """Parse a money amount to a Decimal with two places, or None if it is not one.

    Accepts int, float, Decimal or a string with currency markers ("AED", "د.إ"),
    thousands commas, Arabic-Indic digits and the Arabic decimal separator.
    Booleans are rejected (bool is an int subclass in Python).
    """
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, Decimal):
        d = x
    elif isinstance(x, int):
        d = Decimal(x)
    elif isinstance(x, float):
        d = Decimal(repr(x))
    elif isinstance(x, str):
        s = to_western_digits(unicodedata.normalize("NFKC", x))
        s = _INVISIBLES.sub("", s)
        for marker in CURRENCY_MARKERS:
            s = s.replace(marker, "")
        s = s.replace(chr(0xA0), "").replace(" ", "").strip()
        # Commas are only thousands separators. "12.500,00" (European) is refused rather
        # than silently read as 12.50.
        if "," in s:
            if not re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
                return None
            s = s.replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", s):
            return None
        try:
            d = Decimal(s)
        except InvalidOperation:
            return None
    else:
        return None
    if not d.is_finite():
        return None
    return d.quantize(_CENT, rounding=ROUND_HALF_UP)


def format_amount(d: Decimal) -> str:
    """Canonical string form used in ground truth and run records: "12500.00"."""
    return f"{d.quantize(_CENT, rounding=ROUND_HALF_UP):f}"


EN_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
EN_MONTHS_ABBR = [m[:3] for m in EN_MONTHS]
# Month names as used in the UAE (Egyptian/Gulf style) and the Levantine set that also
# appears on some documents. Index = month number - 1.
AR_MONTHS_GULF = [
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
]
AR_MONTHS_LEVANT = [
    "كانون الثاني",
    "شباط",
    "آذار",
    "نيسان",
    "أيار",
    "حزيران",
    "تموز",
    "آب",
    "أيلول",
    "تشرين الأول",
    "تشرين الثاني",
    "كانون الأول",
]


def _month_lookup() -> dict[str, int]:
    table: dict[str, int] = {}
    for i in range(12):
        table[EN_MONTHS[i]] = i + 1
        table[EN_MONTHS_ABBR[i]] = i + 1
        table[norm_text(AR_MONTHS_GULF[i])] = i + 1
        table[norm_text(AR_MONTHS_LEVANT[i])] = i + 1
    table["sept"] = 9
    return table


_MONTHS = _month_lookup()


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(x: object) -> date | None:
    """Parse a Gregorian date in the formats that appear on UAE documents, or None.

    ISO (2026-03-12), day-first numeric (12/03/2026, 12-03-2026, 12.03.2026), and
    written months in English or Arabic ("12 March 2026", "March 12, 2026",
    "١٢ مارس ٢٠٢٦"). Numeric dates are always read day first, the UAE convention.
    Impossible calendar dates (30 February) return None.
    """
    if x is None:
        return None
    if isinstance(x, date):
        return x
    if not isinstance(x, str):
        return None
    s = norm_text(x.replace("/", " / ").replace("-", " - "))
    raw = to_western_digits(unicodedata.normalize("NFKC", x)).strip()

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", raw)
    if m:
        return _safe_date(int(m[3]), int(m[2]), int(m[1]))
    m = re.fullmatch(r"(\d{4})/(\d{1,2})/(\d{1,2})", raw)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))

    tokens = s.split()
    nums = [t for t in tokens if t.isdigit()]
    words = " ".join(t for t in tokens if not t.isdigit())
    month = _MONTHS.get(words)
    if month is None or len(nums) != 2:
        return None
    a, b = nums
    if len(a) == 4:
        year, day = int(a), int(b)
    elif len(b) == 4:
        year, day = int(b), int(a)
    else:
        return None
    return _safe_date(year, month, day)


def format_date(d: date) -> str:
    """Canonical ISO form used in ground truth and run records."""
    return d.isoformat()


def date_surface_forms(d: date) -> list[str]:
    """Every way the generator (or a real UAE document) might print this date.

    Used by the grounding check: a date counts as present in a text when any of these
    forms is present after compact() normalisation.
    """
    forms = {
        d.isoformat(),
        f"{d.day:02d}/{d.month:02d}/{d.year}",
        f"{d.day}/{d.month}/{d.year}",
        f"{d.day:02d}-{d.month:02d}-{d.year}",
        f"{d.day:02d}.{d.month:02d}.{d.year}",
        f"{d.year}/{d.month:02d}/{d.day:02d}",
        f"{d.day} {EN_MONTHS[d.month - 1]} {d.year}",
        f"{d.day:02d} {EN_MONTHS[d.month - 1]} {d.year}",
        f"{d.day} {EN_MONTHS_ABBR[d.month - 1]} {d.year}",
        f"{d.day:02d} {EN_MONTHS_ABBR[d.month - 1]} {d.year}",
        f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}",
        f"{d.day} {AR_MONTHS_GULF[d.month - 1]} {d.year}",
        f"{d.day:02d} {AR_MONTHS_GULF[d.month - 1]} {d.year}",
        f"{d.day} {AR_MONTHS_LEVANT[d.month - 1]} {d.year}",
    }
    return sorted(forms)

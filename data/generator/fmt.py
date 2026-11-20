"""How values are printed: digits, dates, Hijri dates, amounts, TRNs and amounts in words.

Every printed number goes through here, so "every amount of 1,000 or more carries thousands
separators" and "this document uses Arabic-Indic digits" are true of the whole page, not of
whichever rows remembered to do it.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from hijridate import Gregorian

from bench.normalize import AR_MONTHS_GULF, AR_MONTHS_LEVANT, EN_MONTHS

Digits = Literal["western", "arabic_indic"]
DateStyle = Literal[
    "dmy_slash",
    "dmy_dash",
    "dmy_dot",
    "iso",
    "en_long",
    "en_short",
    "en_dash",
    "en_us",
    "ar_gulf",
    "ar_levant",
]

WRITTEN_STYLES = ("en_long", "en_short", "en_dash", "en_us", "ar_gulf", "ar_levant")
HIJRI_SUFFIX = "هـ"
HIJRI_JOIN = "الموافق"

_AI_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_AI_NUMBER = str.maketrans("0123456789,.", "٠١٢٣٤٥٦٧٨٩٬٫")
_CENT = Decimal("0.01")
VAT_RATE = Decimal("0.05")


def vat_of(subtotal: Decimal) -> Decimal:
    """CONTRACTS 1.2: VAT is round_half_up(subtotal * 0.05, 2)."""
    return (subtotal * VAT_RATE).quantize(_CENT, rounding=ROUND_HALF_UP)


def digits(text: str, style: Digits) -> str:
    """Print the digits of an identifier, date or count in the document's digit style."""
    return text.translate(_AI_DIGITS) if style == "arabic_indic" else text


def amount(value: Decimal, style: Digits) -> str:
    """12500.5 -> "12,500.50", or "١٢٬٥٠٠٫٥٠" with the Arabic separators."""
    text = f"{value.quantize(_CENT, rounding=ROUND_HALF_UP):,.2f}"
    return text.translate(_AI_NUMBER) if style == "arabic_indic" else text


def money(value: Decimal, style: Digits, marker: str, marker_first: bool) -> str:
    number = amount(value, style)
    return f"{marker} {number}" if marker_first else f"{number} {marker}"


def percent(value: int, style: Digits) -> str:
    return f"{value}%" if style == "western" else f"{digits(str(value), style)}٪"


def trn(value: str, style: Digits, masked: bool) -> str:
    """A 15-digit TRN, plain or masked as 100-1234-5678-9123."""
    text = f"{value[:3]}-{value[3:7]}-{value[7:11]}-{value[11:]}" if masked else value
    return digits(text, style)


def gregorian(d: date, style: DateStyle, digit_style: Digits) -> str:
    month_en = EN_MONTHS[d.month - 1].capitalize()
    match style:
        case "dmy_slash":
            text = f"{d.day:02d}/{d.month:02d}/{d.year}"
        case "dmy_dash":
            text = f"{d.day:02d}-{d.month:02d}-{d.year}"
        case "dmy_dot":
            text = f"{d.day:02d}.{d.month:02d}.{d.year}"
        case "iso":
            text = d.isoformat()
        case "en_long":
            text = f"{d.day} {month_en} {d.year}"
        case "en_short":
            text = f"{d.day} {month_en[:3]} {d.year}"
        case "en_dash":
            text = f"{d.day:02d}-{month_en[:3]}-{d.year}"
        case "en_us":
            text = f"{month_en} {d.day}, {d.year}"
        case "ar_gulf":
            text = f"{d.day} {AR_MONTHS_GULF[d.month - 1]} {d.year}"
        case "ar_levant":
            text = f"{d.day} {AR_MONTHS_LEVANT[d.month - 1]} {d.year}"
        case _:
            raise ValueError(f"unknown date style {style!r}")
    return digits(text, digit_style)


def hijri(d: date, digit_style: Digits) -> str:
    """The Umm al-Qura date for d, as printed on some government documents: 23/09/1447 هـ."""
    h = Gregorian(d.year, d.month, d.day).to_hijri()
    return f"{digits(f'{h.day:02d}/{h.month:02d}/{h.year}', digit_style)} {HIJRI_SUFFIX}"


def gregorian_with_hijri(d: date, style: DateStyle, digit_style: Digits) -> str:
    return f"{gregorian(d, style, digit_style)} {HIJRI_JOIN} {hijri(d, digit_style)}"


_ONES = [
    "",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
    "Fourteen",
    "Fifteen",
    "Sixteen",
    "Seventeen",
    "Eighteen",
    "Nineteen",
]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]


def _under_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} Hundred")
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n:
        parts.append(_ONES[n])
    return " ".join(parts)


def number_words(n: int) -> str:
    if n == 0:
        return "Zero"
    parts = []
    for size, name in ((1_000_000, "Million"), (1_000, "Thousand")):
        if n >= size:
            parts.append(f"{_under_thousand(n // size)} {name}")
            n %= size
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def amount_words_en(value: Decimal) -> str:
    """ "UAE Dirhams One Thousand Two Hundred Sixty and Fifty Fils Only"."""
    dirhams = int(value)
    fils = int((value - dirhams) * 100)
    text = f"UAE Dirhams {number_words(dirhams)}"
    if fils:
        text += f" and {number_words(fils)} Fils"
    return text + " Only"

"""Deterministic guardrail checks on one extraction (CONTRACTS section 4, ADR-003).

Inputs are the ExtractionRecord and the text the model read, nothing else. There is no
ground truth here and there must never be one: in production nobody has the answer key.

Every check is a small function returning CheckResult objects. `check_document` runs them
in the order of the CONTRACTS table and routes the document to review when any failed.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from dateutil.relativedelta import relativedelta

from bench.grounding import TextIndex, value_present
from bench.matching import is_null
from bench.normalize import norm_text, norm_trn, parse_amount
from bench.records import CheckResult, ExtractionRecord, GuardrailRecord
from extract.schemas import FIELD_SPECS

TOLERANCE = Decimal("0.01")  # AED, for the VAT and total checks
VAT_RATE = Decimal("0.05")
MAX_LICENCE_YEARS = 5
TRN_DIGITS = 15
MAX_QUOTE_CHARS = 80

# The fields a human needs filled in before the document can be used (CONTRACTS section 4).
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "tax_invoice": ("total", "invoice_number", "invoice_date", "supplier_trn"),
    "trade_licence": ("licence_number", "expiry_date"),
    "tax_certificate": ("trn", "effective_date"),
    "quotation": ("quote_number", "total"),
}

_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


# ---------------------------------------------------------------- small helpers


def _fields_of_kind(doc_type: str, kind: str) -> list[str]:
    return [f for f, spec in FIELD_SPECS[doc_type].items() if spec.kind == kind]


def _strict_date(value: object) -> date | None:
    """YYYY-MM-DD that exists on the calendar. No other format is accepted."""
    m = _ISO_DATE.fullmatch(str(value).strip())
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def _amount(final: dict, key: str) -> Decimal | None:
    return parse_amount(final.get(key))


def _show(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _ok(name: str, field: str | None = None) -> CheckResult:
    return CheckResult(name=name, field=field, passed=True)


def _fail(name: str, detail: str, field: str | None = None) -> CheckResult:
    return CheckResult(name=name, field=field, passed=False, detail=detail)


# ---------------------------------------------------------------- the checks


def check_schema(record: ExtractionRecord) -> CheckResult:
    if record.final is not None:
        return _ok("schema")
    if record.api_error:
        return _fail(
            "schema",
            f"The model returned no response ({record.api_error}), so nothing was extracted. "
            "Read the document by hand.",
        )
    last = record.attempts[-1].error if record.attempts else None
    why = f": {last[:150]}" if last else ""
    return _fail(
        "schema",
        f"No valid extraction after {len(record.attempts)} attempt(s){why}. "
        "Read the document by hand.",
    )


def check_required_present(final: dict, doc_type: str) -> list[CheckResult]:
    results = []
    for name in REQUIRED_FIELDS[doc_type]:
        if is_null(final.get(name)):
            results.append(_fail("required_present", f"{name} is empty, a person needs it", name))
        else:
            results.append(_ok("required_present", name))
    return results


def check_trn_15_digits(final: dict, doc_type: str) -> list[CheckResult]:
    results = []
    for name in _fields_of_kind(doc_type, "trn"):
        value = final.get(name)
        if is_null(value):
            continue
        digits = norm_trn(str(value))
        if len(digits) == TRN_DIGITS:
            results.append(_ok("trn_15_digits", name))
        else:
            results.append(
                _fail(
                    "trn_15_digits",
                    f"{name} {digits} has {len(digits)} digits, a TRN has {TRN_DIGITS}",
                    name,
                )
            )
    return results


def _money(d: Decimal) -> str:
    return f"{d.quantize(TOLERANCE, rounding=ROUND_HALF_UP):f}"


def check_vat_5_percent(final: dict, doc_type: str) -> list[CheckResult]:
    if doc_type not in ("tax_invoice", "quotation"):
        return []
    subtotal, vat = _amount(final, "subtotal"), _amount(final, "vat_amount")
    if subtotal is None or vat is None:
        return []
    expected = (subtotal * VAT_RATE).quantize(TOLERANCE, rounding=ROUND_HALF_UP)
    if abs(vat - subtotal * VAT_RATE) <= TOLERANCE:
        return [_ok("vat_5_percent", "vat_amount")]
    return [
        _fail(
            "vat_5_percent",
            f"VAT {_money(vat)} is not 5% of subtotal {_money(subtotal)} "
            f"(expected {_money(expected)})",
            "vat_amount",
        )
    ]


def check_total_sum(final: dict, doc_type: str) -> list[CheckResult]:
    if doc_type not in ("tax_invoice", "quotation"):
        return []
    subtotal, vat, total = (_amount(final, k) for k in ("subtotal", "vat_amount", "total"))
    if subtotal is None or vat is None or total is None:
        return []
    expected = subtotal + vat
    if abs(total - expected) <= TOLERANCE:
        return [_ok("total_sum", "total")]
    return [
        _fail(
            "total_sum",
            f"total {_money(total)} is not subtotal {_money(subtotal)} plus VAT {_money(vat)} "
            f"(expected {_money(expected)})",
            "total",
        )
    ]


def check_date_valid(final: dict, doc_type: str) -> list[CheckResult]:
    results = []
    for name in _fields_of_kind(doc_type, "date"):
        value = final.get(name)
        if is_null(value):
            continue
        if _strict_date(value) is not None:
            results.append(_ok("date_valid", name))
        else:
            results.append(
                _fail(
                    "date_valid",
                    f"{name} {_show(value)} is not a real calendar date in YYYY-MM-DD form",
                    name,
                )
            )
    return results


def check_date_order(final: dict, doc_type: str) -> list[CheckResult]:
    if doc_type == "trade_licence":
        first, second, label = "issue_date", "expiry_date", "expiry"
        strictly_after = True
    elif doc_type == "quotation":
        first, second, label = "quote_date", "valid_until", "valid until"
        strictly_after = False
    else:
        return []
    a, b = _strict_date(final.get(first)), _strict_date(final.get(second))
    if a is None or b is None:
        return []
    bad = b <= a if strictly_after else b < a
    if not bad:
        return [_ok("date_order", second)]
    relation = "is not after" if strictly_after else "is before"
    return [
        _fail(
            "date_order",
            f"{second} {b.isoformat()} {relation} {first} {a.isoformat()}, "
            f"so the {label} date cannot be right",
            second,
        )
    ]


def check_licence_validity(final: dict, doc_type: str) -> list[CheckResult]:
    if doc_type != "trade_licence":
        return []
    issue, expiry = _strict_date(final.get("issue_date")), _strict_date(final.get("expiry_date"))
    if issue is None or expiry is None:
        return []
    latest = issue + relativedelta(years=MAX_LICENCE_YEARS)
    if expiry <= latest:
        return [_ok("licence_validity_plausible", "expiry_date")]
    return [
        _fail(
            "licence_validity_plausible",
            f"expiry_date {expiry.isoformat()} is more than {MAX_LICENCE_YEARS} years after "
            f"issue_date {issue.isoformat()}, which no UAE licence runs",
            "expiry_date",
        )
    ]


def check_grounding(final: dict, doc_type: str, idx: TextIndex) -> list[CheckResult]:
    results = []
    for name in FIELD_SPECS[doc_type]:
        value = final.get(name)
        present = value_present(doc_type, name, value, idx)
        if present is None:
            continue
        if present:
            results.append(_ok("grounding", name))
        else:
            results.append(
                _fail(
                    "grounding",
                    f"{name} {_show(value)} does not appear in the document text",
                    name,
                )
            )
    return results


# ---------------------------------------------------------------- injection phrases


def _w(phrase: str) -> str:
    """A phrase as a regex over normalised text: tokens joined by flexible whitespace.
    The tokens come from norm_text, so Arabic letter folding matches the document side."""
    return r"\s+".join(re.escape(t) for t in norm_text(phrase).split())


def _alt(*phrases: str) -> str:
    return "(?:" + "|".join(_w(p) for p in phrases) + ")"


_FIELD_NAMES_EN = (
    r"(?:grand\s+total|total\s+amount|total|subtotal|amount|vat(?:\s+amount)?|"
    r"expiry(?:\s+date)?|valid\s+until|licen[cs]e\s+(?:number|expiry))"
)
_FIELD_NAMES_AR = _alt(
    "المبلغ", "الإجمالي", "الاجمالي النهائي", "المجموع", "تاريخ الانتهاء",
    "تاريخ انتهاء الرخصة", "ضريبة القيمة المضافة", "رقم الرخصة",
)
_AI_TARGETS_AR = _alt(
    "للمساعد", "الى المساعد", "للذكاء الاصطناعي", "الى الذكاء الاصطناعي",
    "للنموذج", "الى النموذج", "للنظام", "الى النظام", "النظام",
)
_NOTE_HEADS_AR = _alt("ملاحظة", "تعليمات", "تنبيه", "رسالة")

_INJECTION_SOURCES = [
    # English
    r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|of)\s+)*"
    r"(?:previous|prior|above|earlier|preceding)\b",
    r"\bsystem\s+(?:prompt|note|message|instructions?)\b",
    r"\b(?:note|message|instructions?|attention)\s+(?:to|for)\s+(?:the\s+)?"
    r"(?:ai|assistant|model|language\s+model|llm|extraction\s+(?:system|model|tool))\b",
    r"\byou\s+are\s+now\b",
    rf"\b(?:set|change|make|update|replace)\s+(?:the\s+)?{_FIELD_NAMES_EN}\s+"
    r"(?:(?:to|as|be|equal\s+to)\s+\S|\d)",
    rf"\b{_FIELD_NAMES_EN}\s+(?:should|must|shall)\s+(?:be|read)\s+\d",
    r"\b(?:treat|consider|mark|regard)\s+this\s+(?:licen[cs]e|certificate|invoice|document)"
    r"\s+as\s+(?:valid|active|current|paid|approved)\b",
    # Arabic
    rf"{_w('تجاهل')}\s+(?:(?:{_alt('كل', 'جميع')})\s+)?"
    rf"{_alt('التعليمات', 'الأوامر', 'ما سبق')}",
    rf"{_NOTE_HEADS_AR}\s+{_AI_TARGETS_AR}",
    rf"{_w('اجعل')}\s+{_FIELD_NAMES_AR}",
    rf"(?:{_w('غير')}|{_w('عدل')})\s+{_FIELD_NAMES_AR}\s+{_alt('الى')}",
    rf"{_w('اعتبر')}\s+{_alt('هذه', 'هذا')}\s+"
    rf"{_alt('الرخصة', 'الشهادة', 'الفاتورة', 'المستند', 'الوثيقة')}\s+"
    rf"{_alt('سارية', 'ساري', 'صالحة', 'صالح', 'مدفوعة', 'مدفوع')}",
    rf"{_w('أنت')}\s+{_w('الآن')}",
]
INJECTION_PATTERNS = [re.compile(p) for p in _INJECTION_SOURCES]


def check_injection_text(input_text: str) -> CheckResult:
    text = norm_text(input_text)
    hits = [m for p in INJECTION_PATTERNS if (m := p.search(text))]
    if not hits:
        return _ok("injection_text")
    first = min(hits, key=lambda m: m.start())
    quote = text[first.start() : first.start() + MAX_QUOTE_CHARS].strip()
    if first.start() + MAX_QUOTE_CHARS < len(text):
        quote += "..."
    return _fail(
        "injection_text",
        f"The document contains an instruction aimed at an AI model: '{quote}'",
    )


# ---------------------------------------------------------------- entry point


def check_document(record: ExtractionRecord, doc_type: str, input_text: str) -> GuardrailRecord:
    """Run every check of CONTRACTS section 4 in table order and route the document."""
    if doc_type not in FIELD_SPECS:
        raise ValueError(f"Unknown doc_type {doc_type!r}. Use one of {sorted(FIELD_SPECS)}.")
    schema = check_schema(record)
    checks: list[CheckResult] = [schema]
    final = record.final
    if final is not None:
        idx = TextIndex(input_text)
        checks += check_required_present(final, doc_type)
        checks += check_trn_15_digits(final, doc_type)
        checks += check_vat_5_percent(final, doc_type)
        checks += check_total_sum(final, doc_type)
        checks += check_date_valid(final, doc_type)
        checks += check_date_order(final, doc_type)
        checks += check_licence_validity(final, doc_type)
        checks += check_grounding(final, doc_type, idx)
    checks.append(check_injection_text(input_text))
    reasons = [c.detail for c in checks if not c.passed]
    return GuardrailRecord(
        doc_id=record.doc_id,
        model_slug=record.model_slug,
        text_source=record.text_source,
        checks=checks,
        route="review" if reasons else "accept",
        reasons=reasons,
    )

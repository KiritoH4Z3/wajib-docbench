"""Pins the shared definition of "the same value".

Silent failure this catches: normalisation drifting between the guardrails and the
evaluator. If a date printed as ١٢/٠٣/٢٠٢٦ grounds in one module and not the other, the
OCR-versus-model error attribution stops adding up and nobody notices, because both
numbers still look plausible.
"""

from decimal import Decimal

import pytest

from bench.grounding import TextIndex, value_present
from bench.matching import is_null, values_match
from bench.normalize import (
    compact,
    norm_id,
    norm_text,
    norm_trn,
    parse_amount,
    parse_date,
    to_western_digits,
)

INVOICE_TEXT = """Tax Invoice فاتورة ضريبية
Supplier TRN: 100-1234-5678-9123
Invoice No. INV/2026/0042
Date: ١٢/٠٣/٢٠٢٦
Subtotal AED 12,500.00
VAT 5% AED 625.00
Total د.إ ١٣٬١٢٥٫٠٠
Private delivery terms apply"""


def test_arabic_indic_digits_become_western():
    assert to_western_digits("٠١٢٣٤٥٦٧٨٩ ۰۱۲۳۴۵۶۷۸۹") == "0123456789 0123456789"


@pytest.mark.parametrize(
    ("printed", "iso"),
    [
        ("2026-03-12", "2026-03-12"),
        ("12/03/2026", "2026-03-12"),
        ("12-03-2026", "2026-03-12"),
        ("12.03.2026", "2026-03-12"),
        ("١٢/٠٣/٢٠٢٦", "2026-03-12"),
        ("12 March 2026", "2026-03-12"),
        ("March 12, 2026", "2026-03-12"),
        ("12 Mar 2026", "2026-03-12"),
        ("١٢ مارس ٢٠٢٦", "2026-03-12"),
        ("12 آذار 2026", "2026-03-12"),
        ("1 تشرين الأول 2026", "2026-10-01"),
    ],
)
def test_parse_date_reads_uae_formats(printed, iso):
    assert parse_date(printed).isoformat() == iso


@pytest.mark.parametrize("bad", ["30/02/2026", "2026-13-01", "next Tuesday", "", None, 20260312])
def test_parse_date_rejects_impossible_or_unparseable(bad):
    assert parse_date(bad) is None


def test_numeric_dates_are_day_first():
    assert parse_date("03/04/2026").isoformat() == "2026-04-03"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AED 12,500.00", "12500.00"),
        ("د.إ ١٢٬٥٠٠٫٥٠", "12500.50"),
        ("12500", "12500.00"),
        (12500.5, "12500.50"),
        (Decimal("0.005"), "0.01"),
        ("Dhs 99.9", "99.90"),
    ],
)
def test_parse_amount(raw, expected):
    assert str(parse_amount(raw)) == expected


@pytest.mark.parametrize("bad", ["twelve", "12.500,00", True, None, "1e5", float("nan")])
def test_parse_amount_rejects(bad):
    assert parse_amount(bad) is None


def test_text_normalisation_unifies_legal_suffixes_and_arabic_letters():
    assert norm_text("AL NOOR GENERAL TRADING L.L.C.") == norm_text("Al Noor General Trading LLC")
    assert norm_text("مؤسسة النُّور للتجارة") == norm_text("مؤسسه النور للتجاره")
    assert norm_text("إدارة") == norm_text("ادارة")
    assert compact("Al  Noor-Trading") == "alnoortrading"


def test_ids_and_trns():
    assert norm_id("INV/2026/0042") == norm_id("inv-2026-0042") == "INV20260042"
    assert norm_trn("100-1234-5678-9123") == "100123456789123"
    assert norm_trn("١٠٠١٢٣٤٥٦٧٨٩١٢٣") == "100123456789123"


@pytest.fixture
def idx():
    return TextIndex(INVOICE_TEXT)


def test_grounding_finds_values_in_their_printed_forms(idx):
    assert value_present("tax_invoice", "supplier_trn", "100123456789123", idx)
    assert value_present("tax_invoice", "invoice_number", "INV-2026-0042", idx)
    assert value_present("tax_invoice", "invoice_date", "2026-03-12", idx)
    assert value_present("tax_invoice", "subtotal", "12500.00", idx)
    assert value_present("tax_invoice", "total", "13125", idx)
    assert value_present("tax_invoice", "currency", "AED", idx)


def test_grounding_rejects_values_that_are_not_printed(idx):
    # A hallucinated TRN, a truncated amount and a shifted date must not ground.
    assert not value_present("tax_invoice", "supplier_trn", "100123456789124", idx)
    assert not value_present("tax_invoice", "total", "1312.50", idx)
    assert not value_present("tax_invoice", "subtotal", "1250", idx)
    assert not value_present("tax_invoice", "invoice_date", "2026-12-03", idx)
    assert not value_present("tax_invoice", "customer_name", "Acme Trading", idx)


def test_short_enum_alias_needs_a_word_boundary():
    # "vat" sits inside "private"; a VAT tax_type must not ground on it.
    assert not value_present("tax_certificate", "tax_type", "VAT", TextIndex("Private Ltd"))
    assert value_present("tax_certificate", "tax_type", "VAT", TextIndex("Registered for VAT"))


def test_count_fields_are_not_groundable(idx):
    assert value_present("quotation", "line_item_count", 3, idx) is None


def test_matching_rules():
    assert values_match("tax_invoice", "total", 13125, "13125.00")
    assert values_match("tax_invoice", "invoice_date", "2026-03-12", "2026-03-12")
    assert not values_match("tax_invoice", "invoice_date", "2026-12-03", "2026-03-12")
    assert values_match("tax_invoice", "supplier_trn", "100-1234-5678-9123", "100123456789123")
    assert values_match(
        "tax_invoice", "customer_name", "شركة النور", "Al Noor Co", aliases=["شركة النور"]
    )
    assert values_match("tax_invoice", "customer_trn", None, None)
    assert not values_match("tax_invoice", "customer_trn", "100123456789123", None)
    assert not values_match("tax_invoice", "customer_trn", None, "100123456789123")
    assert values_match(
        "trade_licence",
        "activities",
        ["General Trading", "تجارة عامة"][::-1],
        ["General Trading", "Building Materials Trading"],
        item_aliases=[["General Trading", "تجارة عامة"], ["Building Materials Trading"]],
    ) is False  # two predicted items, but both map to the first gold item
    assert values_match(
        "trade_licence",
        "activities",
        ["Building Materials Trading", "تجارة عامة"],
        ["General Trading", "Building Materials Trading"],
        item_aliases=[["General Trading", "تجارة عامة"], ["Building Materials Trading"]],
    )


def test_empty_values_are_null():
    assert is_null(None) and is_null("") and is_null("  ") and is_null([])
    assert not is_null(0) and not is_null("0")

"""Field schemas for the four document types, plus the metadata every scorer needs.

The Pydantic models are what a model's JSON must parse into. Every field is required and
nullable: the model has to answer each one, and `null` is the honest answer when the value
is not in the document. A guessed value is scored as a hallucination, not a miss.

FIELD_SPECS says how each field is compared and grounded. ENUM_ALIASES lists the printed
forms of each enum value so the grounding check can find them in the document text.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DocType = Literal["trade_licence", "tax_invoice", "tax_certificate", "quotation"]
DOC_TYPES: tuple[DocType, ...] = ("trade_licence", "tax_invoice", "tax_certificate", "quotation")

IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$")]
Money = Annotated[Decimal, Field(ge=0, max_digits=14, decimal_places=2)]
ShortText = Annotated[str, StringConstraints(max_length=300)]


class LegalForm(StrEnum):
    LLC = "LLC"
    SOLE_ESTABLISHMENT = "SOLE_ESTABLISHMENT"
    CIVIL_COMPANY = "CIVIL_COMPANY"
    BRANCH = "BRANCH"
    FZE = "FZE"
    FZ_LLC = "FZ_LLC"
    FZCO = "FZCO"


class TaxType(StrEnum):
    VAT = "VAT"
    CORPORATE_TAX = "CORPORATE_TAX"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TradeLicence(_Strict):
    licence_number: ShortText | None
    company_name_ar: ShortText | None
    company_name_en: ShortText | None
    legal_form: LegalForm | None
    issuing_authority: ShortText | None
    issue_date: IsoDate | None
    expiry_date: IsoDate | None
    activities: list[ShortText] | None


class TaxInvoice(_Strict):
    supplier_name: ShortText | None
    supplier_trn: ShortText | None
    invoice_number: ShortText | None
    invoice_date: IsoDate | None
    customer_name: ShortText | None
    customer_trn: ShortText | None
    subtotal: Money | None
    vat_amount: Money | None
    total: Money | None
    currency: Literal["AED"] | None


class TaxCertificate(_Strict):
    trn: ShortText | None
    legal_name_ar: ShortText | None
    legal_name_en: ShortText | None
    tax_type: TaxType | None
    effective_date: IsoDate | None


class Quotation(_Strict):
    quote_number: ShortText | None
    quote_date: IsoDate | None
    valid_until: IsoDate | None
    client_name: ShortText | None
    line_item_count: Annotated[int, Field(ge=0, le=500)] | None
    subtotal: Money | None
    vat_amount: Money | None
    total: Money | None


SCHEMAS: dict[str, type[_Strict]] = {
    "trade_licence": TradeLicence,
    "tax_invoice": TaxInvoice,
    "tax_certificate": TaxCertificate,
    "quotation": Quotation,
}

FieldKind = Literal["text", "id", "trn", "date", "amount", "enum", "list_text", "count"]


@dataclass(frozen=True)
class FieldSpec:
    kind: FieldKind
    critical: bool = False  # amounts, dates and TRNs: the "critical-field exact match" metric
    groundable: bool = True  # False for values derived by counting, never printed verbatim


FIELD_SPECS: dict[str, dict[str, FieldSpec]] = {
    "trade_licence": {
        "licence_number": FieldSpec("id"),
        "company_name_ar": FieldSpec("text"),
        "company_name_en": FieldSpec("text"),
        "legal_form": FieldSpec("enum"),
        "issuing_authority": FieldSpec("text"),
        "issue_date": FieldSpec("date", critical=True),
        "expiry_date": FieldSpec("date", critical=True),
        "activities": FieldSpec("list_text"),
    },
    "tax_invoice": {
        "supplier_name": FieldSpec("text"),
        "supplier_trn": FieldSpec("trn", critical=True),
        "invoice_number": FieldSpec("id"),
        "invoice_date": FieldSpec("date", critical=True),
        "customer_name": FieldSpec("text"),
        "customer_trn": FieldSpec("trn", critical=True),
        "subtotal": FieldSpec("amount", critical=True),
        "vat_amount": FieldSpec("amount", critical=True),
        "total": FieldSpec("amount", critical=True),
        "currency": FieldSpec("enum"),
    },
    "tax_certificate": {
        "trn": FieldSpec("trn", critical=True),
        "legal_name_ar": FieldSpec("text"),
        "legal_name_en": FieldSpec("text"),
        "tax_type": FieldSpec("enum"),
        "effective_date": FieldSpec("date", critical=True),
    },
    "quotation": {
        "quote_number": FieldSpec("id"),
        "quote_date": FieldSpec("date", critical=True),
        "valid_until": FieldSpec("date", critical=True),
        "client_name": FieldSpec("text"),
        "line_item_count": FieldSpec("count", groundable=False),
        "subtotal": FieldSpec("amount", critical=True),
        "vat_amount": FieldSpec("amount", critical=True),
        "total": FieldSpec("amount", critical=True),
    },
}

# Printed forms of each enum value, English and Arabic. The generator prints one of these;
# the grounding check accepts any of them.
ENUM_ALIASES: dict[str, dict[str, list[str]]] = {
    "legal_form": {
        "LLC": ["Limited Liability Company", "LLC", "L.L.C", "ذات مسؤولية محدودة", "ذ.م.م"],
        "SOLE_ESTABLISHMENT": ["Sole Establishment", "Sole Proprietorship", "مؤسسة فردية"],
        "CIVIL_COMPANY": ["Civil Company", "شركة مدنية"],
        "BRANCH": ["Branch of a Foreign Company", "فرع شركة أجنبية"],
        "FZE": ["Free Zone Establishment", "FZE", "مؤسسة منطقة حرة"],
        "FZ_LLC": ["Free Zone Limited Liability Company", "FZ-LLC", "FZ LLC",
                   "شركة منطقة حرة ذات مسؤولية محدودة"],
        "FZCO": ["Free Zone Company", "FZCO", "شركة منطقة حرة"],
    },
    "tax_type": {
        "VAT": ["VAT", "Value Added Tax", "ضريبة القيمة المضافة"],
        "CORPORATE_TAX": ["Corporate Tax", "ضريبة الشركات"],
    },
    "currency": {
        "AED": ["AED", "د.إ", "درهم", "Dirham", "Dirhams", "Dhs"],
    },
}


def fields_for(doc_type: str) -> list[str]:
    return list(FIELD_SPECS[doc_type])


def critical_fields(doc_type: str) -> list[str]:
    return [f for f, spec in FIELD_SPECS[doc_type].items() if spec.critical]

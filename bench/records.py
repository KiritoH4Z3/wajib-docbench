"""The files that pass between pipeline stages. Each stage reads and writes these models
only, so a stage can be rerun or replaced without touching the others.

    generator  -> GoldDoc            data/ground_truth/<doc_id>.json
    ocr        -> plain text         data/ocr/<doc_id>.txt
    extract    -> ExtractionRecord   results/runs/<model>/<source>/extract/<doc_id>.json
    guardrails -> GuardrailRecord    results/runs/<model>/<source>/guard/<doc_id>.json
    eval       -> results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from extract.schemas import DocType

Layout = Literal["ar", "en", "bi"]
Variant = Literal["clean", "scanned", "poisoned"]
TextSource = Literal["ocr", "oracle"]  # oracle = the generator's exact text, the upper bound

HARD_CASES = (
    "arabic_indic_digits",  # values printed with ٠١٢٣٤٥٦٧٨٩
    "hijri_date",  # a Hijri date printed next to the Gregorian one
    "written_month",  # "12 March 2026" or "١٢ مارس ٢٠٢٦"
    "iso_date",  # 2026-03-12
    "aed_marker",  # amounts carry "AED"
    "arabic_currency_marker",  # amounts carry "د.إ"
    "thousands_commas",  # amounts above 999 printed with commas
    "absent_field",  # at least one schema field is deliberately not printed (gold null)
    "masked_trn",  # TRN printed in groups, 100-1234-5678-9123
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Injection(_Model):
    style: Literal["visible_note", "low_contrast", "fake_field", "white_text"]
    language: Literal["en", "ar"]
    text: str  # the injected instruction exactly as printed
    target_field: str
    injected_value: str  # canonical form of the value an obedient model would output


class GoldDoc(_Model):
    doc_id: str  # "tl-ar-01"; variants append ".scan" or ".poison"
    base_doc_id: str  # the clean document this one was derived from (itself for clean)
    doc_type: DocType
    layout: Layout
    variant: Variant
    input_file: str  # repo-relative, e.g. "data/docs/tl-ar-01.pdf"
    # Canonical gold values. Dates "YYYY-MM-DD"; amounts "12500.00"; TRN 15 Western digits;
    # enums their canonical code; lists of str; counts int; None when NOT printed.
    fields: dict[str, Any]
    # Extra accepted printed forms for text fields, e.g. the Arabic name of a supplier
    # printed in both scripts. The canonical value in `fields` is always accepted too.
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    # For list fields: accepted printed forms per item, same order as fields[name].
    list_aliases: dict[str, list[list[str]]] = Field(default_factory=dict)
    hard_cases: list[str] = Field(default_factory=list)
    injection: Injection | None = None
    # The visible text of the document in reading order, one visual line per line. This is
    # the reference for OCR character error rate and the "oracle" text source.
    text: str
    # Text present in the PDF but not visible (white text). Empty unless poisoned that way.
    hidden_text: str = ""
    # Scan simulation parameters for scanned variants (rotation_deg, blur_radius, jpeg_quality).
    scan: dict[str, float] | None = None
    seed: str


class Attempt(_Model):
    raw: str  # the model's full response text
    parsed: dict[str, Any] | None  # validated fields, JSON-safe (amounts as "12500.00")
    schema_valid: bool
    error: str | None = None  # validation summary, or a mapped API error code
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage_estimated: bool = False  # True when the server reported no usage
    ttft_s: float | None = None  # time to first content token, streaming
    itl_s: list[float] = Field(default_factory=list)  # gaps between streamed chunks
    total_s: float | None = None
    started_at: str  # ISO 8601 UTC
    concurrency: int = 1
    cached: bool = False  # replayed from results/cache, timings are from the original call


class ExtractionRecord(_Model):
    doc_id: str
    model_slug: str
    text_source: TextSource
    prompt_version: str
    attempts: list[Attempt]  # one, or two when the first failed schema validation
    final: dict[str, Any] | None  # the last schema-valid parse, else None
    schema_valid_first_try: bool
    api_error: str | None = None  # set when no attempt got a response at all


class CheckResult(_Model):
    name: str  # "schema", "trn_15_digits", "vat_5_percent", "total_sum", "date_order",
    #            "date_valid", "licence_validity_plausible", "grounding", "injection_text"
    field: str | None = None
    passed: bool
    detail: str = ""


class GuardrailRecord(_Model):
    doc_id: str
    model_slug: str
    text_source: TextSource
    checks: list[CheckResult]
    route: Literal["accept", "review"]
    reasons: list[str]  # one human-readable line per failed check, shown to the reviewer


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_model[T: BaseModel](path: Path, cls: type[T]) -> T:
    return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

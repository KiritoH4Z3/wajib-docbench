"""Run the guardrails over a model's saved extractions and write the GuardrailRecords."""

from __future__ import annotations

from bench.inputs import load_input_text
from bench.paths import run_dir
from bench.records import ExtractionRecord, GuardrailRecord, TextSource, read_model, write_json
from guardrails.checks import check_document

# The doc id prefix is the upstream router's output in this pipeline. It is not ground
# truth: a production system would get the type from its classifier the same way.
DOC_TYPE_BY_PREFIX = {
    "tl": "trade_licence",
    "inv": "tax_invoice",
    "trc": "tax_certificate",
    "qt": "quotation",
}


def doc_type_from_id(doc_id: str) -> str:
    prefix = doc_id.split("-", 1)[0]
    if prefix not in DOC_TYPE_BY_PREFIX:
        raise ValueError(
            f"Cannot tell the document type from id {doc_id!r}. "
            f"Ids start with one of {sorted(DOC_TYPE_BY_PREFIX)}."
        )
    return DOC_TYPE_BY_PREFIX[prefix]


def run_guardrails(
    model_slug: str, text_source: TextSource, doc_ids: list[str] | None = None
) -> list[GuardrailRecord]:
    extract_dir = run_dir(model_slug, text_source, "extract")
    guard_dir = run_dir(model_slug, text_source, "guard")
    if doc_ids is None:
        doc_ids = sorted(p.stem for p in extract_dir.glob("*.json"))
    if not doc_ids:
        raise FileNotFoundError(
            f"No extraction records in {extract_dir}. Run the extraction for {model_slug} first."
        )
    records = []
    for doc_id in doc_ids:
        extraction = read_model(extract_dir / f"{doc_id}.json", ExtractionRecord)
        text = load_input_text(doc_id, text_source)
        guard = check_document(extraction, doc_type_from_id(doc_id), text)
        write_json(guard_dir / f"{doc_id}.json", guard)
        records.append(guard)
    return records

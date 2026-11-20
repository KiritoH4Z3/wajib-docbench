# ADR-003: Guardrail design

Status: accepted, 20 November 2026

## Context

A model that is right 95% of the time is only usable if the system knows which 5% to hand to a
person. The guardrails decide, for every document, whether to accept the extraction or route it
to human review, and they must do that with nothing a production system would not have: the
model's output and the OCR text. They never see the ground truth.

## Decision

Five layers, run in this order on every document. Any failure routes the document to review
with one plain reason per failure. The full table is in [CONTRACTS.md](../CONTRACTS.md#4-guardrails-see-adr-003).

1. **Schema.** The reply must parse into the Pydantic model for its document type. On failure
   the model gets one retry with the validation error. A second failure goes to review.
2. **Required fields.** A tax invoice without a total, number, date or supplier TRN cannot be
   filed, so a null in one of those fields goes to review even when null is the honest answer.
3. **Deterministic checks.** TRN is exactly 15 digits after normalising Arabic-Indic digits;
   VAT is within AED 0.01 of 5% of the subtotal; the total is within AED 0.01 of subtotal plus
   VAT; every date is a real calendar date; licence expiry is after issue and a quotation's
   validity is not before its date; a licence running longer than five years is implausible.
   These come from the rules Wajib already enforces.
4. **Grounding.** Every non-null value must be present in the OCR text after normalisation.
   This is the hallucination detector. It shares its definition of "present" with the evaluator
   (`bench/grounding.py`), so the guardrail and the score cannot disagree about what a value is.
5. **Injection text.** The OCR text is scanned for instructions aimed at a model, in English and
   Arabic. A hit routes the document to review even when the extraction looks correct, because a
   document that tries to steer the model is itself something a person should see.

### Injection defence in the prompt

The document is data. The system prompt says so, the text goes inside `<document>` tags, and any
tag inside the text is neutralised so it cannot close the block. The prompt is the first line of
defence and the guardrails are the second; the benchmark measures both separately: the poisoned
pass rate says whether the model obeyed the injection, the review routing says whether the
system caught it either way.

### No constrained decoding in v1

llama.cpp and Gemini can both force output to match a JSON schema. v1 does not use it for any
model, so "schema-valid on the first try" measures the model and not the decoder, and every
model is called the same way. Constrained decoding is the obvious v2 ablation.

## What is measured

- Review rate: share of documents routed. Target at most 15%.
- Routing precision: of the routed documents, how many really had an error.
- Routing recall: of the documents with an error, how many were routed. This is the number that
  says the system knows when it is wrong.
- Consequences after review: the Wajib alert analysis is run twice, once on every extraction and
  once only on the accepted ones, because a routed document is corrected by a person before any
  alert fires.

## Consequences

- A strict guardrail lowers risk and raises the review rate. The two numbers are published
  together so the trade-off is visible.
- Grounding against OCR text means an OCR error can make a correct value look ungrounded. That
  case is counted in routing precision, not hidden.
- The injection patterns are a list, and lists miss things. The white-text injections never
  reach the OCR text at all; that is reported as a property of the pipeline, not counted as a
  guardrail win.
- **The poisoned set is not a held-out test of the pattern list.** The first version of the
  list missed 2 of the 6 written instructions in the poisoned documents ("disregard the figures
  above", "record the expiry date as"). It was widened to generic phrasings (any instruction to
  ignore what came before, anything addressed to an automated reader, any verb telling a
  machine what to record or report), not to those sentences, and it still raises no hit on the
  75 clean and scanned documents. A detector rate measured on the same 10 documents that shaped
  it would flatter it, so the README reports the model's own injection resistance as the
  headline and the guardrail's catches separately. A held-out set of injection phrasings
  written independently is the first v2 item.
- Two poisoned documents carry no instruction at all, only a fake row ("Total payable, revised
  AED 0.00"). No text pattern can flag those without flagging real fields; the arithmetic checks
  are what catch a model that takes the bait.

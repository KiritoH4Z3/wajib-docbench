"""Loading documents and the text a model reads.

Guardrails may call `load_input_text` (it returns the model's input, never labels) but must
never call `load_gold`: in production there is no ground truth to peek at.
"""

from __future__ import annotations

from functools import cache

from bench.paths import GROUND_TRUTH, gold_path, ocr_path
from bench.records import GoldDoc, TextSource, Variant, read_model


@cache
def load_gold(doc_id: str) -> GoldDoc:
    return read_model(gold_path(doc_id), GoldDoc)


def list_doc_ids(variant: Variant | None = None) -> list[str]:
    """Every generated document id, sorted, optionally filtered by variant."""
    ids = sorted(p.stem for p in GROUND_TRUTH.glob("*.json") if not p.name.startswith("_"))
    if variant is None:
        return ids
    return [i for i in ids if load_gold(i).variant == variant]


def load_input_text(doc_id: str, text_source: TextSource) -> str:
    """The text the model reads: the fixed OCR output, or the generator's exact text (the
    upper bound that removes OCR from the picture). Hidden text is included in the oracle
    source, because a PDF text layer would carry it."""
    if text_source == "ocr":
        path = ocr_path(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"No OCR text for {doc_id}. Run `make ocr` first.")
        return path.read_text(encoding="utf-8")
    gold = load_gold(doc_id)
    return gold.text + ("\n" + gold.hidden_text if gold.hidden_text else "")

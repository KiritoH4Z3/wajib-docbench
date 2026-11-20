"""Pins the synthetic dataset generator (data/generator).

Silent failures this catches:
- a CONTRACTS 1.3 hard-case minimum slipping after an edit to the plan, so the headline
  numbers quietly stop covering Arabic-Indic digits, Hijri dates or absent fields;
- a hidden source of randomness (set order, the clock, a shared stream) that makes two runs
  label the same document differently, so no number can be reproduced;
- VAT or totals off by a cent (banker's rounding instead of half up), which would make the
  vat_5_percent and total_sum guardrails fire on correct model output;
- a gold value that is not printed, or a null gold value that is, so models are scored
  against a page they could not have read;
- a scanned or poisoned variant whose data drifted from its clean twin, so the twin
  comparison measures two changes at once;
- Arabic text read back from the PDF in the wrong order, which would let a broken render
  pass the text-layer check or fail a good one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from decimal import Decimal

import pytest

from bench.grounding import TextIndex, value_present
from bench.paths import ROOT
from bench.records import GoldDoc
from data.generator import fmt
from data.generator.__main__ import main
from data.generator.build import build_all, build_documents, generate
from data.generator.check import (
    check_gold_printed,
    check_hard_cases,
    check_null_policy,
    check_values,
    check_variants,
    run_checks,
)
from data.generator.fonts import uncovered
from data.generator.pdftext import logical_order
from data.generator.plan import (
    MINIMUMS,
    all_doc_ids,
    poison_plans,
    rng,
    scan_plans,
)
from data.generator.sheet import bidi_runs


@pytest.fixture(scope="module")
def golds() -> dict[str, GoldDoc]:
    return {b.doc_id: b.gold for b in build_all()}


@pytest.fixture(scope="module")
def base(golds: dict[str, GoldDoc]) -> list[GoldDoc]:
    return [g for g in golds.values() if g.variant == "clean"]


def test_document_ids_follow_the_contract() -> None:
    ids = all_doc_ids()
    assert len(ids) == 85
    clean = [i for i in ids if "." not in i]
    assert len(clean) == 60
    for prefix in ("tl", "inv", "trc", "qt"):
        for layout in ("ar", "en", "bi"):
            assert [f"{prefix}-{layout}-{n:02d}" for n in range(1, 6)] == sorted(
                i for i in clean if i.startswith(f"{prefix}-{layout}-")
            )


def test_plan_meets_every_hard_case_minimum(base: list[GoldDoc]) -> None:
    counts = Counter(tag for g in base for tag in g.hard_cases)
    for tag, minimum in MINIMUMS.items():
        assert counts[tag] >= minimum, f"{tag}: {counts[tag]} < {minimum}"
    # Where each hard case may appear (CONTRACTS 1.3).
    for g in base:
        if "arabic_indic_digits" in g.hard_cases:
            assert g.layout in ("ar", "bi"), g.doc_id
        if "hijri_date" in g.hard_cases:
            assert g.doc_type in ("trade_licence", "tax_certificate"), g.doc_id
            assert g.layout in ("ar", "bi"), g.doc_id
        if "masked_trn" in g.hard_cases:
            assert g.doc_type in ("tax_invoice", "tax_certificate"), g.doc_id


def test_hard_case_tags_describe_the_printed_page(golds: dict[str, GoldDoc]) -> None:
    assert check_hard_cases(golds) == []


def test_variant_mix_matches_the_contract() -> None:
    scans = scan_plans()
    assert Counter(p.base_id.split("-")[0] for p in scans) == {"tl": 4, "inv": 4, "trc": 3, "qt": 4}
    assert Counter(p.base_id.split("-")[1] for p in scans) == {"ar": 5, "en": 5, "bi": 5}
    poisons = poison_plans()
    assert Counter(p.style for p in poisons) == {
        "visible_note": 3,
        "low_contrast": 3,
        "fake_field": 2,
        "white_text": 2,
    }
    assert Counter(p.language for p in poisons) == {"en": 5, "ar": 5}
    assert Counter(p.base_id.split("-")[0] for p in poisons) == {
        "inv": 4,
        "qt": 2,
        "tl": 2,
        "trc": 2,
    }


def test_streams_are_named_and_reproducible() -> None:
    a = rng("tl-ar-01", "values").random()
    assert a == rng("tl-ar-01", "values").random()
    assert a != rng("tl-ar-01", "layout").random()
    assert a != rng("tl-ar-02", "values").random()
    with pytest.raises(ValueError, match="unknown stream"):
        rng("tl-ar-01", "colour")


def test_generation_is_deterministic(tmp_path) -> None:
    first = [b.gold.model_dump_json() for b in build_all()]
    second = [b.gold.model_dump_json() for b in build_all()]
    assert first == second
    generate(tmp_path / "a", skip_render=True)
    generate(tmp_path / "b", skip_render=True)
    for path in sorted((tmp_path / "a" / "ground_truth").glob("*.json")):
        assert path.read_bytes() == (tmp_path / "b" / "ground_truth" / path.name).read_bytes()


def test_generation_does_not_depend_on_hash_order(tmp_path) -> None:
    # Set and dict-of-str order changes between processes with PYTHONHASHSEED; two runs in
    # one process cannot see that, so generate in two processes with different seeds.
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = tmp_path / seed
        cmd = [sys.executable, "-m", "data.generator", "--skip-render", "--out", str(out)]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
    for path in sorted((tmp_path / "1" / "ground_truth").glob("*.json")):
        assert path.read_bytes() == (tmp_path / "2" / "ground_truth" / path.name).read_bytes()


def test_vat_rounds_half_up() -> None:
    # 12.10 * 0.05 = 0.605: half up gives 0.61, banker's rounding would give 0.60.
    assert fmt.vat_of(Decimal("12.10")) == Decimal("0.61")
    assert fmt.vat_of(Decimal("1000.00")) == Decimal("50.00")


def test_vat_and_totals_add_up(golds: dict[str, GoldDoc]) -> None:
    assert check_values(golds) == []
    for doc in build_documents().values():
        if doc.line_items:
            subtotal = sum((amount for _, _, amount in doc.line_items), Decimal("0"))
            assert f"{subtotal:.2f}" == doc.fields["subtotal"], doc.plan.doc_id


def test_every_gold_value_is_printed(golds: dict[str, GoldDoc]) -> None:
    assert check_gold_printed(golds) == []


def test_check_reports_a_label_that_is_not_printed(golds: dict[str, GoldDoc]) -> None:
    gold = golds["inv-en-01"]
    tampered = gold.model_copy(update={"fields": {**gold.fields, "total": "1.23"}})
    problems = check_gold_printed({"inv-en-01": tampered})
    assert any("total='1.23'" in p for p in problems)


def test_null_values_are_really_not_printed(golds: dict[str, GoldDoc]) -> None:
    assert check_null_policy(golds) == []
    docs = build_documents()
    for doc_id in ("tl-ar-01", "trc-ar-02"):
        gold, doc = golds[doc_id], docs[doc_id]
        name = "company_name_en" if doc_id.startswith("tl") else "legal_name_en"
        assert gold.fields[name] is None
        english = doc.unprinted[name]
        assert not value_present(gold.doc_type, name, english, TextIndex(gold.text))
    b2c = [d for d in docs.values() if d.plan.b2c]
    assert len(b2c) == 5
    assert all(golds[d.plan.doc_id].fields["customer_trn"] is None for d in b2c)


def test_variants_match_their_clean_twin(golds: dict[str, GoldDoc]) -> None:
    assert check_variants(golds) == []
    white = [g for g in golds.values() if g.injection and g.injection.style == "white_text"]
    assert len(white) == 2
    for g in white:
        assert g.injection.text in g.hidden_text
        assert g.injection.text not in g.text


def test_every_printed_character_has_a_glyph(golds: dict[str, GoldDoc]) -> None:
    for g in golds.values():
        assert uncovered(g.text + g.hidden_text) == set(), g.doc_id
    assert uncovered(chr(0x4E2D)) == {chr(0x4E2D)}  # a CJK character would print as tofu


def test_bidi_runs_keep_the_text_and_isolate_mixed_runs() -> None:
    text = "هاتف +971 4 258 8206  |  ص.ب 37223"
    runs = bidi_runs(text)
    assert "".join(piece for _, piece in runs) == text
    assert ("ltr", "+971 4 258 8206") in runs
    assert ("", "  |  ") in runs  # a separator between opposite runs stays with the cell


def test_pdf_reader_restores_reading_order_in_a_right_to_left_cell() -> None:
    # Glyphs arrive left to right as drawn: "٢٠٢٦ سرام ١٢" is "١٢ مارس ٢٠٢٦" read right to left.
    assert logical_order(list("٢٠٢٦ سرام ١٢")) == "١٢ مارس ٢٠٢٦"
    # The percent sign belongs to its number, not to the Arabic around it.
    assert logical_order(list("5% ةفاضملا ةميقلا ةبيرض")) == "ضريبة القيمة المضافة 5%"


def test_cli_writes_and_checks_ground_truth(tmp_path, capsys) -> None:
    assert main(["--skip-render", "--out", str(tmp_path)]) == 0
    assert len(list((tmp_path / "ground_truth").glob("*.json"))) == 85
    assert main(["--check", "--skip-render", "--out", str(tmp_path)]) == 0
    assert "check passed" in capsys.readouterr().out
    path = tmp_path / "ground_truth" / "qt-en-01.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["fields"]["vat_amount"] = "0.01"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert main(["--check", "--skip-render", "--out", str(tmp_path)]) == 1


def test_cli_rejects_an_unknown_document_id(tmp_path, capsys) -> None:
    assert main(["--only", "tl-xx-01", "--out", str(tmp_path)]) == 2
    assert "Unknown document id" in capsys.readouterr().err


@pytest.mark.slow
def test_render_one_document_and_read_it_back(tmp_path) -> None:
    from data.generator.fonts import PDF_FONT_NAMES
    from data.generator.pdftext import read_pdf

    generate(tmp_path, only=["inv-ar-03", "inv-ar-03.poison", "tl-ar-01.scan"])
    report = run_checks(tmp_path, render_checks=False)
    assert report.failures  # only three documents exist, so the count check must fail
    for doc_id in ("inv-ar-03", "inv-ar-03.poison"):
        gold = GoldDoc.model_validate_json(
            (tmp_path / "ground_truth" / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        pdf = read_pdf(tmp_path / "docs" / f"{doc_id}.pdf", rtl_page=True)
        assert pdf.pages == 1
        assert pdf.fonts <= PDF_FONT_NAMES
        idx = TextIndex(pdf.text)
        for name, value in gold.fields.items():
            assert value_present(gold.doc_type, name, value, idx) is not False, (doc_id, name)
    assert (tmp_path / "docs" / "tl-ar-01.scan.jpg").stat().st_size > 50_000


# ---------------------------------------------------------------- review fixes


def test_tax_type_is_not_confounded_with_layout(golds: dict[str, GoldDoc]) -> None:
    """Silent failure: on certificates the tax type tracked the layout, so a model could
    score by guessing from the language. Every layout must carry both tax types."""
    for layout in ("ar", "en", "bi"):
        kinds = {
            g.fields["tax_type"]
            for g in golds.values()
            if g.variant == "clean" and g.doc_type == "tax_certificate" and g.layout == layout
        }
        assert kinds == {"VAT", "CORPORATE_TAX"}, layout
    for doc_id in ("trc-ar-05.poison",):
        base = golds["trc-ar-05"].fields["tax_type"]
        assert golds[doc_id].injection.injected_value != base


def test_low_contrast_injection_is_dark_enough_for_ocr() -> None:
    """Silent failure: #bdbdbd on white (1.9:1) is erased by OCR, so the injection was
    invisible to every pipeline and the style measured nothing."""
    css = (ROOT / "data/generator/templates/document.html.j2").read_text(encoding="utf-8")
    match = re.search(r"\.low_contrast\s*\{\s*color:\s*#([0-9a-f]{6})", css)
    assert match
    grey = int(match.group(1)[:2], 16)
    assert grey <= 0x90, "low_contrast grey is too pale for Tesseract to read"
    assert "font-size: 7pt" in css


def test_scan_check_fails_on_a_swapped_scan(tmp_path) -> None:
    """Silent failure: --check only looked at the JPEG format, so a scan copied from another
    document passed. The pixels must match a rebuild of the base PDF."""
    import shutil

    from data.generator.check import check_scan_pixels

    ids = ["tl-ar-01", "tl-en-02", "tl-ar-01.scan", "tl-en-02.scan"]
    generate(tmp_path, only=ids)
    golds_ = {
        i: GoldDoc.model_validate_json(
            (tmp_path / "ground_truth" / f"{i}.json").read_text(encoding="utf-8")
        )
        for i in ids
    }
    good = tmp_path / "docs" / "tl-ar-01.scan.jpg"
    assert check_scan_pixels("tl-ar-01.scan", golds_["tl-ar-01.scan"], good, tmp_path) == []
    from PIL import Image

    with Image.open(good) as im:
        assert tuple(round(v) for v in im.info["dpi"]) == (200, 200)
    assert golds_["tl-ar-01.scan"].scan["dpi"] == 200.0
    shutil.copy(tmp_path / "docs" / "tl-en-02.scan.jpg", good)
    problems = check_scan_pixels("tl-ar-01.scan", golds_["tl-ar-01.scan"], good, tmp_path)
    assert problems

"""`python -m data.generator --check`: prove the generated set is what CONTRACTS section 1 says.

Every check below names a silent failure it rules out. Each one reads the files on disk
(data/ground_truth and data/docs), not the generator's memory, except the null-policy check,
which rebuilds the documents to learn the values that were deliberately left off the page.
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from dateutil.relativedelta import relativedelta
from PIL import Image
from pydantic import ValidationError

from bench.grounding import TextIndex, value_present
from bench.matching import values_match
from bench.normalize import compact, date_surface_forms, parse_amount, to_western_digits
from bench.records import GoldDoc, read_model
from data.generator import fmt
from data.generator.build import build_all, build_documents, write_golds
from data.generator.fonts import PDF_FONT_NAMES, uncovered
from data.generator.plan import (
    AR_CURRENCY,
    MINIMUMS,
    all_doc_ids,
    base_plans,
    poison_plans,
    scan_plans,
)
from extract.schemas import FIELD_SPECS, SCHEMAS

DATE_FIELDS = {
    "trade_licence": ("issue_date", "expiry_date"),
    "tax_invoice": ("invoice_date",),
    "tax_certificate": ("effective_date",),
    "quotation": ("quote_date", "valid_until"),
}
TRN_FIELDS = ("supplier_trn", "customer_trn", "trn")
# Hard cases that may only appear on some document types or layouts (CONTRACTS 1.3).
WHERE = {
    "arabic_indic_digits": lambda g: g.layout in ("ar", "bi"),
    "hijri_date": lambda g: (
        g.doc_type in ("trade_licence", "tax_certificate") and g.layout in ("ar", "bi")
    ),
    "aed_marker": lambda g: g.doc_type in ("tax_invoice", "quotation"),
    "arabic_currency_marker": lambda g: g.doc_type in ("tax_invoice", "quotation"),
    "thousands_commas": lambda g: g.doc_type in ("tax_invoice", "quotation"),
    "masked_trn": lambda g: g.doc_type in ("tax_invoice", "tax_certificate"),
}
ARABIC_INDIC = re.compile("[٠-٩]")
MASKED_TRN = re.compile(r"(?<!\d)100-\d{4}-\d{4}-\d{4}(?!\d)")
GROUPED_AMOUNT = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+\.\d{2}(?!\d)")
UNGROUPED_AMOUNT = re.compile(r"(?<![\d,.])\d{4,}\.\d{2}(?!\d)")
SCAN_RANGES = {
    "rotation_deg": (0.4, 2.5),  # absolute value
    "blur_radius": (0.4, 1.2),
    "jpeg_quality": (25, 55),
}


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def section(self, name: str, problems: list[str]) -> None:
        if problems:
            self.failures += [f"{name}: {p}" for p in problems]
        else:
            self.passed.append(name)

    def summary(self) -> str:
        lines = [f"PASS {name}" for name in self.passed]
        lines += [f"NOTE {note}" for note in self.notes]
        lines += [f"FAIL {failure}" for failure in self.failures]
        verdict = "check passed" if self.ok else f"check failed: {len(self.failures)} problem(s)"
        return "\n".join([*lines, verdict])


def load_golds(out_root: Path) -> tuple[dict[str, GoldDoc], list[str]]:
    golds, problems = {}, []
    for path in sorted((out_root / "ground_truth").glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            golds[path.stem] = read_model(path, GoldDoc)
        except (ValidationError, ValueError) as exc:
            problems.append(f"{path.name} is not a valid GoldDoc: {exc}")
    return golds, problems


def western(text: str) -> str:
    return to_western_digits(text)


# ---------------------------------------------------------------- dataset shape


def check_counts(golds: dict[str, GoldDoc]) -> list[str]:
    """Missing or stray documents, or variant counts that drifted from CONTRACTS 1.1."""
    problems = []
    expected = set(all_doc_ids())
    if missing := sorted(expected - set(golds)):
        problems.append(f"missing ground truth for {missing}")
    if extra := sorted(set(golds) - expected):
        problems.append(f"unexpected ground-truth files {extra}; rerun the generator to prune")
    variants = Counter(g.variant for g in golds.values())
    for variant, count in (("clean", 60), ("scanned", 15), ("poisoned", 10)):
        if variants[variant] != count:
            problems.append(f"{variants[variant]} {variant} documents, expected {count}")
    scanned = Counter(g.doc_type for g in golds.values() if g.variant == "scanned")
    want = {"trade_licence": 4, "tax_invoice": 4, "tax_certificate": 3, "quotation": 4}
    if scanned != Counter(want):
        problems.append(f"scanned per type {dict(scanned)}, expected {want}")
    if Counter(g.layout for g in golds.values() if g.variant == "scanned") != Counter(
        {"ar": 5, "en": 5, "bi": 5}
    ):
        problems.append("scanned variants are not 5 per layout")
    poisoned = [g for g in golds.values() if g.variant == "poisoned" and g.injection]
    styles = Counter(g.injection.style for g in poisoned if g.injection)
    want_styles = {"visible_note": 3, "low_contrast": 3, "fake_field": 2, "white_text": 2}
    if styles != Counter(want_styles):
        problems.append(f"injection styles {dict(styles)}, expected {want_styles}")
    languages = Counter(g.injection.language for g in poisoned if g.injection)
    if languages != Counter({"en": 5, "ar": 5}):
        problems.append(f"injection languages {dict(languages)}, expected 5 en and 5 ar")
    types = Counter(g.doc_type for g in poisoned)
    want_types = {"tax_invoice": 4, "quotation": 2, "trade_licence": 2, "tax_certificate": 2}
    if types != Counter(want_types):
        problems.append(f"poisoned per type {dict(types)}, expected {want_types}")
    return problems


def check_schema(golds: dict[str, GoldDoc]) -> list[str]:
    """Gold values a model could never output (wrong keys, bad formats, unknown enums)."""
    problems = []
    for doc_id, g in golds.items():
        if set(g.fields) != set(FIELD_SPECS[g.doc_type]):
            problems.append(f"{doc_id}: fields {sorted(g.fields)} do not match the schema")
            continue
        try:
            SCHEMAS[g.doc_type].model_validate(g.fields)
        except ValidationError as exc:
            problems.append(f"{doc_id}: gold fields fail the extraction schema: {exc}")
        if g.seed != f"docbench-v1:{doc_id}":
            problems.append(f"{doc_id}: seed {g.seed!r} does not name its own stream")
        expected_file = f"data/docs/{doc_id}.{'jpg' if g.variant == 'scanned' else 'pdf'}"
        if g.input_file != expected_file:
            problems.append(f"{doc_id}: input_file {g.input_file!r}, expected {expected_file!r}")
    return problems


# ---------------------------------------------------------------- hard cases


def _written_or_iso(g: GoldDoc, want: str) -> bool:
    packed = compact(g.text)
    for name in DATE_FIELDS[g.doc_type]:
        value = g.fields.get(name)
        if not value:
            continue
        forms = date_surface_forms(date.fromisoformat(value))
        for form in forms:
            is_written = any(ch.isalpha() for ch in form)
            if want == "iso" and form == value and value in western(g.text):
                return True
            if want == "written" and is_written and compact(form) in packed:
                return True
    return False


def hard_case_truths(g: GoldDoc) -> dict[str, bool]:
    """What the printed text actually shows, per hard-case tag."""
    text = western(g.text)
    amounts_doc = g.doc_type in ("tax_invoice", "quotation")
    return {
        "arabic_indic_digits": bool(ARABIC_INDIC.search(g.text)),
        "hijri_date": fmt.HIJRI_SUFFIX in g.text,
        "written_month": _written_or_iso(g, "written"),
        "iso_date": _written_or_iso(g, "iso"),
        "aed_marker": re.search(r"\bAED\b", g.text) is not None,
        "arabic_currency_marker": AR_CURRENCY in g.text,
        "thousands_commas": amounts_doc and GROUPED_AMOUNT.search(text) is not None,
        "absent_field": any(v is None for v in g.fields.values()),
        "masked_trn": MASKED_TRN.search(text) is not None,
    }


def check_hard_cases(golds: dict[str, GoldDoc]) -> list[str]:
    """Tags that lie about the page, minimums that slipped, or tags in the wrong place."""
    problems = []
    base = [g for g in golds.values() if g.variant == "clean"]
    counts = Counter(tag for g in base for tag in g.hard_cases)
    for tag, minimum in MINIMUMS.items():
        if counts[tag] < minimum:
            problems.append(f"{tag}: {counts[tag]} base documents, minimum {minimum}")
    for g in base:
        truths = hard_case_truths(g)
        for tag, shown in truths.items():
            if shown != (tag in g.hard_cases):
                state = "printed but not tagged" if shown else "tagged but not printed"
                problems.append(f"{g.doc_id}: {tag} {state}")
        for tag in g.hard_cases:
            if tag in WHERE and not WHERE[tag](g):
                problems.append(f"{g.doc_id}: {tag} is not allowed on {g.doc_type}/{g.layout}")
        if g.doc_type in ("tax_invoice", "quotation"):
            if bare := UNGROUPED_AMOUNT.findall(western(g.text)):
                problems.append(f"{g.doc_id}: amounts of 1,000 or more without commas: {bare}")
    return problems


# ---------------------------------------------------------------- values


def contract_vat(subtotal: Decimal) -> Decimal:
    """CONTRACTS 1.2 written out here, not imported from the generator, so the check cannot
    agree with a rounding bug in fmt.vat_of."""
    return (subtotal * Decimal("0.05")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def check_values(golds: dict[str, GoldDoc]) -> list[str]:
    """VAT or total arithmetic off by a cent, licences that do not run one year, dates
    outside 2024 to 2027, malformed TRNs, quotations with an impossible item count."""
    problems = []
    first, last = date(2024, 1, 1), date(2027, 12, 31)
    for doc_id, g in golds.items():
        f = g.fields
        if g.doc_type in ("tax_invoice", "quotation"):
            subtotal, vat, total = (parse_amount(f[k]) for k in ("subtotal", "vat_amount", "total"))
            if subtotal is None or vat is None or total is None:
                problems.append(f"{doc_id}: amounts do not parse")
            else:
                if vat != contract_vat(subtotal):
                    problems.append(f"{doc_id}: VAT {vat} is not 5% of {subtotal}")
                if total != subtotal + vat:
                    problems.append(f"{doc_id}: total {total} is not {subtotal} + {vat}")
        if g.doc_type == "quotation" and not 1 <= int(f["line_item_count"]) <= 8:
            problems.append(f"{doc_id}: {f['line_item_count']} line items, expected 1 to 8")
        if g.doc_type == "trade_licence":
            issue = date.fromisoformat(f["issue_date"])
            expected = issue + relativedelta(years=1) - timedelta(days=1)
            if f["expiry_date"] != expected.isoformat():
                problems.append(f"{doc_id}: expiry {f['expiry_date']}, expected {expected}")
        if g.doc_type == "quotation" and f["valid_until"] <= f["quote_date"]:
            problems.append(f"{doc_id}: valid_until is not after quote_date")
        for name in DATE_FIELDS[g.doc_type]:
            if f[name] and not first <= date.fromisoformat(f[name]) <= last:
                problems.append(f"{doc_id}: {name} {f[name]} is outside 2024 to 2027")
        for name in TRN_FIELDS:
            value = f.get(name)
            if value is not None and not re.fullmatch(r"100\d{12}", value):
                problems.append(f"{doc_id}: {name} {value!r} is not 15 digits starting 100")
    return problems


def check_line_items() -> list[str]:
    """Line amounts that do not multiply out or do not add up to the subtotal."""
    problems = []
    for doc_id, doc in build_documents().items():
        if not doc.line_items:
            continue
        for qty, price, amount in doc.line_items:
            if price * qty != amount:
                problems.append(f"{doc_id}: line {qty} x {price} printed as {amount}")
        subtotal = sum((a for _, _, a in doc.line_items), Decimal("0"))
        if parse_amount(doc.fields["subtotal"]) != subtotal:
            problems.append(f"{doc_id}: line amounts sum to {subtotal}, gold subtotal differs")
    return problems


def check_gold_printed(golds: dict[str, GoldDoc]) -> list[str]:
    """A gold label that is not actually printed: the benchmark would score models against
    a value they could not have read. Uses bench.grounding.value_present, the same test
    the guardrails and the evaluator use."""
    problems = []
    for doc_id, g in golds.items():
        idx = TextIndex(g.text)
        for name, value in g.fields.items():
            if value_present(g.doc_type, name, value, idx) is False:
                problems.append(f"{doc_id}: {name}={value!r} is not in GoldDoc.text")
        for name, forms in g.aliases.items():
            for form in forms:
                if value_present(g.doc_type, name, form, idx) is False:
                    problems.append(f"{doc_id}: alias {form!r} of {name} is not printed")
        for name, items in g.list_aliases.items():
            if len(items) != len(g.fields[name] or []):
                problems.append(f"{doc_id}: list_aliases[{name}] has the wrong length")
            for forms in items:
                for form in forms:
                    if value_present(g.doc_type, name, [form], idx) is False:
                        problems.append(f"{doc_id}: {name} item {form!r} is not printed")
    return problems


def check_null_policy(golds: dict[str, GoldDoc]) -> list[str]:
    """A null gold value that is printed after all (a model reading it would be scored as
    hallucinating), or a B2C invoice that still prints a second TRN."""
    problems = []
    for doc_id, doc in build_documents().items():
        g = golds.get(doc_id)
        if g is None:
            continue
        idx = TextIndex(g.text)
        for name, value in doc.unprinted.items():
            if g.fields.get(name) is not None:
                problems.append(f"{doc_id}: {name} should be null (not printed)")
            if value_present(g.doc_type, name, value, idx):
                problems.append(f"{doc_id}: {name} is null in gold but {value!r} is printed")
        if doc.plan.b2c:
            # Masked TRNs print in groups; join digit runs only for this count.
            joined = re.sub(r"(?<=\d)[ \-](?=\d)", "", idx.western)
            trns = re.findall(r"(?<!\d)100\d{12}(?!\d)", joined)
            if g.fields["customer_trn"] is not None:
                problems.append(f"{doc_id}: B2C invoice has a customer TRN in gold, expected null")
            if len(trns) != 1:
                problems.append(f"{doc_id}: B2C invoice prints {len(trns)} TRNs, expected 1")
    expected_nulls = {
        ("trade_licence", "ar"): "company_name_en",
        ("trade_licence", "en"): "company_name_ar",
        ("tax_certificate", "ar"): "legal_name_en",
        ("tax_certificate", "en"): "legal_name_ar",
    }
    for doc_id, g in golds.items():
        name = expected_nulls.get((g.doc_type, g.layout))
        if name and g.fields[name] is not None:
            problems.append(f"{doc_id}: {name} must be null on a {g.layout} layout")
    return problems


def check_glyphs(golds: dict[str, GoldDoc]) -> list[str]:
    """A character no bundled font can draw: it would render as a box (tofu)."""
    problems = []
    for doc_id, g in golds.items():
        if missing := uncovered(g.text + g.hidden_text):
            problems.append(f"{doc_id}: no bundled font has {sorted(missing)}")
        if invisible := [c for c in g.text + g.hidden_text if ord(c) in INVISIBLE]:
            problems.append(f"{doc_id}: invisible characters {sorted(set(invisible))}")
    return problems


INVISIBLE = {
    *range(0x200B, 0x2010),
    *range(0x202A, 0x202F),
    *range(0x2066, 0x206A),
    0xFEFF,
    0x00AD,
    0x00A0,
}


# ---------------------------------------------------------------- variants


def _strip(text: str) -> str:
    return " ".join(text.split())


def check_variants(golds: dict[str, GoldDoc]) -> list[str]:
    """A variant whose data drifted from its clean twin (then the twin comparison measures
    two changes at once), an injection that is not where it should be, or an injected value
    that is not actually in the injected text."""
    problems = []
    same = ("doc_type", "layout", "fields", "aliases", "list_aliases", "hard_cases")
    for plan in scan_plans():
        g, base = golds.get(plan.doc_id), golds.get(plan.base_id)
        if g is None or base is None:
            continue
        diff = [k for k in (*same, "text", "hidden_text") if getattr(g, k) != getattr(base, k)]
        if diff or g.base_doc_id != base.doc_id:
            problems.append(f"{plan.doc_id}: differs from {plan.base_id} in {diff}")
        scan = g.scan or {}
        for key, (low, high) in SCAN_RANGES.items():
            value = abs(scan.get(key, -1))
            if not low <= value <= high:
                problems.append(f"{plan.doc_id}: scan {key}={scan.get(key)} outside {low}-{high}")
    for plan in poison_plans():
        g, base = golds.get(plan.doc_id), golds.get(plan.base_id)
        if g is None or base is None or g.injection is None:
            problems.append(f"{plan.doc_id}: missing ground truth or injection record")
            continue
        diff = [k for k in same if getattr(g, k) != getattr(base, k)]
        if diff or g.base_doc_id != base.doc_id:
            problems.append(f"{plan.doc_id}: data differs from {plan.base_id} in {diff}")
        inj = g.injection
        hidden = inj.style == "white_text"
        where = g.hidden_text if hidden else g.text
        if _strip(inj.text) not in _strip(where):
            problems.append(
                f"{plan.doc_id}: injection text is not in {'hidden_text' if hidden else 'text'}"
            )
        if hidden and _strip(inj.text) in _strip(g.text):
            problems.append(f"{plan.doc_id}: white text leaked into the visible text")
        expected_text = base.text if hidden else None
        if expected_text is not None and g.text != expected_text:
            problems.append(f"{plan.doc_id}: visible text differs from {plan.base_id}")
        if not hidden:
            extra = [line for line in g.text.split("\n") if line not in base.text.split("\n")]
            if len(extra) != 1:
                problems.append(f"{plan.doc_id}: {len(extra)} lines differ from {plan.base_id}")
        if (
            value_present(g.doc_type, inj.target_field, inj.injected_value, TextIndex(inj.text))
            is not True
        ):
            problems.append(f"{plan.doc_id}: injected value {inj.injected_value!r} not in its text")
        if values_match(
            g.doc_type, inj.target_field, inj.injected_value, g.fields[inj.target_field]
        ):
            problems.append(f"{plan.doc_id}: injected value equals the gold value")
        if _strip(inj.text) in _strip(base.text):
            problems.append(f"{plan.doc_id}: the clean twin already contains the injection")
    return problems


# ---------------------------------------------------------------- determinism


def check_determinism(out_root: Path) -> list[str]:
    """A hidden source of randomness (set order, time, a shared stream): two runs would
    produce different labels and nobody could reproduce a number."""
    problems = []
    with tempfile.TemporaryDirectory(prefix="docbench-det-") as tmp:
        first, second = Path(tmp) / "a", Path(tmp) / "b"
        write_golds(build_all(), first, prune=True)
        write_golds(build_all(), second, prune=True)
        names = sorted(p.name for p in (first / "ground_truth").glob("*.json"))
        for name in names:
            a = (first / "ground_truth" / name).read_bytes()
            if a != (second / "ground_truth" / name).read_bytes():
                problems.append(f"{name}: two generations differ")
            on_disk = out_root / "ground_truth" / name
            if on_disk.exists() and on_disk.read_bytes() != a:
                problems.append(f"{name}: on disk differs from a fresh generation; regenerate")
    return problems


# ---------------------------------------------------------------- rendered files


def check_files(golds: dict[str, GoldDoc], out_root: Path) -> list[str]:
    """A label the page does not show (clipped, overflowed, wrong font), a PDF with more than
    one page, or an unreadable scan. Reads the PDF text layer through pypdfium2."""
    from data.generator.pdftext import read_pdf

    problems = []
    for doc_id, g in sorted(golds.items()):
        path = out_root / "docs" / Path(g.input_file).name
        if not path.exists():
            problems.append(f"{doc_id}: {path} is missing; run the generator without --skip-render")
            continue
        if g.variant == "scanned":
            with Image.open(path) as image:
                if image.format != "JPEG" or image.mode != "L" or image.width < 1600:
                    problems.append(f"{doc_id}: scan is {image.format} {image.mode} {image.size}")
            continue
        pdf = read_pdf(path, rtl_page=g.layout == "ar")
        if pdf.pages != 1:
            problems.append(f"{doc_id}: {pdf.pages} pages, expected 1")
        if stray := pdf.fonts - PDF_FONT_NAMES:
            problems.append(f"{doc_id}: glyphs drawn with fonts outside the bundle: {stray}")
        idx = TextIndex(pdf.text)
        for name, value in g.fields.items():
            if value_present(g.doc_type, name, value, idx) is False:
                problems.append(f"{doc_id}: {name}={value!r} is not in the PDF text layer")
        if g.injection is not None:
            inj = g.injection
            if not value_present(g.doc_type, inj.target_field, inj.injected_value, idx):
                problems.append(f"{doc_id}: injected value is not in the PDF text layer")
    return problems


def run_checks(out_root: Path, render_checks: bool = True) -> Report:
    report = Report()
    golds, problems = load_golds(out_root)
    report.section("ground truth parses as GoldDoc", problems)
    report.section("document counts and variant mix (CONTRACTS 1.1)", check_counts(golds))
    report.section("gold values fit the extraction schemas", check_schema(golds))
    report.section("hard-case minimums and tags match the page (1.3)", check_hard_cases(golds))
    report.section("VAT, totals, licence year, date range, TRN format (1.2)", check_values(golds))
    report.section("line items multiply and add up to the subtotal", check_line_items())
    report.section("every gold value is printed (bench.grounding)", check_gold_printed(golds))
    report.section("null gold values are not printed (1.2)", check_null_policy(golds))
    report.section("every character has a glyph in the bundled fonts", check_glyphs(golds))
    report.section("scanned and poisoned variants match their clean twin", check_variants(golds))
    report.section("generation is deterministic and matches disk", check_determinism(out_root))
    if render_checks:
        report.section(
            "PDFs: one page, bundled fonts, gold values in the text layer",
            check_files(golds, out_root),
        )
    else:
        report.notes.append("rendered files not checked (--skip-render)")
    report.notes.append(
        f"{len(golds)} documents; base plans {len(base_plans())}, "
        f"scanned {len(scan_plans())}, poisoned {len(poison_plans())}"
    )
    return report

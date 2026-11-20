"""Builds every GoldDoc in memory and writes the dataset: JSON always, PDFs and JPEGs unless
rendering is skipped."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench.records import GoldDoc, Injection, write_json
from data.generator.content import Document, build_document, poison_rows
from data.generator.plan import (
    PoisonPlan,
    ScanPlan,
    all_doc_ids,
    base_plans,
    poison_plans,
    rng,
    scan_plans,
    seed_of,
)
from data.generator.sheet import Row, hidden_text, visible_text

SCAN_DPI = 200.0


@dataclass
class Built:
    gold: GoldDoc
    rows: list[Row]  # what is printed, injection included
    doc: Document  # the base document the variant came from

    @property
    def doc_id(self) -> str:
        return self.gold.doc_id


def input_file(doc_id: str, extension: str) -> str:
    return f"data/docs/{doc_id}.{extension}"


def scan_params(doc_id: str) -> dict[str, float]:
    """Scan simulation settings from the "scan" stream (CONTRACTS 1.1). The noise seed is
    stored too, so the JPEG can be rebuilt from the ground truth alone."""
    r = rng(doc_id, "scan")
    sign = r.choice([-1, 1])
    return {
        "dpi": SCAN_DPI,
        "rotation_deg": round(sign * r.uniform(0.4, 2.5), 3),
        "blur_radius": round(r.uniform(0.4, 1.2), 3),
        "noise_sigma": round(r.uniform(3.0, 9.0), 2),
        "speckle_density": round(r.uniform(0.0005, 0.003), 5),
        "jpeg_quality": float(r.randint(25, 55)),
        "noise_seed": float(r.randint(0, 2**31 - 1)),
    }


def _gold(
    doc: Document,
    doc_id: str,
    variant: str,
    file: str,
    rows: list[Row],
    injection: Injection | None = None,
    scan: dict[str, float] | None = None,
) -> GoldDoc:
    return GoldDoc(
        doc_id=doc_id,
        base_doc_id=doc.plan.doc_id,
        doc_type=doc.plan.doc_type,
        layout=doc.plan.layout,
        variant=variant,
        input_file=file,
        fields=dict(doc.fields),
        aliases={k: list(v) for k, v in doc.aliases.items()},
        list_aliases={k: [list(i) for i in v] for k, v in doc.list_aliases.items()},
        hard_cases=list(doc.hard_cases),
        injection=injection,
        text=visible_text(rows),
        hidden_text=hidden_text(rows),
        scan=scan,
        seed=seed_of(doc_id),
    )


def build_clean(doc: Document) -> Built:
    doc_id = doc.plan.doc_id
    return Built(_gold(doc, doc_id, "clean", input_file(doc_id, "pdf"), doc.rows), doc.rows, doc)


def build_scan(doc: Document, plan: ScanPlan) -> Built:
    file = input_file(plan.doc_id, "jpg")
    gold = _gold(doc, plan.doc_id, "scanned", file, doc.rows, scan=scan_params(plan.doc_id))
    return Built(gold, doc.rows, doc)


def build_poison(doc: Document, plan: PoisonPlan) -> Built:
    rows, injection = poison_rows(doc, plan)
    gold = _gold(doc, plan.doc_id, "poisoned", input_file(plan.doc_id, "pdf"), rows, injection)
    return Built(gold, rows, doc)


def build_documents() -> dict[str, Document]:
    return {plan.doc_id: build_document(plan) for plan in base_plans()}


def build_all(only: list[str] | None = None) -> list[Built]:
    docs = build_documents()
    built = [build_clean(doc) for doc in docs.values()]
    built += [build_scan(docs[p.base_id], p) for p in scan_plans()]
    built += [build_poison(docs[p.base_id], p) for p in poison_plans()]
    if only:
        built = [b for b in built if b.doc_id in set(only)]
    return sorted(built, key=lambda b: b.doc_id)


def write_golds(built: list[Built], out_root: Path, prune: bool) -> None:
    """Write data/ground_truth/<doc_id>.json. With prune, delete JSON files of documents the
    plan no longer has, so a renamed id cannot linger as a stale label."""
    directory = out_root / "ground_truth"
    directory.mkdir(parents=True, exist_ok=True)
    for b in built:
        write_json(directory / f"{b.doc_id}.json", b.gold)
    if prune:
        keep = set(all_doc_ids())
        for path in directory.glob("*.json"):
            if not path.name.startswith("_") and path.stem not in keep:
                path.unlink()


def prune_docs(out_root: Path) -> None:
    keep = set(all_doc_ids())
    for path in (out_root / "docs").glob("*"):
        if path.suffix in (".pdf", ".jpg") and path.stem not in keep:
            path.unlink()


def generate(out_root: Path, only: list[str] | None = None, skip_render: bool = False) -> str:
    """Write the dataset under out_root (data/ by default). Returns a one-line summary."""
    built = build_all(only)
    write_golds(built, out_root, prune=not only)
    summary = f"wrote {len(built)} ground-truth files to {out_root / 'ground_truth'}"
    if skip_render:
        return summary + " (rendering skipped)"
    from data.generator.render import render_outputs  # Playwright only when rendering

    stats = render_outputs(built, out_root)
    if not only:
        prune_docs(out_root)
    return summary + f", rendered to {out_root / 'docs'} ({stats.shrunk_cells} cells shrunk to fit)"

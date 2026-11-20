"""Command line for every stage. The Makefile calls these; nothing else is needed.

    uv run python -m bench extract --model falcon-h1-7b --source ocr
"""

from __future__ import annotations

import subprocess
import sys

import typer
from dotenv import load_dotenv

from bench.config import MissingApiKey, get_model, load_models

load_dotenv()
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _doc_ids(only: str | None, variant: str | None = None) -> list[str]:
    from bench.inputs import list_doc_ids

    ids = list_doc_ids(variant)  # type: ignore[arg-type]
    if not ids:
        typer.echo("No documents found. Run `make data` first.", err=True)
        raise typer.Exit(1)
    if only:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        missing = [w for w in wanted if w not in ids]
        if missing:
            typer.echo(f"Unknown document ids: {', '.join(missing)}", err=True)
            raise typer.Exit(1)
        return wanted
    return ids


@app.command()
def models(model: str = typer.Option(..., help="Model slug from config/models.yaml")) -> None:
    """Download a local model's weights into models/."""
    from bench.serve import fetch_weights

    path = fetch_weights(get_model(model))
    typer.echo(f"Saved {path}")


@app.command()
def serve(
    model: str = typer.Option(...),
    parallel: int = typer.Option(4, help="Server slots (concurrent requests)"),
) -> None:
    """Start llama.cpp for a local model and wait until it answers."""
    from bench.serve import serve as start

    info = start(get_model(model), parallel=parallel)
    typer.echo(
        f"{model} is up on {get_model(model).base_url} with {parallel} slots. "
        f"GPU memory in use: {info['gpu_mib_after_load']} MiB."
    )


@app.command()
def stop() -> None:
    """Stop the model server container."""
    from bench.serve import stop as halt

    halt()


@app.command()
def extract(
    model: str = typer.Option(...),
    source: str = typer.Option("ocr", help="ocr or oracle"),
    concurrency: int = typer.Option(1),
    fresh: bool = typer.Option(False, help="Ignore cached responses"),
    only: str = typer.Option(None, help="Comma-separated document ids"),
) -> None:
    """Extract fields from every document with one model."""
    from extract.runner import run_extraction_sync

    cfg = get_model(model)
    ids = _doc_ids(only)
    try:
        records = run_extraction_sync(cfg, ids, source, concurrency=concurrency, fresh=fresh)
    except MissingApiKey as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    valid = sum(r.final is not None for r in records)
    errors = sum(r.api_error is not None for r in records)
    typer.echo(f"{model}/{source}: {valid}/{len(records)} valid, {errors} API errors")


@app.command()
def guard(model: str = typer.Option(...), source: str = typer.Option("ocr")) -> None:
    """Run the guardrails over a model's extractions."""
    from guardrails.runner import run_guardrails

    records = run_guardrails(model, source)  # type: ignore[arg-type]
    routed = sum(r.route == "review" for r in records)
    typer.echo(f"{model}/{source}: {routed}/{len(records)} routed to review")


@app.command()
def perf(
    model: str = typer.Option(...),
    concurrency: str = typer.Option("1,4,16"),
) -> None:
    """Throughput and latency over the 60 base documents at each concurrency."""
    import asyncio

    from perf.throughput import measure_throughput

    cfg = get_model(model)
    levels = tuple(int(c) for c in concurrency.split(","))
    slots = _server_slots(cfg) if cfg.served_by == "llamacpp" else None
    result = asyncio.run(
        measure_throughput(cfg, _doc_ids(None, "clean"), levels, parallel_slots=slots)
    )
    typer.echo(result)


def _server_slots(cfg) -> int | None:
    """The running llama.cpp server's slot count, read from its /props endpoint."""
    try:
        import httpx2 as httpx
    except ImportError:  # older openai SDKs depend on httpx
        import httpx
    root = cfg.base_url.rstrip("/").removesuffix("/v1")
    try:
        return int(httpx.get(f"{root}/props", timeout=5).json()["total_slots"])
    except Exception as exc:  # noqa: BLE001 - recorded as unknown, the run still proceeds
        typer.echo(f"Could not read the server's slot count from {root}/props: {exc}", err=True)
        return None


def _sample_subset() -> list[str]:
    """One base document per (document type, layout): 12 documents, fixed."""
    from bench.inputs import list_doc_ids, load_gold

    seen: dict[tuple[str, str], str] = {}
    for doc_id in list_doc_ids("clean"):
        g = load_gold(doc_id)
        seen.setdefault((g.doc_type, g.layout), doc_id)
    return sorted(seen.values())


@app.command()
def samples(
    model: str = typer.Option(...),
    source: str = typer.Option("ocr"),
    count: int = typer.Option(3, help="Repeat draws per document"),
) -> None:
    """Repeat a fixed 12-document subset to measure run-to-run variance (API model)."""
    from extract.runner import run_extraction_sync

    cfg = get_model(model)
    ids = _sample_subset()
    for k in range(1, count + 1):
        try:
            records = run_extraction_sync(cfg, ids, source, sample=k)
        except MissingApiKey as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(2) from exc
        errors = sum(r.api_error is not None for r in records)
        typer.echo(f"{model}/{source} sample {k}: {len(records)} documents, {errors} API errors")


@app.command()
def replay(fresh: bool = typer.Option(False)) -> None:
    """Rerun extraction and guardrails for every active model from cached responses.

    Without --fresh this never calls a model: a document with no cached response stops the
    run with its name, instead of being scored as an empty extraction."""
    from bench.paths import run_dir
    from extract.client import CacheMiss
    from extract.runner import run_extraction_sync
    from guardrails.runner import run_guardrails

    ids = _doc_ids(None)
    for cfg in load_models():
        for source in ("ocr", "oracle"):
            if not fresh and not run_dir(cfg.slug, source, "extract").exists():
                continue
            try:
                run_extraction_sync(
                    cfg, ids, source, concurrency=1, fresh=fresh, cache_only=not fresh
                )
            except CacheMiss as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1) from exc
            except MissingApiKey as exc:
                typer.echo(f"Skipping {cfg.slug}/{source}: {exc}", err=True)
                continue
            run_guardrails(cfg.slug, source)  # type: ignore[arg-type]
            typer.echo(f"replayed {cfg.slug}/{source}")


@app.command()
def report(check: bool = typer.Option(False, help="Fail if results.json is stale")) -> None:
    """Write results/results.json, the chart and the README results block."""
    from bench.report import write_report

    ok = write_report(check=check)
    if not ok:
        typer.echo(
            "results/results.json does not match the run records. Run `make report` and commit.",
            err=True,
        )
        raise typer.Exit(1)


@app.command("demo-poison")
def demo_poison(model: str = typer.Option(...), source: str = typer.Option("ocr")) -> None:
    """Show one poisoned invoice: the injection, what the model extracted, and the verdict."""
    from bench.inputs import list_doc_ids, load_gold
    from bench.paths import run_dir
    from bench.records import ExtractionRecord, GuardrailRecord, read_model

    for doc_id in list_doc_ids("poisoned"):
        gold = load_gold(doc_id)
        ext = run_dir(model, source, "extract") / f"{doc_id}.json"
        grd = run_dir(model, source, "guard") / f"{doc_id}.json"
        if gold.doc_type != "tax_invoice" or not (ext.exists() and grd.exists()):
            continue
        record = read_model(ext, ExtractionRecord)
        verdict = read_model(grd, GuardrailRecord)
        inj = gold.injection
        typer.echo(f"Document      {doc_id} ({gold.layout})")
        typer.echo(f"Injected text {inj.text!r} ({inj.style})" if inj else "No injection")
        if inj:
            field = inj.target_field
            got = (record.final or {}).get(field)
            typer.echo(f"Target field  {field}: true {gold.fields.get(field)!r}, extracted {got!r}")
        typer.echo(f"Route         {verdict.route.upper()}")
        for reason in verdict.reasons:
            typer.echo(f"  reason: {reason}")
        typer.echo("")
    raise typer.Exit(0)


def main() -> None:
    try:
        app()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

"""Run one model over a list of documents: attempt, one schema retry, record."""

from __future__ import annotations

import asyncio
import time

from bench.config import ModelConfig
from bench.inputs import load_gold, load_input_text
from bench.paths import run_dir
from bench.records import Attempt, ExtractionRecord, TextSource, write_json
from extract.client import ApiError, CacheMiss, CallResult, Pacer, get_completion
from extract.parse import extract_json, validate
from extract.prompts import PROMPT_VERSION, build_messages


def _attempt(result: CallResult, parsed: dict | None, error: str | None, conc: int) -> Attempt:
    return Attempt(
        raw=result.raw,
        parsed=parsed,
        schema_valid=parsed is not None,
        error=error,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        usage_estimated=result.usage_estimated,
        ttft_s=result.ttft_s,
        itl_s=result.itl_s,
        total_s=result.total_s,
        started_at=result.started_at,
        concurrency=conc,
        cached=result.cached,
        reasoning_tokens=result.reasoning_tokens,
        prompt_cached_tokens=result.prompt_cached_tokens,
        prompt_evaluated_tokens=result.prompt_evaluated_tokens,
    )


def _check(doc_type: str, raw: str) -> tuple[dict | None, str | None]:
    obj = extract_json(raw)
    if obj is None:
        return None, "no JSON object found in the reply"
    return validate(doc_type, obj)


def _retry_messages(messages: list[dict], raw: str, error: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": raw or "(empty reply)"},
        {
            "role": "user",
            "content": f"Your reply did not match the schema: {error}. "
            "Reply with the corrected JSON object only.",
        },
    ]


async def _extract_one(
    model: ModelConfig,
    doc_id: str,
    text_source: TextSource,
    conc: int,
    sem: asyncio.Semaphore,
    pacer: Pacer,
    use_cache: bool,
    fresh: bool,
    sample: int = 0,
    cache_only: bool = False,
) -> ExtractionRecord:
    doc_type = load_gold(doc_id).doc_type
    messages = build_messages(doc_type, load_input_text(doc_id, text_source))
    attempts: list[Attempt] = []
    final: dict | None = None
    api_error: str | None = None
    t0 = time.perf_counter()
    async with sem:
        try:
            for _ in range(2):
                try:
                    result = await get_completion(
                        model, messages, use_cache, fresh, pacer, sample, cache_only
                    )
                except CacheMiss as miss:
                    raise CacheMiss(doc_id, miss.key, model.slug) from None
                parsed, error = _check(doc_type, result.raw)
                attempts.append(_attempt(result, parsed, error, conc))
                if parsed is not None:
                    final = parsed
                    break
                messages = _retry_messages(messages, result.raw, error or "")
        except ApiError as err:
            api_error = f"{err.code}: {err.message}"
    record = ExtractionRecord(
        doc_id=doc_id,
        model_slug=model.slug,
        text_source=text_source,
        prompt_version=PROMPT_VERSION,
        attempts=attempts,
        final=final,
        schema_valid_first_try=bool(attempts) and attempts[0].schema_valid,
        api_error=api_error,
    )
    status = f"api error ({api_error.split(':')[0]})" if api_error else (
        "valid" if final is not None else "invalid"
    )
    print(f"{model.slug} {doc_id}: {status}, {time.perf_counter() - t0:.1f}s", flush=True)
    return record


async def run_extraction(
    model: ModelConfig,
    doc_ids: list[str],
    text_source: TextSource,
    concurrency: int = 1,
    use_cache: bool = True,
    fresh: bool = False,
    write: bool = True,
    cache_only: bool = False,
    sample: int = 0,
) -> list[ExtractionRecord]:
    """Extract every document.

    cache_only=True replays stored responses and never calls the server; a miss raises
    CacheMiss naming the document, and no record is written for the run (records are written
    after every document has been replayed). sample=k > 0 re-draws into its own cache entries
    and run directory `extract-s{k}`. Samples are meant for the API model: a local model at
    temperature 0 is expected to repeat itself exactly.
    """
    conc = max(1, min(concurrency, model.max_concurrency))
    sem = asyncio.Semaphore(conc)
    pacer = Pacer(model.min_interval_s)
    out = run_dir(model.slug, text_source, f"extract-s{sample}" if sample > 0 else "extract")

    async def one(doc_id: str) -> ExtractionRecord:
        rec = await _extract_one(
            model, doc_id, text_source, conc, sem, pacer, use_cache, fresh, sample, cache_only
        )
        # Written as soon as the document finishes, so a crash or an interrupted free-tier run
        # keeps every record it already paid for. A cache-only run writes after all hits.
        if write and not cache_only:
            write_json(out / f"{rec.doc_id}.json", rec)
        return rec

    records = list(await asyncio.gather(*(one(d) for d in doc_ids)))
    if write and cache_only:
        for rec in records:
            write_json(out / f"{rec.doc_id}.json", rec)
    return records


def run_extraction_sync(
    model: ModelConfig,
    doc_ids: list[str],
    text_source: TextSource,
    concurrency: int = 1,
    use_cache: bool = True,
    fresh: bool = False,
    write: bool = True,
    cache_only: bool = False,
    sample: int = 0,
) -> list[ExtractionRecord]:
    return asyncio.run(
        run_extraction(
            model, doc_ids, text_source, concurrency, use_cache, fresh, write, cache_only, sample
        )
    )

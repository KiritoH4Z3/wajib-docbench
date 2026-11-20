"""One streaming call through the OpenAI-compatible API, with mapped errors, rate-limit
backoff, request pacing and a replay cache.

Every model goes through `get_completion`. Only a real request needs an API key, so a
committed cache replays without one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import openai

try:  # openai 3.x ships on httpx2; older SDKs use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from bench import paths
from bench.config import ModelConfig

BACKOFF_START_S = 30.0  # first wait after a rate limit; doubles each retry. Tests patch it.

# Tests inject a mock transport here. Returning None means "use the default HTTP client".
_http_client_factory: Callable[[], httpx.AsyncClient | None] = lambda: None  # noqa: E731


@dataclass
class CallResult:
    raw: str
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_estimated: bool
    ttft_s: float | None
    itl_s: list[float]
    total_s: float | None
    started_at: str
    finish_reason: str | None = None
    cached: bool = False


@dataclass
class ApiError(Exception):
    code: str  # auth, rate_limit, timeout, connection, bad_request, server
    message: str  # what happened and what to do, safe to show
    status: int | None = None
    body: str = ""
    retry_after_s: float | None = None
    detail: str = field(default="", repr=False)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class Pacer:
    """Keeps request starts at least `interval_s` apart across all tasks of one model."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval_s <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval_s
        if start > now:
            await asyncio.sleep(start - now)


# ---------------------------------------------------------------- error mapping


def _body_of(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        return response.text
    except (httpx.ResponseNotRead, httpx.StreamError):
        return str(getattr(exc, "body", "") or exc)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def map_error(exc: Exception, model: ModelConfig) -> ApiError:
    """Turn an SDK or transport exception into an ApiError that says what to do next."""
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return ApiError(
            "timeout",
            f"{model.display_name} did not answer within {model.request_timeout_s:.0f} s. "
            "Check that the server is not overloaded, or raise request_timeout_s in "
            "config/models.yaml.",
            body=str(exc),
        )
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        if model.kind == "local":
            msg = (
                f"Could not reach {model.display_name} at {model.base_url}. "
                f"Start it with: make serve MODEL={model.slug}"
            )
        else:
            msg = (
                f"Could not reach {model.base_url}. Check the network connection and "
                "the base_url in config/models.yaml."
            )
        return ApiError("connection", msg, body=str(exc))
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        body = _body_of(exc)
        retry_after = _retry_after(exc)
        if "FREE_TIER_VERIFICATION_REQUIRED" in body:
            return ApiError(
                "auth",
                "The provider wants this account verified before the free tier works. "
                "Complete the verification in the provider console, then rerun.",
                status,
                body,
            )
        if status in (401, 403):
            return ApiError(
                "auth",
                f"The provider rejected the API key (HTTP {status}). Check "
                f"{model.api_key_env or 'the key'} in .env and that the key has access to "
                f"model {model.model}.",
                status,
                body,
            )
        if status in (429, 503) or (status == 400 and "RESOURCE_EXHAUSTED" in body):
            return ApiError(
                "rate_limit",
                "The provider is rate limiting requests. The run waits and retries; if it "
                "keeps failing, lower the concurrency or raise min_interval_s.",
                status,
                body,
                retry_after,
            )
        if status >= 500:
            return ApiError(
                "server",
                f"The server failed with HTTP {status}. Rerun later; cached documents are "
                "not repeated.",
                status,
                body,
            )
        return ApiError(
            "bad_request",
            f"The server refused the request (HTTP {status}). See errors.log for its "
            "reply; the prompt, max_tokens or extra_body is probably not accepted by this model.",
            status,
            body,
        )
    raise exc


def log_error(model: ModelConfig, err: ApiError, key: str | None = None) -> None:
    """Append the raw error body to results/runs/<slug>/errors.log. Never writes the key."""
    body = err.body.replace(key, "***") if key else err.body
    path: Path = paths.RUNS / model.slug / "errors.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {err.code} status={err.status} {body}\n")


# ---------------------------------------------------------------- the request


def sampling_params(model: ModelConfig) -> dict:
    params: dict = {"max_tokens": model.max_tokens}
    if model.send_temperature:
        params["temperature"] = model.temperature
    if model.seed is not None:
        params["seed"] = model.seed
    if model.extra_body:
        params["extra_body"] = model.extra_body
    return params


def estimate_tokens(chars: int) -> int:
    return math.ceil(chars / 4)


async def _stream_once(model: ModelConfig, messages: list[dict]) -> CallResult:
    key = model.api_key()  # raises MissingApiKey here, only on a real request
    client = openai.AsyncOpenAI(
        base_url=model.base_url,
        api_key=key,
        max_retries=0,
        timeout=model.request_timeout_s,
        http_client=_http_client_factory(),
    )
    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    t0 = time.perf_counter()
    parts: list[str] = []
    ttft: float | None = None
    itl: list[float] = []
    last_content_t: float | None = None
    usage = None
    finish_reason = None
    try:
        stream = await client.chat.completions.create(
            model=model.model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **sampling_params(model),
        )
        async for chunk in stream:
            now = time.perf_counter()
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            content = getattr(delta, "content", None) or ""
            reasoning = (
                getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
            )
            if (content or reasoning) and ttft is None:
                ttft = now - t0
                last_content_t = now
            elif content and last_content_t is not None:
                itl.append(now - last_content_t)
                last_content_t = now
            parts.append(content)
        total = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001  (mapped below, anything unknown is re-raised)
        err = map_error(exc, model)
        log_error(model, err, key)
        raise err from exc
    finally:
        await client.close()

    raw = "".join(parts)
    if usage is not None and usage.prompt_tokens is not None:
        prompt = usage.prompt_tokens
        # total - prompt also counts thinking tokens, which are billed as output.
        total_tokens = getattr(usage, "total_tokens", None)
        completion = total_tokens - prompt if total_tokens else usage.completion_tokens
        estimated = False
    else:
        prompt = estimate_tokens(sum(len(m["content"]) for m in messages))
        completion = estimate_tokens(len(raw))
        estimated = True
    return CallResult(
        raw, prompt, completion, estimated, ttft, itl, total, started_at, finish_reason
    )


async def call_model(
    model: ModelConfig, messages: list[dict], pacer: Pacer | None = None
) -> CallResult:
    """One completion, retrying only rate limits (exponential backoff, Retry-After honoured)."""
    for attempt in range(model.max_rate_limit_retries + 1):
        if pacer is not None:
            await pacer.wait()
        try:
            return await _stream_once(model, messages)
        except ApiError as err:
            if err.code != "rate_limit" or attempt == model.max_rate_limit_retries:
                raise
            delay = err.retry_after_s
            if delay is None:
                delay = BACKOFF_START_S * 2**attempt
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------- cache


def cache_key(model: ModelConfig, messages: list[dict]) -> str:
    payload = {
        "slug": model.slug,
        "model": model.model,
        "base_url": model.base_url,
        "messages": messages,
        "sampling": sampling_params(model),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cache_path(model: ModelConfig, key: str) -> Path:
    return paths.CACHE / model.slug / f"{key}.json"


def _from_cache(path: Path) -> CallResult | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return CallResult(
            raw=d["raw"],
            prompt_tokens=d["usage"]["prompt_tokens"],
            completion_tokens=d["usage"]["completion_tokens"],
            usage_estimated=d["usage"]["estimated"],
            ttft_s=d["timings"]["ttft_s"],
            itl_s=d["timings"]["itl_s"],
            total_s=d["timings"]["total_s"],
            started_at=d["started_at"],
            finish_reason=d.get("finish_reason"),
            cached=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None  # unreadable entry: treat as a miss and overwrite it


def _to_cache(path: Path, r: CallResult) -> None:
    doc = {
        "raw": r.raw,
        "usage": {
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "estimated": r.usage_estimated,
        },
        "timings": {"ttft_s": r.ttft_s, "itl_s": r.itl_s, "total_s": r.total_s},
        "started_at": r.started_at,
        "finish_reason": r.finish_reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


async def get_completion(
    model: ModelConfig,
    messages: list[dict],
    use_cache: bool = True,
    fresh: bool = False,
    pacer: Pacer | None = None,
) -> CallResult:
    """Cached replay when possible, else a real call that is stored. fresh=True skips the
    read but still writes. use_cache=False neither reads nor writes."""
    path = _cache_path(model, cache_key(model, messages))
    if use_cache and not fresh:
        hit = _from_cache(path)
        if hit is not None:
            return hit
    result = await call_model(model, messages, pacer)
    if use_cache:
        _to_cache(path, result)
    return result

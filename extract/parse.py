"""Turn a model's raw reply into a validated, JSON-safe dict."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import ValidationError

from bench.normalize import format_amount
from extract.schemas import SCHEMAS

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[A-Za-z0-9_-]*")
MAX_ERROR_CHARS = 600


def _strip(raw: str) -> str:
    text = _THINK.sub("", raw)
    if "</think>" in text.lower():  # an opening tag the server already removed
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return _FENCE.sub("", text)


def _balanced_end(text: str, start: int) -> int | None:
    """Index just past the object that opens at `start`, ignoring braces inside strings."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_json(raw: str) -> dict | None:
    """The first balanced JSON object in the reply, or None. Floats stay exact (Decimal)."""
    text = _strip(raw)
    pos = text.find("{")
    while pos != -1:
        end = _balanced_end(text, pos)
        if end is not None:
            try:
                obj = json.loads(text[pos:end], parse_float=Decimal)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                return obj
        pos = text.find("{", pos + 1)
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format_amount(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _summarise(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        field = ".".join(str(p) for p in e["loc"]) or "(object)"
        parts.append(f"{field}: {e['msg']}")
    return "; ".join(parts)[:MAX_ERROR_CHARS]


def validate(doc_type: str, obj: dict) -> tuple[dict | None, str | None]:
    """(parsed, None) when obj fits the schema, else (None, short error summary)."""
    try:
        model = SCHEMAS[doc_type].model_validate(obj)
    except ValidationError as err:
        return None, _summarise(err)
    return _json_safe(model.model_dump()), None

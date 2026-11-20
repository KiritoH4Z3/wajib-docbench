"""Model and pricing configuration, read from config/*.yaml.

Every model goes through the same OpenAI-compatible client; a ModelConfig only says where
it lives and how to call it. Nothing here is a secret: API keys are read from the
environment variable a model names, and a missing key stops that model with a message
saying which variable to set (fail closed, never a fallback key).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bench.paths import MODELS_YAML, PRICING_YAML


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    display_name: str
    status: Literal["active", "pending", "v2", "excluded"] = "active"
    status_note: str = ""
    kind: Literal["local", "api"]
    served_by: Literal["llamacpp", "vllm", "api"]
    vendor: str
    hf_repo: str | None = None
    hf_file: str | None = None
    hf_revision: str | None = None
    quant: str | None = None
    licence: str
    licence_url: str
    base_url: str
    model: str  # the name sent in the request
    api_key_env: str | None = None  # None for local servers
    send_temperature: bool = True
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 7
    extra_body: dict[str, Any] = Field(default_factory=dict)
    request_timeout_s: float = 300.0
    max_concurrency: int = 16
    min_interval_s: float = 0.0  # minimum gap between request starts (free-tier pacing)
    max_rate_limit_retries: int = 5
    serve: dict[str, Any] = Field(default_factory=dict)  # how `make serve` starts it

    def api_key(self) -> str:
        """The key for this model. Local servers need none and get a placeholder."""
        if self.api_key_env is None:
            return "not-needed"
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise MissingApiKey(
                f"{self.display_name} needs an API key. Set {self.api_key_env} in .env "
                f"(see .env.example), or run with cached responses only."
            )
        return value


class MissingApiKey(RuntimeError):
    pass


def load_models(path: Path = MODELS_YAML, include_inactive: bool = False) -> list[ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = [ModelConfig.model_validate(m) for m in raw["models"]]
    slugs = [m.slug for m in models]
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"duplicate model slug in {path}")
    return models if include_inactive else [m for m in models if m.status == "active"]


def get_model(slug: str, path: Path = MODELS_YAML) -> ModelConfig:
    for m in load_models(path, include_inactive=True):
        if m.slug == slug:
            return m
    raise KeyError(f"no model with slug {slug!r} in {path}")


def load_pricing(path: Path = PRICING_YAML) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

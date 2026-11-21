"""Fetch model weights and run the local model server (llama.cpp in Docker).

One container at a time, always named `docbench-llm`, so `make stop` can never touch anything
else on the machine. The image is pinned; its digest, the weights' sha256 and the GPU memory
in use after loading are written to results/perf/<slug>/server.json as evidence of the setup
every latency number came from.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

try:  # the installed openai SDK ships httpx2; plain httpx is not a dependency
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from bench.config import ModelConfig
from bench.paths import PERF, ROOT

MODELS_DIR = ROOT / "models"
LLAMACPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda12-b12464"
CONTAINER = "docbench-llm"


def weights_path(model: ModelConfig) -> Path:
    return MODELS_DIR / f"{model.slug}.gguf"


def fetch_weights(model: ModelConfig) -> Path:
    """Download the model's GGUF file from Hugging Face, resuming a partial download, and
    check its size against the Hub's record."""
    if not (model.hf_repo and model.hf_file):
        raise ValueError(f"{model.slug} has no hf_repo/hf_file in config/models.yaml")
    MODELS_DIR.mkdir(exist_ok=True)
    dest = weights_path(model)
    revision = model.hf_revision or "main"
    url = f"https://huggingface.co/{model.hf_repo}/resolve/{revision}/{model.hf_file}"
    headers = {}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    partial = dest.with_suffix(".part")
    if partial.exists():
        headers["Range"] = f"bytes={partial.stat().st_size}-"
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise PermissionError(
                f"Hugging Face refused {model.hf_repo} ({r.status_code}). This repo is gated: "
                "accept its terms on huggingface.co with your account, create a read token and "
                "put it in .env as HF_TOKEN."
            )
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with partial.open(mode) as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    partial.rename(dest)
    verify_weights(model, dest)
    return dest


class WeightsMismatch(RuntimeError):
    pass


def verify_weights(model: ModelConfig, path: Path) -> str:
    """Hash the file and compare it with `serve.weights_sha256` from config/models.yaml when
    one is declared. Returns the digest; raises WeightsMismatch on a difference."""
    value = write_sha256(path)
    expected = model.serve.get("weights_sha256")
    if expected and value != str(expected).lower():
        raise WeightsMismatch(
            f"{path.name} has sha256 {value}, but config/models.yaml pins {expected} for "
            f"{model.slug}. Delete the file and rerun `make models MODEL={model.slug}`, or "
            "update weights_sha256 if you chose a new revision on purpose."
        )
    return value


def write_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    path.with_suffix(".sha256").write_text(value + "\n")
    return value


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def stop() -> None:
    _run(["docker", "rm", "-f", CONTAINER], check=False)


def gpu_memory_used_mib() -> int | None:
    out = _run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], check=False
    )
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def serve(
    model: ModelConfig,
    parallel: int = 4,
    extra_args: list[str] | None = None,
    shared_ctx: int | None = None,
) -> dict:
    """Start llama-server for `model` with `parallel` slots and wait until it is healthy.

    By default every slot gets its own ctx_per_slot tokens of KV cache. With `shared_ctx`
    the slots draw from one pool of that many tokens (llama.cpp's unified KV cache), each
    still capped at ctx_per_slot. That is how 16 slots fit on an 8 GB laptop GPU: 16 private
    4096-token caches need 2.8 GB for Falcon-H1, but 16 real requests use far less."""
    if model.served_by != "llamacpp":
        raise ValueError(f"{model.slug} is not served by llama.cpp")
    weights = weights_path(model)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} is missing. Run: make models MODEL={model.slug}")
    port = int(model.serve.get("port", 8080))
    ctx_per_slot = int(model.serve.get("ctx_per_slot", 4096))
    weights_digest = verify_weights(model, weights)
    stop()
    before = gpu_memory_used_mib()
    cmd = [
        "docker", "run", "-d", "--rm", "--name", CONTAINER, "--gpus", "all",
        "-p", f"127.0.0.1:{port}:8080",
        "-v", f"{MODELS_DIR}:/models:ro",
        LLAMACPP_IMAGE,
        "-m", f"/models/{weights.name}",
        "--host", "0.0.0.0", "--port", "8080",
        "-np", str(parallel), "-c", str(shared_ctx or ctx_per_slot * parallel),
        *(["-kvu", "--kv-unified-per-slot", str(ctx_per_slot)] if shared_ctx else []),
        "--alias", model.model, "--metrics",
        *[str(a) for a in model.serve.get("args", [])],
        *(extra_args or []),
    ]
    _run(cmd)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if _run(["docker", "inspect", CONTAINER], check=False).returncode != 0:
            raise RuntimeError(
                "The model server exited while loading. Its log is gone with the container; "
                "rerun without -d to see it, or try fewer slots (PARALLEL=1) if it ran out of "
                "GPU memory."
            )
        time.sleep(2)
    else:
        logs = _run(["docker", "logs", "--tail", "40", CONTAINER], check=False)
        stop()
        raise TimeoutError(f"The model server did not become healthy in 10 minutes:\n{logs.stderr}")
    logs = _run(["docker", "logs", CONTAINER], check=False)
    info = {
        "model": model.slug,
        "image": LLAMACPP_IMAGE,
        "image_digest": _run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", LLAMACPP_IMAGE],
            check=False,
        ).stdout.strip(),
        "weights_file": model.hf_file,
        "weights_sha256": weights_digest,
        "parallel_slots": parallel,
        "ctx_per_slot": ctx_per_slot,
        "shared_ctx": shared_ctx,
        "args": cmd[cmd.index(LLAMACPP_IMAGE) + 1 :],
        "gpu_mib_before": before,
        "gpu_mib_after_load": gpu_memory_used_mib(),
        "server_log_tail": [
            line for line in (logs.stdout + logs.stderr).splitlines()
            if any(k in line for k in ("n_ctx", "n_slots", "build", "CUDA", "offloaded", "KV"))
        ][-20:],
    }
    out = PERF / model.slug / f"server-np{parallel}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info

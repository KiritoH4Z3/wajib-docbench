"""Cost per 1,000 documents, from measured tokens (API) or measured throughput (self-hosted)."""

from __future__ import annotations

from bench.config import ModelConfig


def _none(reason: str) -> dict:
    return {"per_1000_docs_usd": None, "basis": reason, "sensitivity": []}


def _best_docs_per_hour(perf: dict) -> tuple[float, str] | None:
    """The highest measured docs per minute across concurrency levels, as docs per hour."""
    raw = perf.get("throughput_docs_per_min") or perf.get("throughput") or {}
    raw = raw.get("levels", raw)  # accept the throughput.json shape as well as the flat map
    best: tuple[float, str] | None = None
    for level, entry in raw.items():
        value = entry.get("docs_per_min") if isinstance(entry, dict) else entry
        if isinstance(value, int | float) and value and (best is None or value > best[0]):
            best = (float(value), str(level))
    if best is None:
        return None
    return best[0] * 60.0, best[1]


def cost_per_1000(model: ModelConfig, perf: dict, pricing: dict) -> dict:
    """API: (mean prompt x input price + mean completion x output price) / 1e6 x 1000, where
    completion tokens include thinking tokens. Self-hosted: GPU price per hour divided by
    documents per hour at the best measured concurrency, times 1000. Returns None with a
    reason when an input is missing."""
    if model.kind == "api":
        price = (pricing.get("api") or {}).get(model.slug)
        if not price:
            return _none(f"No price for {model.slug} in config/pricing.yaml. Add one.")
        p_tok, c_tok = perf.get("mean_prompt_tokens"), perf.get("mean_completion_tokens")
        if p_tok is None or c_tok is None:
            return _none("No token counts measured yet. Run the API baseline first.")
        cost = (p_tok * price["input_usd_per_mtok"] + c_tok * price["output_usd_per_mtok"])
        cost = cost / 1e6 * 1000
        basis = (
            f"{p_tok:.0f} input and {c_tok:.0f} output tokens per document (output includes "
            f"thinking) at ${price['input_usd_per_mtok']}/${price['output_usd_per_mtok']} "
            f"per million tokens, {price['tier']}, {price['source']}, fetched {price['fetched']}."
        )
        return {"per_1000_docs_usd": cost, "basis": basis, "sensitivity": []}

    gpu = pricing.get("gpu") or {}
    primary = gpu.get("primary")
    if not primary:
        return _none("No gpu.primary price in config/pricing.yaml. Add one.")
    best = _best_docs_per_hour(perf)
    if best is None:
        return _none("No throughput measured yet. Run the throughput stage first.")
    docs_per_hour, level = best

    def per_1000(price_per_hour: float) -> float:
        return price_per_hour / docs_per_hour * 1000

    basis = (
        f"{primary['name']} at ${primary['usd_per_hour']}/hour ({primary['source']}, fetched "
        f"{primary['fetched']}), divided by {docs_per_hour:.0f} documents per hour measured "
        f"at concurrency {level} on the local GPU."
    )
    sensitivity = [
        {
            "name": s["name"],
            "usd_per_hour": s["usd_per_hour"],
            "per_1000_docs_usd": per_1000(s["usd_per_hour"]),
            "source": s["source"],
            "fetched": s["fetched"],
        }
        for s in gpu.get("sensitivity", [])
    ]
    return {
        "per_1000_docs_usd": per_1000(primary["usd_per_hour"]),
        "basis": basis,
        "sensitivity": sensitivity,
    }

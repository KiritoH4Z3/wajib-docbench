"""The results chart: field F1 on Arabic and bilingual documents against cost per 1,000
documents."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Colour plus marker shape per kind, so the chart reads in greyscale. Colours pass 4.5:1 on white.
STYLE = {
    "local": {"color": "#0B5C73", "marker": "o", "name": "Self-hosted"},
    "api": {"color": "#9A3412", "marker": "s", "name": "API"},
}
INK = "#1F2937"
GRID = "#D1D5DB"
TARGET_F1 = 0.85  # the scope's Arabic target (eval/targets.py)


def accuracy_vs_cost_chart(points: list[dict], out_dir: Path) -> list[Path]:
    """Scatter of Arabic field F1 (y, 0 to 1) against cost per 1,000 documents in USD (x).
    The x axis is logarithmic when the cost range exceeds 20x. Each point is labelled
    directly. Writes accuracy_vs_cost.svg and .png into `out_dir` with no generator metadata.
    Points with a missing value are left out and named in the footnote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plotted = [
        p for p in points if p.get("cost_usd") is not None and p.get("arabic_f1") is not None
    ]
    skipped = [p["label"] for p in points if p not in plotted]

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    costs = [p["cost_usd"] for p in plotted]
    log_x = bool(costs) and min(costs) > 0 and max(costs) / min(costs) > 20
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    xlo, xhi = (min(costs), max(costs)) if costs else (0.0, 1.0)
    for p in plotted:
        style = STYLE.get(p["kind"], STYLE["local"])
        ax.scatter(p["cost_usd"], p["arabic_f1"], s=90, color=style["color"],
                   marker=style["marker"], zorder=3)
        right_half = (p["cost_usd"] > (xlo * xhi) ** 0.5) if log_x else (
            p["cost_usd"] > (xlo + xhi) / 2)
        ax.annotate(p["label"], (p["cost_usd"], p["arabic_f1"]),
                    xytext=(-8, 8) if right_half else (8, 8), textcoords="offset points",
                    ha="right" if right_half else "left", fontsize=10, color=style["color"],
                    fontweight="bold")
    ax.set_ylim(0, 1)
    if costs and log_x:
        pad = (xhi / xlo) ** 0.15
        ax.set_xlim(xlo / pad, xhi * pad)
    elif costs:
        # From zero, so $0.35 against $0.61 is not drawn as a wide spread.
        ax.set_xlim(0.0, xhi * 1.25)
    ax.axhline(TARGET_F1, color=INK, linestyle="--", linewidth=0.9, zorder=2)
    ax.annotate(f"Target {TARGET_F1:.2f}", (0.01, TARGET_F1), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Cost per 1,000 documents (USD" + (", log scale)" if log_x else ")"),
                  color=INK, fontsize=10)
    ax.set_ylabel("Field F1, Arabic and bilingual documents", color=INK, fontsize=10)
    ax.set_title("Arabic and bilingual F1 against cost per 1,000 docs", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    kinds = sorted({p["kind"] for p in plotted})
    note = "Marker: " + ", ".join(
        f"{STYLE.get(k, STYLE['local'])['name']} ({'circle' if k != 'api' else 'square'})"
        for k in kinds)
    if skipped:
        note += ". Not plotted, no measurement: " + ", ".join(skipped)
    note += ". Self-hosted: laptop throughput priced at a cloud T4, always busy."
    fig.text(0.01, 0.01, note, fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    written = []
    for ext, meta in (("svg", {"Creator": None, "Date": None}),
                      ("png", {"Software": None})):
        path = out_dir / f"accuracy_vs_cost.{ext}"
        fig.savefig(path, format=ext, metadata=meta, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written

"""Turn a SearchBench summary.json into PNG charts for the docs.

    uv sync --extra plot
    uv run python bench/plot.py bench/results/<run>/summary.json [--out docs/img]

Writes bench-accuracy.png (accuracy by model x mode) and bench-latency.png (p50 latency).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Categorical slots 1-4 of a CVD-validated palette, assigned to modes in a fixed order
# so a mode keeps its colour across every chart, whatever subset of modes was run.
MODE_COLORS = {"closed": "#2a78d6", "rag": "#eb6834", "agent": "#1baf7a", "multi": "#8b5cc8"}
MODE_LABELS = {
    "closed": "closed-book",
    "rag": "RAG",
    "agent": "agent (tools)",
    "multi": "multi-agent",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _grouped_bars(
    summary: dict[str, Any], metric: str, ylabel: str, title: str, out: Path, percent: bool
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups"]
    run_order = summary.get("meta", {}).get("models") or []
    seen = list(dict.fromkeys(g["model"] for g in groups))
    models = [m for m in run_order if m in seen] + [m for m in seen if m not in run_order]
    modes = [m for m in MODE_COLORS if any(g["mode"] == m for g in groups)]
    value = {(g["model"], g["mode"]): g.get(metric) for g in groups}

    width = min(0.8 / max(len(modes), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(modes) * len(models) + 2.6), 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, mode in enumerate(modes):
        xs, ys = [], []
        for j, model in enumerate(models):
            v = value.get((model, mode))
            if v is None:
                continue
            xs.append(j + (i - (len(modes) - 1) / 2) * width)
            ys.append(100 * v if percent else v)
        bars = ax.bar(
            xs,
            ys,
            width=width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{y:.0f}%" if percent else f"{y:.1f}s" for y in ys],
            padding=2,
            fontsize=8,
            color=INK_2,
        )

    ax.set_xticks(range(len(models)), models, color=INK)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=12, pad=12)
    if percent:
        ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(
        frameon=False,
        ncols=len(modes),
        loc="upper left",
        bbox_to_anchor=(0, -0.08),
        labelcolor=INK_2,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", type=Path, help="path to summary.json")
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    track = summary.get("meta", {}).get("track", "offline")
    _grouped_bars(
        summary,
        "accuracy",
        "% of questions answered correctly",
        f"SearchBench ({track}): accuracy",
        args.out / "bench-accuracy.png",
        percent=True,
    )
    _grouped_bars(
        summary,
        "latency_p50_s",
        "median seconds per question",
        f"SearchBench ({track}): median latency",
        args.out / "bench-latency.png",
        percent=False,
    )


if __name__ == "__main__":
    main()

"""Cross-suite accuracy heatmap: every model x mode against every benchmark suite.

    uv sync --extra plot
    uv run python bench/plot_suites.py [--results bench/results] [--out docs/img]

Reads each suite's summary.json and writes suites-heatmap.png. Cells that were not run
(e.g. closed-book on RGB, whose questions only make sense with the passages) stay blank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# (results folder, column label). rgb-negative's accuracy is its rejection rate.
SUITES = [
    ("offline-full", "SearchBench\n(own, 42)"),
    ("hotpotqa", "HotpotQA\n(60)"),
    ("kazqad", "KazQAD, kk\n(60)"),
    ("rgb-noise", "RGB noise\n(120)"),
    ("rgb-negative", "RGB negative\nrejection (40)"),
    ("simpleqa", "SimpleQA\nlive web (60)"),
]
MODELS = ["qwen2.5:1.5b", "qwen3.5:2b", "qwen2.5:3b", "llama3.2:3b"]
MODES = ["closed", "rag", "agent", "multi"]
MODE_LABELS = {"closed": "closed-book", "rag": "RAG", "agent": "agent", "multi": "multi-agent"}
SURFACE = "#ffffff"  # GitHub's light page background, so the figure has no visible box
INK = "#0b0b0b"
INK_2 = "#52514e"


def load(results: Path) -> dict[tuple[str, str, str], float]:
    acc: dict[tuple[str, str, str], float] = {}
    for folder, _ in SUITES:
        summary: dict[str, Any] = json.loads((results / folder / "summary.json").read_text("utf-8"))
        for g in summary["groups"]:
            if g.get("accuracy") is not None:
                acc[(folder, g["model"], g["mode"])] = 100 * g["accuracy"]
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("bench/results"))
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    acc = load(args.results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [(m, mode) for m in MODELS for mode in MODES]
    grid = np.full((len(rows), len(SUITES)), np.nan)
    for i, (model, mode) in enumerate(rows):
        for j, (folder, _) in enumerate(SUITES):
            v = acc.get((folder, model, mode))
            if v is not None:
                grid[i, j] = v

    fig, ax = plt.subplots(figsize=(9.4, 8.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(SURFACE)
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(SUITES)):
            v = grid[i, j]
            text, color = ("–", INK_2) if np.isnan(v) else (f"{v:.0f}", "white" if v >= 60 else INK)
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(range(len(SUITES)), [label for _, label in SUITES], fontsize=8.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(
        range(len(rows)),
        [MODE_LABELS[mode] for _ in MODELS for mode in MODES],
        fontsize=8.5,
        color=INK_2,
    )
    for b in range(1, len(MODELS)):
        ax.axhline(b * len(MODES) - 0.5, color=SURFACE, linewidth=6)
    # Bold model name to the left of each 4-row group.
    for b, model in enumerate(MODELS):
        ax.text(
            -0.15,
            1 - (b * len(MODES) + len(MODES) / 2) / len(rows),
            model,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
        )
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(
        "Accuracy (%) by model, mode and benchmark",
        color=INK,
        loc="left",
        fontsize=12,
        pad=44,
    )
    fig.tight_layout()
    out = args.out / "suites-heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

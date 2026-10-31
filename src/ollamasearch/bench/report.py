"""Render a SearchBench summary as Markdown tables."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .dataset import CATEGORIES


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{100 * float(value):.0f}%"


def _num(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_report(summary: Mapping[str, Any]) -> str:
    meta = summary.get("meta", {})
    groups: list[Mapping[str, Any]] = summary.get("groups", [])
    parts = [f"# SearchBench report: {meta.get('track', '?')} track", ""]
    if meta:
        parts += [
            f"- Models: {', '.join(meta.get('models', []))}",
            f"- Modes: {', '.join(meta.get('modes', []))}",
            f"- Dataset: `{meta.get('dataset')}` "
            f"(sha256 {str(meta.get('dataset_sha256', ''))[:12]})",
            f"- Repeats: {meta.get('repeats')}, limit: {meta.get('limit') or 'none'}",
            f"- ollamasearch {meta.get('ollamasearch_version')}, "
            f"Ollama {meta.get('ollama_server_version') or '?'}, started {meta.get('started_at')}",
            "",
        ]
    if not groups:
        return "\n".join([*parts, "No runs yet.", ""])

    parts += ["## Overall", ""]
    parts.append(
        _table(
            [
                "Model",
                "Mode",
                "N",
                "Accuracy",
                "Answerable",
                "Abstains when absent",
                "Citation hit",
                "Gold in context",
                "p50 s",
                "p95 s",
                "tok/s",
            ],
            [
                [
                    g["model"],
                    g["mode"],
                    str(g["n"]),
                    _pct(g["accuracy"]),
                    _pct(g["answerable_accuracy"]),
                    _pct(g["unanswerable_accuracy"]),
                    _pct(g["citation_hit_rate"]),
                    _pct(g["gold_in_context_rate"]),
                    _num(g["latency_p50_s"]),
                    _num(g["latency_p95_s"]),
                    _num(g["tokens_per_s"], 0),
                ]
                for g in groups
            ],
        )
    )

    cats = [c for c in CATEGORIES if any(c in g["by_category"] for g in groups)]
    parts += ["", "## Accuracy by category", ""]
    parts.append(
        _table(
            ["Model", "Mode", *cats],
            [
                [g["model"], g["mode"], *(_pct(g["by_category"].get(c)) for c in cats)]
                for g in groups
            ],
        )
    )

    langs = sorted({lang for g in groups for lang in g["by_lang"]})
    parts += ["", "## Accuracy by question language", ""]
    parts.append(
        _table(
            ["Model", "Mode", *langs],
            [[g["model"], g["mode"], *(_pct(g["by_lang"].get(x)) for x in langs)] for g in groups],
        )
    )

    agents = [g for g in groups if "agent" in g]
    if agents:
        parts += ["", "## Agent tool use", ""]
        parts.append(
            _table(
                [
                    "Model",
                    "Calls / question",
                    "Invalid calls",
                    "Searched at all",
                    "Fetched a page",
                    "Text-JSON calls",
                    "Hit step limit",
                ],
                [
                    [
                        g["model"],
                        _num(g["agent"]["tool_calls_per_question"]),
                        _pct(g["agent"]["invalid_call_rate"]),
                        _pct(g["agent"]["searched_rate"]),
                        _pct(g["agent"]["fetched_rate"]),
                        _pct(g["agent"]["text_tool_call_rate"]),
                        _pct(g["agent"]["step_limit_rate"]),
                    ]
                    for g in agents
                ],
            )
        )

    parts += ["", "## Other", ""]
    parts.append(
        _table(
            [
                "Model",
                "Mode",
                "Stale answers (conflicting)",
                "False abstentions",
                "Errors",
                "Prompt tok",
                "Gen tok",
            ],
            [
                [
                    g["model"],
                    g["mode"],
                    _pct(g["stale_rate"]),
                    _pct(g["false_abstention_rate"]),
                    _pct(g["error_rate"]),
                    _num(g["mean_prompt_tokens"], 0),
                    _num(g["mean_eval_tokens"], 0),
                ]
                for g in groups
            ],
        )
    )
    parts.append("")
    return "\n".join(parts)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    path.write_text(render_report(summary), encoding="utf-8")

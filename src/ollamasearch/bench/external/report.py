"""Render an external-benchmark summary as Markdown (same style as the SearchBench report)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..report import _num, _pct, _table


def _metric(value: Any) -> str:
    """Task metrics are rates (floats, shown as %) or counts."""
    if isinstance(value, float):
        return _pct(value)
    return "–" if value is None else str(value)


def render_report(
    summary: Mapping[str, Any],
    *,
    metric_labels: Mapping[str, str] | None = None,
    protocol_note: str = "",
) -> str:
    meta = summary.get("meta", {})
    groups: Sequence[Mapping[str, Any]] = summary.get("groups", [])
    title = meta.get("title") or meta.get("suite", "external benchmark")
    parts = [f"# {title}: ollamasearch report", ""]
    if meta:
        parts += [
            f"> **{title}**, {meta.get('citation', '?')}. Source: <{meta.get('source_url')}>, "
            f"license {meta.get('license', '?')}. Downloaded at run time, not redistributed.",
            "",
            f"- Models: {', '.join(meta.get('models', []))}",
            f"- Modes: {', '.join(meta.get('modes', []))}",
            f"- Items: {meta.get('n_items')} (sample {meta.get('sample') or 'default'}, "
            f"seed {meta.get('seed')}), repeats {meta.get('repeats')}",
        ]
        for f in meta.get("dataset_files", []):
            parts.append(
                f"- Dataset file: <{f.get('url')}> ({f.get('bytes')} bytes, "
                f"sha256 {str(f.get('sha256', ''))[:12]})"
            )
        parts += [
            f"- Grader: {meta.get('grader', 'string')}",
            f"- ollamasearch {meta.get('ollamasearch_version')}, "
            f"Ollama {meta.get('ollama_server_version') or '?'}, started {meta.get('started_at')}",
            "",
        ]
    if not groups:
        return "\n".join([*parts, "No runs yet.", ""])

    parts += ["## Overall", ""]
    parts.append(
        _table(
            ["Model", "Mode", "N", "Accuracy", "p50 s", "p95 s", "tok/s", "Errors", "Gen tok"],
            [
                [
                    g["model"],
                    g["mode"],
                    str(g["n"]),
                    _pct(g["accuracy"]),
                    _num(g["latency_p50_s"]),
                    _num(g["latency_p95_s"]),
                    _num(g["tokens_per_s"], 0),
                    _pct(g["error_rate"]),
                    _num(g["mean_eval_tokens"], 0),
                ]
                for g in groups
            ],
        )
    )

    cats = sorted({c for g in groups for c in g["by_category"]})
    if len(cats) > 1:
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

    labels = dict(metric_labels or {})
    for g in groups:
        for key in g.get("metrics", {}):
            labels.setdefault(key, key)
    if labels:
        parts += ["", "## Task metrics", ""]
        parts.append(
            _table(
                ["Model", "Mode", *labels.values()],
                [
                    [g["model"], g["mode"], *(_metric(g.get("metrics", {}).get(k)) for k in labels)]
                    for g in groups
                ],
            )
        )

    agents = [g for g in groups if "agent" in g]
    if agents:
        parts += ["", "## Agent tool use", ""]
        parts.append(
            _table(
                ["Model", "Calls / question", "Invalid calls", "Searched", "Fetched", "Step limit"],
                [
                    [
                        g["model"],
                        _num(g["agent"]["tool_calls_per_question"]),
                        _metric(g["agent"]["invalid_call_rate"]),
                        _metric(g["agent"]["searched_rate"]),
                        _metric(g["agent"]["fetched_rate"]),
                        _metric(g["agent"]["step_limit_rate"]),
                    ]
                    for g in agents
                ],
            )
        )
    multis = [g for g in groups if "multi" in g]
    if multis:
        parts += ["", "## Multi-agent trace", ""]
        parts.append(
            _table(
                ["Model", "LLM calls / q", "Searches / q", "Verifier rejected", "Follow-up round"],
                [
                    [
                        g["model"],
                        _num(g["multi"]["llm_calls_per_question"]),
                        _num(g["multi"]["searches_per_question"]),
                        _metric(g["multi"]["verifier_rejection_rate"]),
                        _metric(g["multi"]["follow_up_rate"]),
                    ]
                    for g in multis
                ],
            )
        )

    if protocol_note:
        parts += ["", "## How this differs from the official protocol", "", protocol_note]
    parts.append("")
    return "\n".join(parts)


def write_report(
    summary: Mapping[str, Any],
    path: Path,
    *,
    metric_labels: Mapping[str, str] | None = None,
    protocol_note: str = "",
) -> None:
    path.write_text(
        render_report(summary, metric_labels=metric_labels, protocol_note=protocol_note),
        encoding="utf-8",
    )

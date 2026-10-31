from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ollamasearch.bench.dataset import DatasetError, load_questions
from ollamasearch.bench.report import render_report
from ollamasearch.bench.runner import BenchConfig, finalize, read_rows, run_bench
from ollamasearch.config import Settings

from .conftest import BENCH, FakeOllama, tool


def smart_reply(kwargs: dict[str, Any]) -> Any:
    """Answer from whatever the prompt contains, like a perfect reader would."""
    text = json.dumps(kwargs["messages"], ensure_ascii=False)
    if kwargs.get("tools") and '"role": "tool"' not in text:
        return {"tool_calls": [tool("web_search", {"query": kwargs["messages"][-1]["content"]})]}
    if "phone number" in text:  # the unanswerable test question
        return "I could not find this information."
    for needle in ("7,420", "1987", "48 метров"):
        if needle in text:
            return f"{needle} [1]"
    return "I could not find this information."


def config(tmp_path: Path, **kw: Any) -> BenchConfig:
    base: dict[str, Any] = {
        "track": "offline",
        "models": ["qwen3.5:2b", "llama3.2:3b"],
        "modes": ["closed", "rag", "agent"],
        "out_dir": tmp_path / "out",
        "bench_dir": BENCH,
        "settings": Settings(cache_dir=None),
    }
    base.update(kw)
    return BenchConfig(**base)


def test_run_bench_end_to_end(tmp_path: Path) -> None:
    fake = FakeOllama(default="")
    fake.script = [smart_reply] * 1000
    progress: list[int] = []
    summary = run_bench(config(tmp_path), fake, progress=lambda i, n, row: progress.append(i))

    out = tmp_path / "out"
    rows = read_rows(out / "runs.jsonl")
    assert len(rows) == 2 * 3 * 4 == len(progress)
    assert {"meta.json", "runs.jsonl", "summary.json", "report.md"} <= {
        p.name for p in out.iterdir()
    }
    groups = {(g["model"], g["mode"]): g for g in summary["groups"]}
    assert groups[("qwen3.5:2b", "closed")]["accuracy"] == 0.25  # only abstains correctly
    rag = groups[("qwen3.5:2b", "rag")]
    assert rag["by_category"]["single-hop"] == 1.0
    assert rag["by_category"]["multilingual"] == 1.0
    assert rag["unanswerable_accuracy"] == 1.0
    assert rag["citation_hit_rate"] is not None
    assert groups[("qwen3.5:2b", "agent")]["agent"]["searched_rate"] == 1.0
    assert fake.unloads == ["qwen3.5:2b", "llama3.2:3b"]  # VRAM freed between models
    # every model call is deterministic
    assert all(c["options"]["temperature"] == 0 and c["options"]["seed"] == 42 for c in fake.calls)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["dataset_sha256"] and "ollama_api_key" not in meta["settings"]


def test_run_bench_resumes(tmp_path: Path) -> None:
    fake = FakeOllama(default="I could not find this information.")
    cfg = config(tmp_path, models=["qwen3.5:2b"], modes=["closed"], limit=2)
    run_bench(cfg, fake)
    first = len(fake.calls)
    run_bench(cfg, fake)
    assert len(fake.calls) == first  # nothing re-run
    assert fake.unloads == ["qwen3.5:2b"]  # second run did not load anything
    run_bench(config(tmp_path, models=["qwen3.5:2b"], modes=["closed"], limit=3), fake)
    assert len(fake.calls) == first + 1
    assert len(read_rows(tmp_path / "out" / "runs.jsonl")) == 3


def test_repeats_and_category_filter(tmp_path: Path) -> None:
    fake = FakeOllama()
    cfg = config(tmp_path, models=["m"], modes=["closed"], repeats=2, categories=["unanswerable"])
    run_bench(cfg, fake)
    rows = read_rows(tmp_path / "out" / "runs.jsonl")
    assert [(r["question_id"], r["repeat"]) for r in rows] == [("t-004", 0), ("t-004", 1)]


def test_bad_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_bench(config(tmp_path, modes=["telepathy"]), FakeOllama())


def test_read_rows_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text('{"key": "a"}\n{truncated\n\n{"key": "b"}\n')
    assert [r["key"] for r in read_rows(path)] == ["a", "b"]
    assert read_rows(tmp_path / "nope.jsonl") == []


def test_finalize_and_report_without_runs(tmp_path: Path) -> None:
    summary = finalize(tmp_path)
    assert summary["groups"] == []
    assert "No runs yet." in (tmp_path / "report.md").read_text()


def test_report_tables(tmp_path: Path) -> None:
    fake = FakeOllama()
    fake.script = [smart_reply] * 100
    summary = run_bench(config(tmp_path, models=["m"]), fake)
    report = render_report(summary)
    for heading in (
        "## Overall",
        "## Accuracy by category",
        "## Agent tool use",
        "## Accuracy by question language",
    ):
        assert heading in report
    assert "| m | rag |" in report


def test_load_questions_validation(tmp_path: Path) -> None:
    assert len(load_questions(BENCH / "datasets" / "offline.jsonl")) == 4
    bad = tmp_path / "bad.jsonl"
    for line, msg in [
        (
            '{"id": "1", "question": "q", "lang": "en", "answers": ["a"], "category": "x"}',
            "category",
        ),
        ('{"id": "1", "question": "q", "lang": "en", "category": "single-hop"}', "without answers"),
        ('{"question": "q"}', "id"),
        ("not json", "Expecting"),
    ]:
        bad.write_text(line + "\n")
        with pytest.raises(DatasetError, match=msg):
            load_questions(bad)
    line = '{"id": "1", "question": "q", "lang": "en", "answers": ["a"], "category": "single-hop"}'
    bad.write_text(line + "\n" + line + "\n")
    with pytest.raises(DatasetError, match="duplicate"):
        load_questions(bad)

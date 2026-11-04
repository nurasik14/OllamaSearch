from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ollamasearch.bench.external.report import render_report
from ollamasearch.bench.external.runner import (
    ExternalConfig,
    finalize_external,
    run_external,
)
from ollamasearch.bench.external.tasks import _common
from ollamasearch.bench.runner import read_rows
from ollamasearch.config import Settings

from .conftest import DATA, FakeOllama, tool

EXTERNAL = DATA / "external"
GOLD = ("January 2, 2022", "Ada Brightwater and Milo Quist", "936,172", "Tallinn")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must not download datasets")

    monkeypatch.setattr(_common, "_new_client", boom)


def reader(kwargs: dict[str, Any]) -> Any:
    """A perfect reader: searches first (agent), then answers from whatever it was shown."""
    text = json.dumps(kwargs["messages"], ensure_ascii=False)
    if kwargs.get("tools") and '"role": "tool"' not in text:
        return {"tool_calls": [tool("web_search", {"query": kwargs["messages"][-1]["content"]})]}
    for needle in GOLD:
        if needle in text:
            return f"{needle} [1]"
    return "I could not find this information."


def config(tmp_path: Path, **kw: Any) -> ExternalConfig:
    base: dict[str, Any] = {
        "task": "rgb-noise",
        "models": ["qwen3.5:2b", "llama3.2:3b"],
        "modes": ["closed", "rag", "agent", "multi"],
        "out_dir": tmp_path / "out",
        "data_dir": EXTERNAL,
        "sample": 0,
        "settings": Settings(cache_dir=None),
    }
    base.update(kw)
    return ExternalConfig(**base)


def test_run_external_end_to_end(tmp_path: Path) -> None:
    fake = FakeOllama()
    fake.script = [reader] * 1000
    progress: list[dict[str, Any]] = []
    summary = run_external(config(tmp_path), fake, progress=lambda i, n, r: progress.append(r))

    out = tmp_path / "out"
    rows = read_rows(out / "runs.jsonl")
    assert len(rows) == 2 * 4 * 12 == len(progress)
    assert {"meta.json", "runs.jsonl", "summary.json", "report.md"} <= {
        p.name for p in out.iterdir()
    }
    row = rows[0]
    for key in (
        "key",
        "suite",
        "model",
        "mode",
        "question_id",
        "repeat",
        "category",
        "lang",
        "question",
        "answers",
        "score",
        "answer",
        "timestamp",
    ):
        assert key in row
    assert row["suite"] == "rgb-noise" and row["score"]["grader"] == "string"
    groups = {(g["model"], g["mode"]): g for g in summary["groups"]}
    assert groups[("qwen3.5:2b", "closed")]["accuracy"] == 0.0
    rag = groups[("qwen3.5:2b", "rag")]
    assert rag["by_category"]["noise-0.0"] == 1.0
    assert rag["metrics"]["gold_in_context"] == 1.0
    assert groups[("qwen3.5:2b", "agent")]["accuracy"] > 0.5
    assert groups[("qwen3.5:2b", "multi")]["n"] == 12
    assert fake.unloads == ["qwen3.5:2b", "llama3.2:3b"]

    meta = json.loads((out / "meta.json").read_text())
    assert meta["suite"] == "rgb-noise" and meta["license"] == "CC BY-NC-SA 4.0"
    assert meta["source_url"] == "https://github.com/chen700564/RGB"
    files = meta["dataset_files"]
    assert (
        files[0]["sha256"] and files[0]["bytes"] > 0 and files[0]["url"].endswith("en_refine.json")
    )
    assert meta["grader"] == "string" and "ollama_api_key" not in meta["settings"]
    assert meta["n_items"] == 12 and meta["seed"] == 42


def test_run_external_resumes(tmp_path: Path) -> None:
    fake = FakeOllama(default="I could not find this information.")
    cfg = config(tmp_path, task="rgb-negative", models=["m"], modes=["closed", "rag"], sample=2)
    summary = run_external(cfg, fake)
    first = len(fake.calls)
    assert first == 4 and summary["groups"][0]["accuracy"] == 1.0  # all rejected
    run_external(cfg, fake)
    assert len(fake.calls) == first  # nothing re-run
    assert fake.unloads == ["m"]  # the second run loaded nothing
    run_external(
        config(tmp_path, task="rgb-negative", models=["m"], modes=["closed", "rag"], sample=3), fake
    )
    assert len(fake.calls) == first + 2  # prefix-stable sample: only the new item ran
    assert len(read_rows(tmp_path / "out" / "runs.jsonl")) == 6


def test_run_external_refuses_other_suite_in_same_dir(tmp_path: Path) -> None:
    run_external(
        config(tmp_path, task="rgb-negative", models=["m"], modes=["closed"], sample=1),
        FakeOllama(),
    )
    with pytest.raises(ValueError, match="rgb-negative"):
        run_external(config(tmp_path, models=["m"], modes=["closed"], sample=1), FakeOllama())


def test_bad_mode_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        run_external(config(tmp_path, modes=["telepathy"]), FakeOllama())


def grader(kwargs: dict[str, Any]) -> str:
    """Judge model: official SimpleQA letters; answer model: Sugeno or a wrong year."""
    prompt = kwargs["messages"][-1]["content"]
    if "Gold target:" in prompt:
        return "A" if "Michio Sugeno" in prompt.split("Predicted answer:")[-1] else "B"
    return "Michio Sugeno received it." if "Rosenblatt" in prompt else "It was 1900."


def test_simpleqa_judge_pass(tmp_path: Path) -> None:
    fake = FakeOllama()
    fake.script = [grader] * 100
    cfg = config(tmp_path, task="simpleqa", models=["m"], modes=["closed"], judge_model="judge")
    summary = run_external(cfg, fake)
    rows = read_rows(tmp_path / "out" / "runs.jsonl")
    assert len(rows) == 4
    assert all(r["score"]["grader"] == "judge:judge" for r in rows)
    assert [r["score"]["grade"] for r in rows] == ["CORRECT", "INCORRECT", "INCORRECT", "INCORRECT"]
    assert rows[0]["score"]["string_grade"] == "CORRECT"
    metrics = summary["groups"][0]["metrics"]
    assert metrics["is_correct"] == 0.25 and metrics["accuracy_given_attempted"] == 0.25
    assert metrics["string_is_correct"] == 0.25
    assert fake.unloads == ["m", "judge"]
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["grader"] == "judge:judge" and meta["license"] == "MIT"
    calls = len(fake.calls)
    run_external(cfg, fake)
    assert len(fake.calls) == calls  # neither answers nor grades are redone
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Correct given attempted" in report and "Grader: judge:judge" in report


def test_judge_ignored_for_tasks_without_grader(tmp_path: Path) -> None:
    fake = FakeOllama()
    cfg = config(tmp_path, task="kazqad", models=["m"], modes=["closed"], judge_model="judge")
    run_external(cfg, fake)
    assert fake.unloads == ["m"]
    assert json.loads((tmp_path / "out" / "meta.json").read_text())["grader"] == "string"


def test_kazqad_rag_end_to_end(tmp_path: Path) -> None:
    def kk_reader(kwargs: dict[str, Any]) -> str:
        text = json.dumps(kwargs["messages"], ensure_ascii=False)
        return "Ұлтабарға ашылады [1]." if "ұлтабарға түседі" in text else "Білмеймін."

    fake = FakeOllama()
    fake.script = [kk_reader] * 100
    summary = run_external(config(tmp_path, task="kazqad", models=["m"], modes=["rag"]), fake)
    rows = read_rows(tmp_path / "out" / "runs.jsonl")
    liver = next(r for r in rows if r["question_id"] == "bio0001")
    assert liver["lang"] == "kk" and liver["score"]["correct"] and liver["score"]["gold_in_context"]
    assert liver["answer"]["context_urls"][0].startswith("http://corpus.local/")
    assert summary["groups"][0]["by_category"]["biology"] == pytest.approx(1 / 3)


def test_report_rendering(tmp_path: Path) -> None:
    fake = FakeOllama()
    fake.script = [reader] * 100
    summary = run_external(config(tmp_path, models=["m"], modes=["rag"]), fake)
    report = (tmp_path / "out" / "report.md").read_text()
    assert report.startswith("# RGB noise robustness (en): ollamasearch report")
    assert "Chen et al., AAAI 2024" in report and "CC BY-NC-SA 4.0" in report
    for heading in (
        "## Overall",
        "## Accuracy by category",
        "## Task metrics",
        "## How this differs from the official protocol",
    ):
        assert heading in report
    assert "| m | rag | 12 |" in report and "Positive passage in context" in report
    assert "noise-0.4" in report
    # report-only rebuild gives the same file
    finalize_external(tmp_path / "out")
    assert (tmp_path / "out" / "report.md").read_text() == report
    assert "No runs yet." in render_report({"meta": summary["meta"], "groups": []})


def test_finalize_without_runs(tmp_path: Path) -> None:
    summary = finalize_external(tmp_path)
    assert summary["groups"] == []
    assert "No runs yet." in (tmp_path / "report.md").read_text()

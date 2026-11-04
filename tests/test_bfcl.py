"""BFCL runner: data loading, schema conversion, text-call parsing, run/resume, report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from ollamasearch import cli
from ollamasearch.bench import bfcl
from ollamasearch.bench.bfcl import (
    BfclConfig,
    BfclItem,
    convert_to_tools,
    finalize_bfcl,
    fit_ollama_schema,
    load_category,
    parse_text_calls,
    run_bfcl,
    sample_items,
)
from ollamasearch.bench.dataset import file_sha256
from ollamasearch.config import Settings
from ollamasearch.llm import ChatResult, ToolCall

from .conftest import DATA, FakeOllama, tool

FIXTURES = DATA / "bfcl"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A dataset cache that already holds the fixture files (so nothing is downloaded)."""
    root = tmp_path / "datasets"
    shutil.copytree(FIXTURES, root / "bfcl")
    return root


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, dest: Path, **_: Any) -> None:
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(bfcl, "_download", fail)


# --------------------------------------------------------------------------- data


def test_load_categories(data_dir: Path) -> None:
    simple = load_category(data_dir, "simple")
    assert [it.id for it in simple] == [f"simple_{i}" for i in range(5)]
    assert simple[0].messages == [{"role": "user", "content": simple[0].messages[0]["content"]}]
    assert simple[1].ground_truth == [{"math.factorial": {"number": [5]}}]
    irrelevance = load_category(data_dir, "irrelevance")
    assert len(irrelevance) == 2 and all(it.ground_truth is None for it in irrelevance)
    with pytest.raises(ValueError, match="unknown BFCL category"):
        load_category(data_dir, "live_simple")


def test_ensure_files_downloads_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_download(url: str, dest: Path, **_: Any) -> None:
        urls.append(url)
        rel = url.split("/resolve/main/", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURES / rel, dest)

    monkeypatch.setattr(bfcl, "_download", fake_download)
    files = bfcl.ensure_files(tmp_path, ["parallel", "irrelevance"])
    assert set(files) == {
        "BFCL_v3_parallel.json",
        "possible_answer/BFCL_v3_parallel.json",
        "BFCL_v3_irrelevance.json",
    }
    assert urls[0] == f"{bfcl.HF_BASE}/BFCL_v3_parallel.json"
    path = tmp_path / "bfcl" / "BFCL_v3_parallel.json"
    assert files["BFCL_v3_parallel.json"]["sha256"] == file_sha256(path)
    bfcl.ensure_files(tmp_path, ["parallel", "irrelevance"])  # cached: no second download
    assert len(urls) == 3


def _items(n: int, category: str = "simple") -> list[BfclItem]:
    return [BfclItem(f"{category}_{i}", category, [], [], []) for i in range(n)]


def test_sampling_is_seeded_and_ordered() -> None:
    items = _items(120)
    a = [it.id for it in sample_items(items, 10, seed=1)]
    assert a == [it.id for it in sample_items(items, 10, seed=1)]
    assert a != [it.id for it in sample_items(items, 10, seed=2)]
    assert a == sorted(a, key=lambda s: int(s.split("_")[1]))  # file order kept
    assert len(sample_items(items, None, seed=1)) == bfcl.DEFAULT_SAMPLE
    assert len(sample_items(items, 0, seed=1)) == 120
    assert len(sample_items(_items(5), 50, seed=1)) == 5


# --------------------------------------------------------------------------- tools

NESTED_DOC: dict[str, Any] = {
    "name": "geo.route",
    "description": "Plan a route.",
    "parameters": {
        "type": "dict",
        "properties": {
            "weight": {"type": "float", "description": "Cargo weight."},
            "stops": {"type": "tuple", "items": {"type": "string"}, "description": "Stops."},
            "matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "float"}},
                "description": "Costs.",
            },
            "legs": {
                "type": "array",
                "items": {"type": "dict", "properties": {"km": {"type": "float"}}},
                "description": "Legs.",
            },
            "options": {
                "type": "dict",
                "description": "Options.",
                "properties": {"avoid": {"type": "any", "description": "What to avoid."}},
                "required": ["avoid"],
            },
            "anything": {"type": "any", "description": "Free value.", "default": "x"},
            "untyped": {"description": "No type given."},
        },
        "required": ["weight"],
    },
}


def test_convert_to_tools_mirrors_bfcl() -> None:
    (tool_,) = convert_to_tools([NESTED_DOC])
    fn = tool_["function"]
    assert tool_["type"] == "function"
    assert fn["name"] == "geo_route"
    assert fn["description"].endswith("Note that the provided function is in Python 3 syntax.")
    params = fn["parameters"]
    assert params["type"] == "object" and params["required"] == ["weight"]
    props = params["properties"]
    assert props["weight"] == {
        "type": "number",
        "format": "float",
        "description": "Cargo weight. This is a float type value.",
    }
    assert props["stops"]["type"] == "array" and props["stops"]["items"] == {"type": "string"}
    assert props["matrix"]["items"] == {"type": "array", "items": {"type": "number"}}
    assert props["legs"]["items"]["type"] == "object"
    assert props["legs"]["items"]["properties"]["km"]["type"] == "number"
    assert props["options"]["type"] == "object"
    assert props["options"]["properties"]["avoid"]["type"] == "string"
    assert props["anything"]["type"] == "string"
    assert props["untyped"]["type"] == "string"
    assert NESTED_DOC["parameters"]["type"] == "dict"  # input not mutated


def test_fit_ollama_schema_keeps_dropped_info_in_descriptions() -> None:
    from ollama._types import Tool

    tools = fit_ollama_schema(convert_to_tools([NESTED_DOC]))
    props = tools[0]["function"]["parameters"]["properties"]
    assert "format" not in props["weight"]
    assert props["anything"]["description"] == "Free value. Default is: x."
    assert "properties" not in props["options"] and "required" not in props["options"]
    assert '"avoid"' in props["options"]["description"]
    assert "Required: ['avoid']." in props["options"]["description"]
    # what the Ollama client sends is now exactly what we built (nothing silently lost)
    dumped = Tool.model_validate(tools[0]).model_dump(exclude_none=True)
    assert dumped == tools[0]


# --------------------------------------------------------------------------- text calls


@pytest.mark.parametrize(
    ("content", "source", "calls"),
    [
        ('{"name": "f", "arguments": {"a": 1}}', "text_json", [("f", {"a": 1})]),
        ('```json\n{"name": "f", "parameters": {"a": 1}}\n```', "text_json", [("f", {"a": 1})]),
        (
            '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>\n'
            '<tool_call>{"name": "g", "arguments": "{\\"b\\": 2}"}</tool_call>',
            "text_json",
            [("f", {"a": 1}), ("g", {"b": 2})],
        ),
        (
            'Sure: [{"name": "f", "arguments": {}}, {"function": {"name": "g", "arguments": {}}}]',
            "text_json",
            [("f", {}), ("g", {})],
        ),
        (
            '{"tool_calls": [{"name": "f", "arguments": {"x": [1, 2]}}]}',
            "text_json",
            [("f", {"x": [1, 2]})],
        ),
        (
            "[math.factorial(number=5), sqrt(x=2.5)]",
            "text_python",
            [("math.factorial", {"number": 5}), ("sqrt", {"x": 2.5})],
        ),
        ("The answer is 42.", "", []),
        ('{"answer": 42}', "", []),
        ("f(x)", "", []),  # positional / non-literal arguments are not a decodable call
        ("", "", []),
    ],
)
def test_parse_text_calls(content: str, source: str, calls: list[tuple[str, Any]]) -> None:
    parsed, got_source = parse_text_calls(content)
    assert got_source == source
    assert [(c.name, c.arguments) for c in parsed] == calls


def test_extract_calls_prefers_native() -> None:
    native = ChatResult(content='{"name": "x", "arguments": {}}', tool_calls=[ToolCall("y", {})])
    assert bfcl.extract_calls(native) == ([ToolCall("y", {})], "native")
    assert bfcl.extract_calls(ChatResult(content="no call")) == ([], "none")


# --------------------------------------------------------------------------- runner

REPLIES: dict[str, Any] = {
    # simple
    "triangle": {"tool_calls": [tool("calculate_triangle_area", {"base": 10, "height": 5})]},
    "factorial of 5": {"tool_calls": [tool("math_factorial", {"number": 5})]},
    "compound interest": (
        '```json\n{"name": "finance_compound_interest", '
        '"arguments": {"principal": 1000, "rate": 0.05, "years": 3}}\n```'
    ),
    "Sort the numbers": "The sorted list is 1, 2, 3.",
    "Book a table": {
        "tool_calls": [
            tool("book_table", {"city": "new york", "preferences": {"seat": "Window", "people": 2}})
        ]
    },
    # multiple
    "weather in Paris": {
        "tool_calls": [tool("weather_get_forecast", {"city": "Paris", "days": 1})]
    },
    "100 US dollars": {
        "tool_calls": [
            tool("currency_convert", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
        ]
    },
    # parallel
    "Taylor Swift": {
        "tool_calls": [
            tool("spotify_play", {"artist": "Maroon 5", "duration": 15}),
            tool("spotify_play", {"artist": "Taylor Swift", "duration": 20}),
        ]
    },
    "square root": "[sqrt(x=16), sqrt(x=81)]",
    # parallel_multiple
    "Tokyo": {
        "tool_calls": [
            tool(
                "currency_convert", {"amount": 50.0, "from_currency": "EUR", "to_currency": "JPY"}
            ),
            tool("weather_get_current", {"city": "Tokyo"}),
        ]
    },
    # irrelevance
    "world cup": "France won the 1998 World Cup.",
    "joke about cats": {"tool_calls": [tool("math_factorial", {"number": 9})]},
}

EXPECTED = {
    "simple_0": (True, "native"),
    "simple_1": (True, "native"),
    "simple_2": (True, "text_json"),
    "simple_3": (False, "none"),
    "simple_4": (True, "native"),
    "multiple_0": (False, "native"),
    "multiple_1": (True, "native"),
    "parallel_0": (True, "native"),
    "parallel_1": (True, "text_python"),
    "parallel_multiple_0": (True, "native"),
    "irrelevance_0": (True, "none"),
    "irrelevance_1": (False, "native"),
}


def responder(kwargs: dict[str, Any]) -> Any:
    text = kwargs["messages"][-1]["content"]
    for needle, reply in REPLIES.items():
        if needle in text:
            return reply
    raise AssertionError(f"no scripted reply for {text!r}")


def scripted() -> FakeOllama:
    return FakeOllama(script=[responder] * 100)


def _cfg(tmp_path: Path, data_dir: Path, **kw: Any) -> BfclConfig:
    return BfclConfig(
        models=kw.pop("models", ["qwen3.5:2b"]),
        out_dir=tmp_path / "out",
        data_dir=data_dir,
        sample=kw.pop("sample", 0),
        settings=Settings(cache_dir=None, temperature=0.7, seed=7),
        **kw,
    )


def _rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]


def test_run_end_to_end(tmp_path: Path, data_dir: Path) -> None:
    fake = scripted()
    seen: list[tuple[int, int, str]] = []
    summary = run_bfcl(
        _cfg(tmp_path, data_dir),
        fake,
        progress=lambda i, total, row: seen.append((i, total, row["id"])),
        server_version="0.18.1",
    )
    out = tmp_path / "out"
    rows = _rows(out)
    assert {r["id"]: (r["score"]["correct"], r["call_source"]) for r in rows} == EXPECTED
    assert seen[-1] == (12, 12, "irrelevance_1")

    row = next(r for r in rows if r["id"] == "simple_1")
    assert row["key"] == "qwen3.5:2b|bfcl|simple|simple_1|0"
    assert (row["suite"], row["mode"], row["category"], row["question_id"]) == (
        "bfcl",
        "bfcl",
        "simple",
        "simple_1",
    )
    assert row["calls"] == [{"name": "math_factorial", "arguments": {"number": 5}}]
    assert row["usage"]["eval_tokens"] == 10 and row["latency_s"] >= 0
    wrong = next(r for r in rows if r["id"] == "multiple_0")
    assert wrong["score"]["error_type"] == "simple_function_checker:wrong_func_name"

    # what Ollama received: native tools with underscored names, temperature 0, settings seed
    call = fake.calls[1]
    assert call["tools"][0]["function"]["name"] == "math_factorial"
    assert call["options"]["temperature"] == 0.0 and call["options"]["seed"] == 7
    assert fake.unloads == ["qwen3.5:2b"]

    meta = json.loads((out / "meta.json").read_text())
    assert meta["suite"] == "bfcl" and meta["categories"] == list(bfcl.BFCL_CATEGORIES)
    assert meta["dataset"]["license"] == "Apache-2.0"
    simple_hash = file_sha256(data_dir / "bfcl" / "BFCL_v3_simple.json")
    assert meta["dataset"]["files"]["BFCL_v3_simple.json"]["sha256"] == simple_hash
    assert meta["item_ids"]["parallel"] == ["parallel_0", "parallel_1"]
    assert meta["ollama_server_version"] == "0.18.1"
    assert not str(meta["dataset"]["data_dir"]).startswith(str(Path.home()))

    groups = {(g["model"], g["category"]): g for g in summary["groups"]}
    assert groups[("qwen3.5:2b", "simple")]["accuracy"] == pytest.approx(0.8)
    assert groups[("qwen3.5:2b", "simple")]["error_types"] == {"ast_decoder:decoder_failed": 1}
    assert groups[("qwen3.5:2b", "simple")]["call_sources"] == {
        "native": 3,
        "text_json": 1,
        "none": 1,
    }
    overall = groups[("qwen3.5:2b", "overall")]
    # unweighted over categories: (0.8 + 0.5 + 1 + 1 + 0.5) / 5, not 9/12 items
    assert overall["accuracy"] == pytest.approx(0.76)
    assert overall["item_accuracy"] == pytest.approx(9 / 12)
    assert overall["ast_accuracy"] == pytest.approx((0.8 + 0.5 + 1 + 1) / 4)
    assert overall["irrelevance_accuracy"] == 0.5
    # text calls fail in native-only scoring: simple 3/5, parallel 1/2
    assert overall["native_accuracy"] == pytest.approx((0.6 + 0.5 + 0.5 + 1 + 0.5) / 5)
    assert groups[("qwen3.5:2b", "parallel")]["text_call_rate"] == 0.5
    assert overall["n"] == 12 and overall["mode"] == "bfcl"
    assert (out / "summary.json").is_file()

    report = (out / "report.md").read_text()
    assert "Berkeley Function Calling Leaderboard" in report and "Apache-2.0" in report
    assert "Patil" in report and "gorilla.cs.berkeley.edu" in report
    assert "| qwen3.5:2b | 76% | 62% | 82% | 80% | 50% | 100% | 100% | 50% |" in report
    assert "`simple_function_checker:wrong_func_name`" in report
    assert "Differences from the official leaderboard" in report


def test_resume_and_second_model(tmp_path: Path, data_dir: Path) -> None:
    fake = scripted()
    run_bfcl(_cfg(tmp_path, data_dir, categories=["simple", "irrelevance"]), fake)
    assert len(fake.calls) == 7
    run_bfcl(_cfg(tmp_path, data_dir, categories=["simple", "irrelevance"]), fake)
    assert len(fake.calls) == 7  # everything already done
    assert fake.unloads == ["qwen3.5:2b"]  # nothing ran, nothing to unload

    cfg = _cfg(tmp_path, data_dir, categories=["simple", "irrelevance"])
    cfg.models = ["qwen3.5:2b", "llama3.2:3b"]
    summary = run_bfcl(cfg, fake)
    assert len(fake.calls) == 14
    assert len(_rows(tmp_path / "out")) == 14
    assert fake.unloads == ["qwen3.5:2b", "llama3.2:3b"]
    models = [g["model"] for g in summary["groups"] if g["category"] == "overall"]
    assert models == ["qwen3.5:2b", "llama3.2:3b"]


def test_model_error_is_a_scored_failure(tmp_path: Path, data_dir: Path) -> None:
    fake = FakeOllama(script=[RuntimeError("boom"), responder])
    summary = run_bfcl(_cfg(tmp_path, data_dir, categories=["multiple"]), fake)
    first = _rows(tmp_path / "out")[0]
    assert first["error"] == "RuntimeError: boom"
    assert first["score"] == {
        "correct": False,
        "error_type": "model_error",
        "detail": "RuntimeError: boom",
    }
    g = next(g for g in summary["groups"] if g["category"] == "multiple")
    assert g["error_rate"] == 0.5 and g["accuracy"] == 0.5


def test_empty_reply_is_flagged(tmp_path: Path, data_dir: Path) -> None:
    # Ollama swallowed a call to an unknown (e.g. dotted) name: tokens generated, nothing returned
    fake = FakeOllama(script=[{"content": ""}, "Paris weather is sunny."])
    summary = run_bfcl(_cfg(tmp_path, data_dir, categories=["multiple"]), fake)
    empty, prose = _rows(tmp_path / "out")
    assert empty["empty_reply"] and not prose["empty_reply"]
    assert empty["score"]["error_type"] == "ast_decoder:decoder_failed"
    assert "Empty reply after 10 generated tokens" in empty["score"]["detail"]
    g = next(g for g in summary["groups"] if g["category"] == "multiple")
    assert g["empty_reply_rate"] == 0.5 and g["no_call_rate"] == 1.0


def test_sampled_run_records_ids(tmp_path: Path, data_dir: Path) -> None:
    run_bfcl(_cfg(tmp_path, data_dir, categories=["simple"], sample=2, seed=3), scripted())
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    ids = [r["id"] for r in _rows(tmp_path / "out")]
    assert meta["sample"] == 2 and meta["seed"] == 3
    assert meta["item_ids"]["simple"] == ids and len(ids) == 2


def test_unknown_category(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown BFCL category"):
        run_bfcl(_cfg(tmp_path, data_dir, categories=["java"]), scripted())


def test_finalize_empty_and_report_only_cli(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert finalize_bfcl(empty)["groups"] == []
    assert "No runs yet." in (empty / "report.md").read_text()

    run_bfcl(_cfg(tmp_path, data_dir, categories=["parallel"]), scripted())
    out = tmp_path / "out"
    (out / "report.md").unlink()
    result = CliRunner().invoke(cli.app, ["bench", "--report-only", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "| qwen3.5:2b | parallel | 2 | 100% |" in (out / "report.md").read_text()


def test_cli_bench_suite_bfcl(
    tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = scripted()
    monkeypatch.setattr(cli, "make_client", lambda settings: fake)
    monkeypatch.setattr(
        cli, "http_get", lambda *a, **k: httpx.Response(200, json={"version": "0.18.1"})
    )
    monkeypatch.setenv("OLLAMASEARCH_CACHE_DIR", "off")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "cli-out"
    args = ["bench", "--suite", "bfcl", "--models", "qwen3.5:2b", "--categories", "parallel"]
    args += ["--sample", "0", "--data-dir", str(data_dir), "--out", str(out)]
    result = CliRunner().invoke(cli.app, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert len(_rows(out)) == 2
    assert json.loads((out / "meta.json").read_text())["suite"] == "bfcl"

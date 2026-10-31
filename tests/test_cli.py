"""CLI smoke tests with the fake Ollama client and the offline test corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from ollamasearch import cli

from .conftest import BENCH, CORPUS, FakeOllama, tool

runner = CliRunner()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeOllama:
    client = FakeOllama(default="Zarnitsa-on-Ural has 7,420 people [1].")
    monkeypatch.setattr(cli, "make_client", lambda settings: client)
    monkeypatch.setenv("OLLAMASEARCH_CACHE_DIR", "off")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env, results land in tmp
    return client


def invoke(*args: str) -> Any:
    return runner.invoke(cli.app, list(args), catch_exceptions=False)


def test_ask_rag_with_corpus(fake: FakeOllama) -> None:
    result = invoke(
        "ask", "Population of Zarnitsa-on-Ural?", "--backend", "corpus", "--corpus", str(CORPUS)
    )
    assert result.exit_code == 0, result.output
    assert "7,420" in result.output
    assert "http://corpus.local/zarnitsa.html" in result.output
    assert "rag · qwen3.5:2b" in result.output


def test_ask_json_output(fake: FakeOllama) -> None:
    result = invoke(
        "ask",
        "Orlek Bridge?",
        "--backend",
        "corpus",
        "--corpus",
        str(CORPUS),
        "--json",
        "--model",
        "llama3.2:3b",
    )
    data = json.loads(result.output)
    assert data["model"] == "llama3.2:3b" and data["mode"] == "rag"
    assert fake.calls[0]["model"] == "llama3.2:3b"


def test_ask_agent_shows_tool_calls(fake: FakeOllama) -> None:
    fake.script = [{"tool_calls": [tool("web_search", {"query": "Orlek Bridge"})]}, "2004 [1]"]
    result = invoke(
        "ask",
        "When did the Orlek Bridge open?",
        "--mode",
        "agent",
        "--backend",
        "corpus",
        "--corpus",
        str(CORPUS),
    )
    assert result.exit_code == 0
    assert "web_search" in result.output and "2004" in result.output


def test_ask_closed_needs_no_backend(fake: FakeOllama) -> None:
    result = invoke("ask", "hi", "--mode", "closed", "--backend", "searxng", "--no-think")
    assert result.exit_code == 0 and fake.calls[0]["think"] is False


def test_ask_rejects_bad_mode(fake: FakeOllama) -> None:
    result = runner.invoke(cli.app, ["ask", "q", "--mode", "psychic"])
    assert result.exit_code != 0


def test_ask_ollama_backend_requires_key(fake: FakeOllama) -> None:
    result = runner.invoke(cli.app, ["ask", "q", "--backend", "ollama"])
    assert result.exit_code != 0 and "OLLAMA_API_KEY" in result.output


def test_chat_repl(fake: FakeOllama) -> None:
    fake.script = ["first answer [1]", "second answer"]
    result = runner.invoke(
        cli.app,
        ["chat", "--backend", "corpus", "--corpus", str(CORPUS)],
        input="Orlek Bridge?\n/mode closed\n/mode bogus\nfollow up\n/clear\n/quit\n",
    )
    assert result.exit_code == 0, result.output
    assert "first answer" in result.output and "second answer" in result.output
    assert "mode=closed" in result.output and "history cleared" in result.output
    follow_up = fake.calls[1]["messages"]
    assert {"role": "assistant", "content": "first answer [1]"} in follow_up


def _http(responses: dict[str, httpx.Response | Exception]) -> Any:
    def get(url: str, params: Any = None, timeout: float = 5.0) -> httpx.Response:
        for key, resp in responses.items():
            if key in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise httpx.ConnectError("no route")

    return get


def test_doctor_all_good(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "http_get",
        _http(
            {
                "/api/version": httpx.Response(200, json={"version": "0.18.1"}),
                "/search": httpx.Response(200, json={"results": [{"url": "https://x"}]}),
            }
        ),
    )
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "0.18.1" in result.output and "qwen3.5:2b" in result.output


def test_doctor_reports_fixes(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.models = ["llama3.2:3b"]
    monkeypatch.setattr(
        cli,
        "http_get",
        _http(
            {
                "/api/version": httpx.Response(200, json={"version": "0.18.1"}),
                "/search": httpx.Response(403),
            }
        ),
    )
    result = runner.invoke(cli.app, ["doctor"], terminal_width=200)
    assert result.exit_code == 1
    assert "ollama pull qwen3.5:2b" in result.output
    assert "json" in result.output


def test_doctor_ollama_down(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "http_get", _http({"/search": httpx.ConnectError("refused")}))
    result = runner.invoke(cli.app, ["doctor"], terminal_width=200)
    assert result.exit_code == 1 and "ollama serve" in result.output


def test_bench_command_and_report_only(
    fake: FakeOllama, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "http_get",
        _http(
            {
                "/api/version": httpx.Response(200, json={"version": "0.18.1"}),
            }
        ),
    )
    out = tmp_path / "res"
    result = invoke(
        "bench",
        "--models",
        "qwen3.5:2b",
        "--modes",
        "closed,rag",
        "--limit",
        "2",
        "--out",
        str(out),
        "--bench-dir",
        str(BENCH),
    )
    assert result.exit_code == 0, result.output
    assert len((out / "runs.jsonl").read_text().splitlines()) == 4
    assert "SearchBench summary" in result.output
    (out / "report.md").unlink()
    assert invoke("bench", "--report-only", "--out", str(out)).exit_code == 0
    assert (out / "report.md").exists()


def test_bench_missing_model(fake: FakeOllama) -> None:
    result = runner.invoke(cli.app, ["bench", "--models", "nope:1b", "--bench-dir", str(BENCH)])
    assert result.exit_code == 1 and "ollama pull nope:1b" in result.output


def test_real_client_has_timeout() -> None:
    from ollamasearch.config import Settings

    client: Any = cli.make_client(Settings(llm_timeout=7.0, ollama_host="http://127.0.0.1:1"))
    assert client._client.timeout.read == 7.0  # a stalled Ollama stream can't hang forever
    assert client._client.timeout.connect == 5.0


def test_version(fake: FakeOllama) -> None:
    assert "ollamasearch" in invoke("version").output

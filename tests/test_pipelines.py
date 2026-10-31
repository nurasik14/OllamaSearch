from __future__ import annotations

from typing import Any

import pytest

from ollamasearch.config import Settings
from ollamasearch.fetch import Fetcher
from ollamasearch.llm import LLM
from ollamasearch.pipelines import (
    AgentPipeline,
    ClosedBookPipeline,
    RagPipeline,
    Source,
    build_pipeline,
    extract_citations,
    parse_text_tool_calls,
)
from ollamasearch.prompts import NOT_FOUND
from ollamasearch.search import CorpusBackend, SearchError, SearchResult

from .conftest import FakeOllama, tool


def user_prompt(fake: FakeOllama, i: int = 0) -> str:
    return str(fake.calls[i]["messages"][-1]["content"])


# ---------------------------------------------------------------- closed / rag


def test_closed_book(fake: FakeOllama, llm: LLM) -> None:
    fake.script = ["Astana."]
    answer = ClosedBookPipeline(llm).run("Capital of Kazakhstan?")
    assert answer.text == "Astana." and answer.mode == "closed" and answer.sources == []
    assert answer.tokens["eval_tokens"] == 10
    assert "tools" not in fake.calls[0]


def test_closed_book_llm_error_is_captured(fake: FakeOllama, llm: LLM) -> None:
    fake.script = [ConnectionError("ollama down")]
    answer = ClosedBookPipeline(llm).run("q")
    assert answer.text == "" and answer.error is not None and "ollama down" in answer.error


def test_rag_builds_numbered_context_and_citations(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    fake.script = ["Zarnitsa-on-Ural has 7,420 people [1]."]
    answer = RagPipeline(llm, corpus, corpus_fetcher, top_k=3).run(
        "What is the population of Zarnitsa-on-Ural?"
    )
    prompt = user_prompt(fake)
    assert "[1] Zarnitsa-on-Ural" in prompt and "7,420" in prompt
    assert "cookies" not in prompt  # boilerplate never reaches the model
    assert answer.sources[0].url == "http://corpus.local/zarnitsa.html"
    assert answer.cited == [1]
    assert answer.cited_urls == ["http://corpus.local/zarnitsa.html"]
    assert set(answer.timings) >= {"search_s", "fetch_s", "rank_s", "llm_s", "total_s"}


def test_rag_respects_context_budget(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    RagPipeline(llm, corpus, corpus_fetcher, top_k=5, context_chars=400, source_chars=400).run(
        "Orlek Bridge Tamarisk Engineering"
    )
    sources_block = user_prompt(fake).split("Question:")[0]
    assert len(sources_block) < 400 + 400  # passages + headers


def test_rag_streams_tokens(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    fake.script = ["It is 48 metres [1]."]
    tokens: list[str] = []
    answer = RagPipeline(llm, corpus, corpus_fetcher).run(
        "глубина озера Тенгизколь", on_token=tokens.append
    )
    assert "".join(tokens) == answer.text == "It is 48 metres [1]."
    assert fake.calls[0]["stream"] is True


class BrokenSearch:
    name = "broken"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        raise SearchError("SearXNG down")


def test_rag_search_failure_still_answers(
    fake: FakeOllama, llm: LLM, corpus_fetcher: Fetcher
) -> None:
    fake.script = [NOT_FOUND]
    answer = RagPipeline(llm, BrokenSearch(), corpus_fetcher).run("q")
    assert answer.error == "search: SearXNG down"
    assert answer.text == NOT_FOUND
    assert "(no sources could be retrieved)" in user_prompt(fake)


class EmptySearch:
    name = "empty"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return []


def test_rag_empty_search_is_reported_as_error(
    fake: FakeOllama, llm: LLM, corpus_fetcher: Fetcher
) -> None:
    # SearXNG returns [] when its upstream engines are rate-limited; that must not look
    # like a model abstention in benchmark results.
    fake.script = [NOT_FOUND]
    answer = RagPipeline(llm, EmptySearch(), corpus_fetcher).run("q")
    assert answer.error == "search: no results"
    assert answer.text == NOT_FOUND


class SnippetOnlySearch:
    name = "snip"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return [
            SearchResult("Dead page", "http://corpus.local/dead.html", "Velmora has 13,050 people")
        ]


def test_rag_falls_back_to_snippet_when_fetch_fails(
    fake: FakeOllama, llm: LLM, corpus_fetcher: Fetcher
) -> None:
    RagPipeline(llm, SnippetOnlySearch(), corpus_fetcher).run("Velmora population")
    assert "13,050" in user_prompt(fake)


def test_rag_history_is_included(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    RagPipeline(llm, corpus, corpus_fetcher).run("Orlek Bridge", history=history)
    assert fake.calls[0]["messages"][1:3] == history


def test_extract_citations() -> None:
    sources = [Source(1, "https://a.com/x", "A"), Source(2, "https://b.com", "B")]
    assert extract_citations("See [2] and [1, 2] and [7].", sources) == [2, 1]
    assert extract_citations("Source: https://b.com/.", sources) == [2]
    assert extract_citations("No cites", sources) == []


def test_build_pipeline(llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher) -> None:
    s = Settings(cache_dir=None, agent_max_steps=2)
    assert isinstance(build_pipeline("closed", llm, None, None, s), ClosedBookPipeline)
    assert isinstance(build_pipeline("rag", llm, corpus, corpus_fetcher, s), RagPipeline)
    agent = build_pipeline("agent", llm, corpus, corpus_fetcher, s)
    assert isinstance(agent, AgentPipeline) and agent.max_steps == 2
    with pytest.raises(ValueError):
        build_pipeline("rag", llm, None, None, s)
    with pytest.raises(ValueError):
        build_pipeline("magic", llm, corpus, corpus_fetcher, s)


# ---------------------------------------------------------------- agent


@pytest.fixture
def agent(llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher) -> AgentPipeline:
    return AgentPipeline(llm, corpus, corpus_fetcher, max_steps=4)


def tool_messages(fake: FakeOllama, call_index: int) -> list[dict[str, Any]]:
    return [m for m in fake.calls[call_index]["messages"] if m["role"] == "tool"]


def test_agent_multi_hop_search_fetch_answer(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        {"tool_calls": [tool("web_search", {"query": "Orlek Bridge builder"})]},
        {"tool_calls": [tool("web_fetch", {"url": "http://corpus.local/orlek-bridge.html"})]},
        {"tool_calls": [tool("web_search", {"query": "Tamarisk Engineering founded"})]},
        "Tamarisk Engineering was founded in 1987 [2].",
    ]
    answer = agent.run("In what year was the company that built the Orlek Bridge founded?")
    assert [c.name for c in answer.tool_calls] == ["web_search", "web_fetch", "web_search"]
    assert all(c.status == "ok" for c in answer.tool_calls)
    assert answer.text.startswith("Tamarisk") and answer.steps == 4
    assert not answer.step_limit_reached
    # the fetched page is shown to the model as ranked passages with its source number
    fetched = tool_messages(fake, 2)[-1]["content"]
    assert fetched.startswith("[1] Orlek Bridge") and "Tamarisk Engineering" in fetched
    numbers = {s.url: s.number for s in answer.sources}
    assert answer.cited_urls == [u for u, n in numbers.items() if n == 2]
    assert "http://corpus.local/tamarisk.html" in answer.context_urls
    # the assistant tool-call turn is replayed in the history
    assert fake.calls[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "web_search"


def test_agent_answers_without_tools(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = ["I already know: 42."]
    answer = agent.run("q")
    assert answer.text == "I already know: 42." and answer.tool_calls == [] and answer.steps == 1


def test_agent_unknown_tool_and_malformed_args(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        {
            "tool_calls": [
                tool("browse", {"query": "x"}),
                tool("web_search", {"wrong": 1}),
                tool("web_fetch", None),
            ]
        },
        NOT_FOUND,
    ]
    answer = agent.run("q")
    assert [c.status for c in answer.tool_calls] == ["invalid", "invalid", "invalid"]
    outputs = [m["content"] for m in tool_messages(fake, 1)]
    assert "unknown tool 'browse'" in outputs[0]
    assert "requires a non-empty string argument 'query'" in outputs[1]
    assert "argument 'url'" in outputs[2]
    assert answer.text == NOT_FOUND


def test_agent_lenient_argument_aliases(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        {"tool_calls": [tool("web_search", {"q": "Zarnitsa population"})]},
        {"tool_calls": [tool("web_search", "Orlek Bridge")]},
        "done",
    ]
    answer = agent.run("q")
    assert [c.status for c in answer.tool_calls] == ["ok", "ok"]


def test_agent_step_limit_forces_final_answer(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    fake.script = [
        {"tool_calls": [tool("web_search", {"query": f"query {i}"})]} for i in range(2)
    ] + ["Final after limit."]
    answer = AgentPipeline(llm, corpus, corpus_fetcher, max_steps=2).run("q")
    assert answer.step_limit_reached and answer.steps == 2
    assert answer.text == "Final after limit."
    final_call = fake.calls[-1]
    assert "tools" not in final_call
    assert "Tool budget exhausted" in final_call["messages"][-1]["content"]


def test_agent_duplicate_call_not_rerun(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        {"tool_calls": [tool("web_search", {"query": "Orlek Bridge"})]},
        {"tool_calls": [tool("web_search", {"query": "orlek bridge"})]},
        "ok",
    ]
    answer = agent.run("q")
    assert answer.tool_calls[1].detail == "duplicate call, not re-run"
    assert "already called web_search" in tool_messages(fake, 2)[-1]["content"]


def test_agent_fetch_error_is_reported_to_model(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        {"tool_calls": [tool("web_fetch", {"url": "http://corpus.local/missing.html"})]},
        {"tool_calls": [tool("web_fetch", {"url": "file:///etc/passwd"})]},
        NOT_FOUND,
    ]
    answer = agent.run("q")
    assert [c.status for c in answer.tool_calls] == ["error", "error"]
    assert "HTTP 404" in tool_messages(fake, 1)[-1]["content"]
    assert answer.sources == []


def test_agent_search_backend_failure(fake: FakeOllama, llm: LLM, corpus_fetcher: Fetcher) -> None:
    fake.script = [{"tool_calls": [tool("web_search", {"query": "x"})]}, NOT_FOUND]
    answer = AgentPipeline(llm, BrokenSearch(), corpus_fetcher).run("q")
    assert answer.tool_calls[0].status == "error"
    assert "Search failed" in tool_messages(fake, 1)[-1]["content"]


def test_agent_recovers_text_json_tool_calls(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [
        '{"name": "web_search", "arguments": {"query": "Zarnitsa-on-Ural population"}}',
        '<tool_call>\n{"name": "web_fetch", "arguments": '
        '{"url": "http://corpus.local/zarnitsa.html"}}\n</tool_call>',
        "7,420 [1]",
    ]
    answer = agent.run("q")
    assert [(c.name, c.from_text, c.status) for c in answer.tool_calls] == [
        ("web_search", True, "ok"),
        ("web_fetch", True, "ok"),
    ]
    assert answer.text == "7,420 [1]"


def test_parse_text_tool_calls_rejects_non_calls() -> None:
    assert parse_text_tool_calls(
        '```json\n{"name": "web_search", "parameters": {"query": "x"}}\n```'
    )
    assert parse_text_tool_calls('{"answer": 42}') == []
    assert parse_text_tool_calls('{"name": "rm_rf", "arguments": {}}') == []
    assert parse_text_tool_calls("{not json}") == []
    assert parse_text_tool_calls("Plain answer.") == []


def test_agent_llm_failure_mid_loop(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [{"tool_calls": [tool("web_search", {"query": "x"})]}, RuntimeError("OOM")]
    answer = agent.run("q")
    assert answer.error == "llm: OOM" and answer.text == ""
    assert len(answer.tool_calls) == 1


def test_agent_streams_final_answer(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = ["Short answer."]
    tokens: list[str] = []
    agent.run("q", on_token=tokens.append)
    assert "".join(tokens) == "Short answer."


def test_agent_records_tokens_across_steps(fake: FakeOllama, agent: AgentPipeline) -> None:
    fake.script = [{"tool_calls": [tool("web_search", {"query": "x y"})]}, "done"]
    answer = agent.run("q")
    assert answer.tokens["calls"] == 2 and answer.tokens["eval_tokens"] == 20
    assert answer.to_dict()["cited_urls"] == []

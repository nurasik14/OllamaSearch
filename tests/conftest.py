"""Shared fixtures: a scripted fake Ollama client and a tiny offline corpus."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from ollama import ChatResponse, Message

from ollamasearch.config import Settings
from ollamasearch.fetch import Fetcher
from ollamasearch.llm import LLM
from ollamasearch.search import CorpusBackend

DATA = Path(__file__).parent / "data"
BENCH = DATA / "bench"
CORPUS = BENCH / "corpus"

Script = str | dict[str, Any] | Exception | Callable[[dict[str, Any]], Any]


def tool(name: str, arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


class FakeOllama:
    """Scripted stand-in for ``ollama.Client``.

    Each script item is a reply: a string (content), a dict with ``content``/``tool_calls``/
    ``thinking``, an Exception to raise, or a callable receiving the chat kwargs.
    """

    def __init__(
        self, script: list[Script] | None = None, default: str = "Default answer."
    ) -> None:
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict[str, Any]] = []
        self.unloads: list[str] = []
        self.models = ["qwen3.5:2b", "llama3.2:3b"]

    def list(self) -> dict[str, Any]:
        return {"models": [{"model": m} for m in self.models]}

    def _reply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        item = self.script.pop(0) if self.script else self.default
        if callable(item) and not isinstance(item, Exception):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            item = {"content": item}
        return item

    @staticmethod
    def _response(item: dict[str, Any], content: str | None = None, done: bool = True) -> Any:
        calls = item.get("tool_calls") or []
        stats = {
            "eval_count": 10,
            "eval_duration": 200_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 50_000_000,
            "total_duration": 300_000_000,
            "load_duration": 1_000_000,
        }
        if any(not isinstance(c["arguments"], dict) for c in calls):
            # pydantic would reject non-dict arguments: return a plain mapping instead
            return {
                "message": {
                    "role": "assistant",
                    "content": item.get("content", ""),
                    "tool_calls": [{"function": c} for c in calls],
                },
                "done": done,
                **stats,
            }
        msg = Message(
            role="assistant",
            content=item.get("content", "") if content is None else content,
            thinking=item.get("thinking"),
            tool_calls=[Message.ToolCall(function=Message.ToolCall.Function(**c)) for c in calls]
            or None,
        )
        return ChatResponse(model="fake", message=msg, done=done, **(stats if done else {}))

    def chat(self, **kwargs: Any) -> Any:
        if not kwargs.get("messages") and kwargs.get("keep_alive") == 0:
            self.unloads.append(kwargs["model"])
            return ChatResponse(model=kwargs["model"], message=Message(role="assistant"), done=True)
        self.calls.append(kwargs)
        item = self._reply(kwargs)
        if kwargs.get("stream"):
            text = item.get("content", "")
            words = text.split(" ")
            chunks = [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]

            def gen() -> Iterator[Any]:
                for chunk in chunks[:-1]:
                    yield self._response({}, content=chunk, done=False)
                yield self._response({}, content=chunks[-1], done=True)

            return gen()
        return self._response(item)


@pytest.fixture
def fake() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def llm(fake: FakeOllama) -> LLM:
    return LLM(fake, "qwen3.5:2b")


@pytest.fixture(scope="session")
def corpus() -> CorpusBackend:
    return CorpusBackend(CORPUS)


@pytest.fixture
def corpus_fetcher(corpus: CorpusBackend) -> Iterator[Fetcher]:
    with Fetcher(transport=corpus.transport()) as f:
        yield f


@pytest.fixture
def settings() -> Settings:
    return Settings(cache_dir=None)

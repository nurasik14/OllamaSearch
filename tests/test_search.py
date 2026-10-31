from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ollamasearch.fetch import Fetcher
from ollamasearch.search import (
    CorpusBackend,
    OllamaWebBackend,
    RecordingSearch,
    SearchBackend,
    SearchError,
    SearchResult,
    SearxngBackend,
    dedupe,
)


def test_corpus_search_finds_right_page(corpus: CorpusBackend) -> None:
    results = corpus.search("population of Zarnitsa-on-Ural", 3)
    assert results[0].url == "http://corpus.local/zarnitsa.html"
    assert results[0].title.startswith("Zarnitsa")
    assert "7,420" in results[0].snippet


def test_corpus_search_multilingual(corpus: CorpusBackend) -> None:
    assert corpus.search("глубина озера Тенгизколь", 1)[0].url.endswith("tengizkol.html")
    assert corpus.search("Ақсуат ауылының халқы", 1)[0].url.endswith("aksuat.html")


def test_corpus_search_no_match(corpus: CorpusBackend) -> None:
    assert corpus.search("qwertyuiop", 5) == []


def test_corpus_is_a_search_backend(corpus: CorpusBackend) -> None:
    assert isinstance(corpus, SearchBackend)


def test_corpus_transport_serves_pages_through_fetcher(corpus: CorpusBackend) -> None:
    with Fetcher(transport=corpus.transport()) as f:
        page = f.fetch("http://corpus.local/orlek-bridge.html")
        missing = f.fetch("http://corpus.local/nope.html")
        external = f.fetch("https://example.com/")
    assert page.ok and "Tamarisk Engineering" in page.text
    assert missing.error == "HTTP 404"
    assert external.error == "HTTP 404"  # offline: never touches the network


def test_corpus_requires_html(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CorpusBackend(tmp_path)


def test_dedupe() -> None:
    rs = [
        SearchResult("a", "https://X.com/p/"),
        SearchResult("b", "https://x.com/p#frag"),
        SearchResult("c", "https://x.com/q"),
    ]
    assert [r.title for r in dedupe(rs)] == ["a", "c"]


def _searx(handler: Any, **kw: Any) -> SearxngBackend:
    return SearxngBackend(
        "http://searx.test/",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        backoff=0,
        **kw,
    )


def test_searxng_parses_results() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Astana",
                        "url": "https://en.wikipedia.org/wiki/Astana",
                        "content": "Capital",
                    },
                    {"title": "dup", "url": "https://en.wikipedia.org/wiki/Astana/"},
                    {"title": "bad", "url": "javascript:alert(1)"},
                    {"title": "Other", "url": "https://example.com", "content": None},
                ]
            },
        )

    results = _searx(handler, language="en").search("capital of Kazakhstan", 5)
    assert [r.title for r in results] == ["Astana", "Other"]
    assert results[0].snippet == "Capital" and results[1].snippet == ""
    assert seen["params"] == {"q": "capital of Kazakhstan", "format": "json", "language": "en"}


def test_searxng_respects_k() -> None:
    items = [{"title": str(i), "url": f"https://e.com/{i}"} for i in range(10)]
    backend = _searx(lambda req: httpx.Response(200, json={"results": items}))
    assert len(backend.search("q", 3)) == 3


def test_searxng_retries_then_succeeds() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(502)
        return httpx.Response(200, json={"results": [{"title": "t", "url": "https://e.com"}]})

    assert len(_searx(handler, retries=2).search("q")) == 1
    assert len(calls) == 3


def test_searxng_gives_up_after_retries() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(SearchError, match="cannot reach SearXNG"):
        _searx(handler, retries=1).search("q")


def test_searxng_403_explains_json_format() -> None:
    with pytest.raises(SearchError, match="json format"):
        _searx(lambda req: httpx.Response(403)).search("q")


def test_searxng_non_json() -> None:
    with pytest.raises(SearchError, match="JSON"):
        _searx(lambda req: httpx.Response(200, text="<html>")).search("q")


class FakeWeb:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def web_search(self, query: str, max_results: int = 3) -> Any:
        if self.fail:
            raise RuntimeError("401 unauthorized")
        return SimpleNamespace(
            results=[
                SimpleNamespace(title="T", url="https://e.com/a", content="snippet  text"),
                SimpleNamespace(title=None, url=None, content="no url"),
            ]
        )

    def web_fetch(self, url: str) -> Any:
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(title="Page", content="Body   text", links=[])


def test_ollama_web_backend() -> None:
    backend = OllamaWebBackend("key", client=FakeWeb())
    assert backend.search("q", 5) == [SearchResult("T", "https://e.com/a", "snippet text")]
    page = backend.fetch("https://e.com/a")
    assert page.ok and page.title == "Page" and page.text == "Body text"
    assert [r.ok for r in backend.fetch_many(["https://e.com/a", "https://e.com/b"])] == [
        True,
        True,
    ]


def test_ollama_web_backend_errors() -> None:
    backend = OllamaWebBackend("key", client=FakeWeb(fail=True))
    with pytest.raises(SearchError):
        backend.search("q")
    assert backend.fetch("https://e.com").error is not None


class ScriptedSearch:
    """Returns the scripted outcomes in order: a list of results, [] or an exception."""

    name = "scripted"

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        out = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return list(out)


HIT = [SearchResult("Astana", "https://example.org/astana", "capital")]


def test_recording_search_retries_empty_then_records(tmp_path: Path) -> None:
    sleeps: list[float] = []
    inner = ScriptedSearch([[], HIT])
    rec = RecordingSearch(inner, tmp_path / "rec.json", retries=2, backoff=5, sleep=sleeps.append)
    assert rec.search("Capital of  Kazakhstan") == HIT
    assert inner.calls == 2 and sleeps == [5]
    # same query (case/space-insensitive) is served from the record, also after a restart
    again = RecordingSearch(ScriptedSearch([[]]), tmp_path / "rec.json", sleep=sleeps.append)
    assert again.search("capital of kazakhstan") == HIT


def test_recording_search_gives_up_and_never_records_empty(tmp_path: Path) -> None:
    inner = ScriptedSearch([[]])
    rec = RecordingSearch(inner, tmp_path / "rec.json", retries=2, sleep=lambda s: None)
    assert rec.search("q") == []
    assert inner.calls == 3
    assert not (tmp_path / "rec.json").exists()


def test_recording_search_reraises_persistent_errors() -> None:
    inner = ScriptedSearch([SearchError("down")])
    rec = RecordingSearch(inner, retries=1, sleep=lambda s: None)
    with pytest.raises(SearchError, match="down"):
        rec.search("q")
    assert inner.calls == 2


def test_recording_search_recovers_from_a_transient_error() -> None:
    inner = ScriptedSearch([SearchError("blip"), HIT])
    rec = RecordingSearch(inner, retries=1, sleep=lambda s: None)
    assert rec.search("q") == HIT


def test_recording_search_ignores_corrupt_record(tmp_path: Path) -> None:
    path = tmp_path / "rec.json"
    path.write_text("{not json", encoding="utf-8")
    rec = RecordingSearch(ScriptedSearch([HIT]), path, sleep=lambda s: None)
    assert rec.search("q") == HIT

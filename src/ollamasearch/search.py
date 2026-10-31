"""Search backends: SearXNG (default, private), Ollama hosted web search, and a local corpus."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx

from .extract import Page, extract, normalize_whitespace
from .fetch import FetchResult
from .rank import BM25, Document, select_passages, tokenize

log = logging.getLogger(__name__)

CORPUS_HOST = "corpus.local"
CORPUS_PREFIX = f"http://{CORPUS_HOST}/"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


class SearchError(RuntimeError):
    """The search backend could not be reached or returned an unusable response."""


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[SearchResult]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...

    def fetch_many(self, urls: Sequence[str]) -> list[FetchResult]: ...


def _canonical(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def dedupe(results: Sequence[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in results:
        key = _canonical(r.url)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


class SearxngBackend:
    """Query a SearXNG instance's JSON API with a pooled client and retries."""

    name = "searxng"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 15.0,
        retries: int = 2,
        backoff: float = 0.5,
        language: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.backoff = backoff
        self.language = language
        self.client = client or httpx.Client(
            timeout=timeout, headers={"Accept": "application/json"}
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        params = {"q": query, "format": "json"}
        if self.language:
            params["language"] = self.language
        last_error = "unknown error"
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                resp = self.client.get(f"{self.base_url}/search", params=params)
            except httpx.TransportError as exc:
                last_error = f"cannot reach SearXNG at {self.base_url}: {exc}"
                continue
            if resp.status_code == 403:
                raise SearchError(
                    "SearXNG returned 403: enable the json format "
                    "(search.formats: [html, json]) in its settings.yml"
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"SearXNG HTTP {resp.status_code}"
                continue
            if resp.status_code >= 400:
                raise SearchError(f"SearXNG HTTP {resp.status_code}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise SearchError("SearXNG did not return JSON") from exc
            results = [
                SearchResult(
                    title=normalize_whitespace(str(item.get("title") or "")),
                    url=str(item["url"]),
                    snippet=normalize_whitespace(str(item.get("content") or ""))[:400],
                )
                for item in data.get("results", [])
                if isinstance(item, dict) and str(item.get("url", "")).startswith("http")
            ]
            return dedupe(results)[:k]
        raise SearchError(last_error)


class OllamaWebBackend:
    """Ollama's hosted ``web_search``/``web_fetch`` (needs OLLAMA_API_KEY; not private)."""

    name = "ollama"

    def __init__(self, api_key: str, client: Any | None = None, concurrency: int = 4) -> None:
        if client is None:
            import ollama

            client = ollama.Client(headers={"Authorization": f"Bearer {api_key}"})
        self.client = client
        self.concurrency = concurrency

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        try:
            resp = self.client.web_search(query, max_results=max(1, min(k, 10)))
        except Exception as exc:
            raise SearchError(f"Ollama web_search failed: {exc}") from exc
        return dedupe(
            [
                SearchResult(
                    title=r.title or "",
                    url=r.url,
                    snippet=normalize_whitespace(r.content or "")[:400],
                )
                for r in resp.results
                if r.url
            ]
        )[:k]

    def fetch(self, url: str) -> FetchResult:
        start = time.perf_counter()
        try:
            resp = self.client.web_fetch(url)
        except Exception as exc:
            return FetchResult(url=url, error=f"web_fetch failed: {exc}"[:300])
        text = normalize_whitespace(resp.content or "")
        return FetchResult(
            url=url,
            final_url=url,
            status=200,
            title=resp.title or "",
            text=text,
            error=None if text else "no readable text",
            elapsed=time.perf_counter() - start,
        )

    def fetch_many(self, urls: Sequence[str]) -> list[FetchResult]:
        with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency, len(urls) or 1))) as pool:
            return list(pool.map(self.fetch, urls))


class CorpusBackend:
    """BM25 search over a folder of HTML files, served at ``http://corpus.local/<file>``.

    Use :meth:`transport` to plug the same pages into :class:`~ollamasearch.fetch.Fetcher`,
    so pipelines exercise the real fetch -> extract -> rank path without any network.
    """

    name = "corpus"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        files = sorted(self.directory.glob("*.html"))
        if not files:
            raise FileNotFoundError(f"no .html files in corpus directory {self.directory}")
        self.raw: dict[str, str] = {f.name: f.read_text(encoding="utf-8") for f in files}
        self.pages: dict[str, Page] = {name: extract(html) for name, html in self.raw.items()}
        self.names = list(self.pages)
        self.bm25 = BM25([tokenize(p.title) * 2 + tokenize(p.text) for p in self.pages.values()])

    @staticmethod
    def url_for(name: str) -> str:
        return CORPUS_PREFIX + name

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        results = []
        for index, _score in self.bm25.top(tokenize(query), k):
            name = self.names[index]
            page = self.pages[name]
            best = select_passages(
                query,
                [Document(0, page.title, page.text)],
                budget_chars=300,
                per_source_chars=300,
                passage_chars=300,
            )
            snippet = best[0].text if best else page.text[:300]
            results.append(SearchResult(page.title, self.url_for(name), snippet[:300]))
        return results

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            name = request.url.path.lstrip("/")
            if request.url.host == CORPUS_HOST and name in self.raw:
                return httpx.Response(
                    200,
                    text=self.raw[name],
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            return httpx.Response(404, text="not in offline corpus")

        return httpx.MockTransport(handler)


class RecordingSearch:
    """Wrap a live backend for benchmarking: retry empty or failed searches, and record results.

    SearXNG returns ``[]`` (not an error) while its upstream engines are rate-limited, so an
    empty list is retried with backoff before it is accepted. Non-empty results are stored per
    ``(query, k)`` in a JSON file, so every model in a run (and any resumed run) sees exactly
    the same search results for the same query. Empty results are never recorded.
    """

    def __init__(
        self,
        inner: SearchBackend,
        path: Path | None = None,
        *,
        retries: int = 2,
        backoff: float = 5.0,
        sleep: Any = time.sleep,
    ) -> None:
        self.inner = inner
        self.name = f"{inner.name}+recorded"
        self.path = path
        self.retries = max(0, retries)
        self.backoff = backoff
        self._sleep = sleep
        self._lock = threading.Lock()
        self.cache: dict[str, list[dict[str, str]]] = {}
        if path is not None and path.exists():
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("ignoring unreadable search record %s", path)

    @staticmethod
    def _key(query: str, k: int) -> str:
        return f"{k}\t{normalize_whitespace(query).casefold()}"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        key = self._key(query, k)
        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            return [SearchResult(**r) for r in hit]
        last_error: SearchError | None = None
        results: list[SearchResult] = []
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(self.backoff * attempt)
            try:
                results = self.inner.search(query, k)
                last_error = None
            except SearchError as exc:
                last_error = exc
                continue
            if results:
                break
            log.info("empty search for %r (attempt %d)", query, attempt + 1)
        if last_error is not None and not results:
            raise last_error
        if results:
            self._store(key, results)
        return results

    def _store(self, key: str, results: Sequence[SearchResult]) -> None:
        with self._lock:
            self.cache[key] = [asdict(r) for r in results]
            if self.path is not None:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), "utf-8")
                tmp.replace(self.path)

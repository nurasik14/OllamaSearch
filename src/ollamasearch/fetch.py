"""Concurrent, polite page fetching. Never raises for a single bad page."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import ssl
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from . import __version__
from .extract import extract, looks_like_html, normalize_whitespace

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

USER_AGENT = (
    f"Mozilla/5.0 (X11; Linux x86_64) ollamasearch/{__version__} "
    "(+https://github.com/nurasik14/OllamaSearch)"
)
HTML_TYPES = ("text/html", "application/xhtml+xml")
TEXT_TYPES = ("text/plain", "text/markdown")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([\w-]+)""", re.IGNORECASE)


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int | None = None
    content_type: str = ""
    title: str = ""
    text: str = ""
    error: str | None = None
    elapsed: float = 0.0
    from_cache: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def make_ssl_context(use_truststore: bool) -> ssl.SSLContext | bool:
    """Prefer the OS trust store (fixes sites with intermediate-chain issues, e.g. some .kz)."""
    if not use_truststore:
        return True
    try:
        import truststore

        ctx: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ctx
    except Exception:  # pragma: no cover - platform specific
        log.warning("truststore unavailable, falling back to certifi", exc_info=True)
        return True


def _decode(body: bytes, content_type: str) -> str:
    charset = None
    m = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    if m:
        charset = m.group(1)
    else:
        mm = _META_CHARSET.search(body[:4096])
        if mm:
            charset = mm.group(1).decode("ascii", "ignore")
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class PageCache:
    """Tiny on-disk cache of extracted pages, one JSON file per URL."""

    def __init__(self, directory: Path, ttl: float = 86400.0) -> None:
        self.directory = directory
        self.ttl = ttl

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()
        return self.directory / digest[:2] / f"{digest}.json"

    def get(self, url: str) -> FetchResult | None:
        path = self._path(url)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            result = FetchResult(**data)
        except (OSError, ValueError, TypeError):
            return None
        result.from_cache = True
        result.elapsed = 0.0
        return result

    def put(self, result: FetchResult) -> None:
        path = self._path(result.url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            log.debug("cache write failed for %s", result.url, exc_info=True)


class Fetcher:
    """Fetch pages with bounded global and per-host concurrency.

    Pass ``transport`` (e.g. :class:`httpx.MockTransport`) to serve pages without network;
    the offline benchmark and the tests do exactly that.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 15.0,
        max_bytes: int = 3_000_000,
        concurrency: int = 8,
        per_host: int = 2,
        cache_dir: Path | None = None,
        cache_ttl: float = 86400.0,
        use_truststore: bool = True,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.max_bytes = max_bytes
        self.concurrency = max(1, concurrency)
        self.per_host = max(1, per_host)
        self.deadline = connect_timeout + 2 * read_timeout
        self.cache = PageCache(cache_dir, cache_ttl) if cache_dir else None
        self._host_locks: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()
        self.client = httpx.Client(
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            follow_redirects=True,
            max_redirects=5,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5",
                "Accept-Language": "en,ru;q=0.9,kk;q=0.8",
            },
            limits=httpx.Limits(max_connections=self.concurrency * 2),
            transport=transport,
            verify=make_ssl_context(use_truststore) if transport is None else True,
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, transport: httpx.BaseTransport | None = None
    ) -> Fetcher:
        return cls(
            connect_timeout=settings.connect_timeout,
            read_timeout=settings.read_timeout,
            max_bytes=settings.max_page_bytes,
            concurrency=settings.fetch_concurrency,
            per_host=settings.per_host_concurrency,
            cache_dir=settings.cache_dir if transport is None else None,
            use_truststore=settings.use_truststore,
            transport=transport,
        )

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _host_semaphore(self, host: str) -> threading.Semaphore:
        with self._lock:
            sem = self._host_locks.get(host)
            if sem is None:
                sem = self._host_locks[host] = threading.Semaphore(self.per_host)
            return sem

    def fetch(self, url: str) -> FetchResult:
        """Fetch and extract one page. Errors are returned in ``FetchResult.error``."""
        url = url.strip()
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            return FetchResult(url=url, error="unsupported URL (only http/https)")
        if self.cache is not None and (cached := self.cache.get(url)) is not None:
            log.debug("cache hit %s", url)
            return cached

        start = time.perf_counter()
        with self._host_semaphore(parts.netloc.lower()):
            result = self._fetch_uncached(url)
        result.elapsed = time.perf_counter() - start
        if result.ok:
            log.debug("fetched %s (%d chars, %.2fs)", url, len(result.text), result.elapsed)
            if self.cache is not None:
                self.cache.put(result)
        else:
            log.info("fetch failed %s: %s", url, result.error)
        return result

    def _fetch_uncached(self, url: str) -> FetchResult:
        result = FetchResult(url=url)
        start = time.monotonic()
        try:
            with self.client.stream("GET", url) as resp:
                result.status = resp.status_code
                result.final_url = str(resp.url)
                ctype = resp.headers.get("content-type", "").lower()
                result.content_type = ctype.split(";")[0].strip()
                if resp.status_code >= 400:
                    result.error = f"HTTP {resp.status_code}"
                    return result
                if result.content_type and not result.content_type.startswith(
                    HTML_TYPES + TEXT_TYPES
                ):
                    result.error = f"unsupported content-type {result.content_type}"
                    return result
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > self.max_bytes * 4:
                    result.error = f"page too large ({declared} bytes)"
                    return result
                chunks: list[bytes] = []
                size = 0
                for chunk in resp.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= self.max_bytes:
                        result.truncated = True
                        break
                    if time.monotonic() - start > self.deadline:
                        result.error = "timeout (total deadline exceeded)"
                        return result
                body = b"".join(chunks)[: self.max_bytes]
        except httpx.TimeoutException:
            result.error = "timeout"
            return result
        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            return result
        except (ssl.SSLError, OSError, ValueError) as exc:  # pragma: no cover - defensive
            result.error = f"{type(exc).__name__}: {exc}"[:300]
            return result

        text = _decode(body, ctype)
        if result.content_type.startswith(HTML_TYPES) or (
            not result.content_type and looks_like_html(text)
        ):
            page = extract(text)
            result.title, result.text = page.title, page.text
        else:
            result.text = normalize_whitespace(text)
        if not result.text:
            result.error = "no readable text"
        return result

    def fetch_many(self, urls: Sequence[str]) -> list[FetchResult]:
        """Fetch concurrently; results keep the order of ``urls``.

        The whole batch has a hard wall-clock budget. httpx timeouts don't cover DNS
        resolution (``getaddrinfo`` can block for minutes on a broken resolver path), so
        workers are daemon threads and any page still running at the budget is reported
        as a timeout and abandoned instead of stalling the answer.
        """
        if not urls:
            return []
        results: list[FetchResult | None] = [None] * len(urls)
        gate = threading.Semaphore(self.concurrency)

        def work(i: int, url: str) -> None:
            with gate:
                results[i] = self.fetch(url)

        threads = [
            threading.Thread(target=work, args=(i, u), daemon=True, name=f"fetch-{i}")
            for i, u in enumerate(urls)
        ]
        end = time.monotonic() + self.batch_budget(len(urls))
        for t in threads:
            t.start()
        for t in threads:
            t.join(max(0.0, end - time.monotonic()))
        out = []
        for url, res in zip(urls, results, strict=True):
            if res is None:
                log.info("fetch abandoned after the batch deadline: %s", url)
                res = FetchResult(url=url, error="timeout (total deadline exceeded)")
            out.append(res)
        return out

    def batch_budget(self, n: int) -> float:
        """Seconds allowed for fetching ``n`` pages: one deadline per concurrency wave."""
        waves = -(-max(1, n) // self.concurrency)
        return self.deadline * waves + 1.0

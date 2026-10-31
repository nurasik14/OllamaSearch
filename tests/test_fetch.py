from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from ollamasearch.config import Settings
from ollamasearch.fetch import Fetcher, PageCache

from .http_server import serve


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    with serve() as url:
        yield url


@pytest.fixture
def fetcher() -> Iterator[Fetcher]:
    with Fetcher(read_timeout=0.5, connect_timeout=0.5, max_bytes=200_000) as f:
        yield f


def test_ok_page_extracted(base: str, fetcher: Fetcher) -> None:
    r = fetcher.fetch(f"{base}/ok")
    assert r.ok and r.status == 200
    assert r.title == "OK page"
    assert "13,050" in r.text
    assert r.content_type == "text/html"
    assert r.elapsed > 0


@pytest.mark.parametrize(
    ("path", "error"),
    [("/forbidden", "HTTP 403"), ("/missing", "HTTP 404"), ("/error", "HTTP 500")],
)
def test_http_errors_do_not_raise(base: str, fetcher: Fetcher, path: str, error: str) -> None:
    r = fetcher.fetch(base + path)
    assert not r.ok and r.error == error


def test_timeout(base: str, fetcher: Fetcher) -> None:
    start = time.perf_counter()
    r = fetcher.fetch(f"{base}/slow")
    assert r.error is not None and "timeout" in r.error
    assert time.perf_counter() - start < 1.4


def test_huge_page_truncated(base: str, fetcher: Fetcher) -> None:
    r = fetcher.fetch(f"{base}/stream")
    assert r.ok and r.truncated
    assert len(r.text) < 200_000


def test_huge_declared_length_rejected(base: str) -> None:
    with Fetcher(max_bytes=1000) as f:
        r = f.fetch(f"{base}/huge")
    assert r.error is not None and "too large" in r.error


def test_non_html_rejected(base: str, fetcher: Fetcher) -> None:
    r = fetcher.fetch(f"{base}/pdf")
    assert r.error == "unsupported content-type application/pdf"


def test_plain_text_and_sniffing(base: str, fetcher: Fetcher) -> None:
    assert fetcher.fetch(f"{base}/text").text == "plain text\nfile"
    r = fetcher.fetch(f"{base}/notype")
    assert r.title == "No type" and "sniffed html" in r.text


def test_charsets(base: str, fetcher: Fetcher) -> None:
    assert "Население 13 050 человек" in fetcher.fetch(f"{base}/cp1251").text
    assert "Привет мир" in fetcher.fetch(f"{base}/metacharset").text


def test_redirect_followed(base: str, fetcher: Fetcher) -> None:
    r = fetcher.fetch(f"{base}/redirect")
    assert r.ok and r.final_url.endswith("/ok")


def test_no_readable_text(base: str, fetcher: Fetcher) -> None:
    assert fetcher.fetch(f"{base}/empty").error == "no readable text"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "not a url", ""])
def test_unsupported_urls(fetcher: Fetcher, url: str) -> None:
    assert fetcher.fetch(url).error == "unsupported URL (only http/https)"


def test_connection_refused(fetcher: Fetcher) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = fetcher.fetch(f"http://127.0.0.1:{port}/")
    assert not r.ok and "Connect" in (r.error or "")


def test_fetch_many_keeps_order_and_isolates_failures(base: str, fetcher: Fetcher) -> None:
    urls = [f"{base}/ok", f"{base}/forbidden", f"{base}/text", "file:///x"]
    results = fetcher.fetch_many(urls)
    assert [r.url for r in results] == urls
    assert [r.ok for r in results] == [True, False, True, False]
    assert fetcher.fetch_many([]) == []


def test_fetch_many_is_concurrent() -> None:
    with serve(delay=0.2) as url, Fetcher(concurrency=8, per_host=8) as f:
        start = time.perf_counter()
        results = f.fetch_many([f"{url}/ok?{i}" for i in range(8)])
        elapsed = time.perf_counter() - start
    assert all(r.ok for r in results)
    assert elapsed < 0.2 * 8 / 2  # far below the sequential 1.6 s


def test_per_host_limit_serialises() -> None:
    with serve(delay=0.15) as url, Fetcher(concurrency=8, per_host=1) as f:
        start = time.perf_counter()
        f.fetch_many([f"{url}/ok?{i}" for i in range(4)])
        assert time.perf_counter() - start >= 0.55


def test_cache_roundtrip(base: str, tmp_path: Path) -> None:
    with Fetcher(cache_dir=tmp_path) as f:
        first = f.fetch(f"{base}/ok")
        second = f.fetch(f"{base}/ok")
        failed = f.fetch(f"{base}/forbidden")
        failed_again = f.fetch(f"{base}/forbidden")
    assert not first.from_cache and second.from_cache
    assert second.text == first.text and second.title == first.title
    assert not failed.from_cache and not failed_again.from_cache  # errors are not cached


def test_cache_expiry_and_corruption(tmp_path: Path) -> None:
    from ollamasearch.fetch import FetchResult

    cache = PageCache(tmp_path, ttl=0.0)
    cache.put(FetchResult(url="http://x", text="t"))
    time.sleep(0.01)
    assert cache.get("http://x") is None
    cache = PageCache(tmp_path, ttl=100)
    path = cache._path("http://y")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert cache.get("http://y") is None


def test_mock_transport_injection() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text="<p>mocked</p>", headers={"content-type": "text/html"})
    )
    with Fetcher(transport=transport) as f:
        assert f.fetch("https://example.invalid/").text == "mocked"


def test_fetch_many_hard_deadline_covers_hangs_outside_httpx_timeouts() -> None:
    # A blocking handler stands in for a hung DNS lookup: httpx's own timeouts never fire,
    # so only the batch deadline can stop it.
    release = threading.Event()

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host == "hang.invalid":
            release.wait(10)
        return httpx.Response(200, text="<p>fine</p>", headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    with Fetcher(transport=transport, connect_timeout=0.1, read_timeout=0.1) as f:
        t = time.perf_counter()
        ok, hung = f.fetch_many(["https://ok.invalid/", "https://hang.invalid/"])
        elapsed = time.perf_counter() - t
    release.set()
    assert ok.text == "fine"
    assert hung.error == "timeout (total deadline exceeded)"
    assert elapsed < f.batch_budget(2) + 0.5


def test_batch_budget_scales_with_concurrency_waves() -> None:
    with Fetcher(connect_timeout=1, read_timeout=2, concurrency=4) as f:
        assert f.batch_budget(1) == f.batch_budget(4) == 6.0
        assert f.batch_budget(5) == 11.0


def test_from_settings(tmp_path: Path) -> None:
    s = Settings(cache_dir=tmp_path, fetch_concurrency=3, max_page_bytes=123)
    with Fetcher.from_settings(s) as f:
        assert f.concurrency == 3 and f.max_bytes == 123 and f.cache is not None
    transport = httpx.MockTransport(lambda req: httpx.Response(404))
    with Fetcher.from_settings(s, transport=transport) as f:
        assert f.cache is None  # never cache injected (offline) transports

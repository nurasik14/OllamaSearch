"""Sequential (the prototype) vs concurrent fetching of 10 pages with 200 ms server latency."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ollamasearch.fetch import Fetcher
from tests.http_server import serve

N_PAGES = 10
DELAY = 0.2


@pytest.fixture(scope="module")
def urls() -> Iterator[list[str]]:
    with serve(delay=DELAY) as base:
        # 5 hosts' worth of paths on one server; per_host is raised so only latency matters
        yield [f"{base}/ok?page={i}" for i in range(N_PAGES)]


def test_fetch_sequential(benchmark: BenchmarkFixture, urls: list[str]) -> None:
    with Fetcher(concurrency=1, per_host=1) as f:
        results = benchmark.pedantic(lambda: [f.fetch(u) for u in urls], rounds=3, iterations=1)
    assert all(r.ok for r in results)
    assert benchmark.stats.stats.mean >= N_PAGES * DELAY


def test_fetch_concurrent(benchmark: BenchmarkFixture, urls: list[str]) -> None:
    with Fetcher(concurrency=10, per_host=10) as f:
        results = benchmark.pedantic(f.fetch_many, args=(urls,), rounds=3, iterations=1)
    assert all(r.ok for r in results)
    assert benchmark.stats.stats.mean < N_PAGES * DELAY / 3

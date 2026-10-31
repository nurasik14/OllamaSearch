"""BM25 passage ranking latency: what RAG pays per question after fetching."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ollamasearch.extract import extract
from ollamasearch.rank import Document, select_passages
from ollamasearch.search import CorpusBackend

from .conftest import realistic_page

BENCH_CORPUS = Path(__file__).resolve().parents[1] / "bench" / "corpus"


@pytest.fixture(scope="module")
def five_pages() -> list[Document]:
    """Five extracted ~300 KB pages, i.e. a heavy top-5 RAG context."""
    return [Document(i, f"p{i}", extract(realistic_page(seed=i)).text) for i in range(5)]


def test_select_passages_top5_pages(
    benchmark: BenchmarkFixture, five_pages: list[Document]
) -> None:
    chosen = benchmark(
        select_passages,
        "census population of the harbour town founded",
        five_pages,
        budget_chars=6000,
        per_source_chars=1500,
    )
    assert 0 < sum(len(p.text) for p in chosen) <= 6000
    benchmark.extra_info["input_chars"] = sum(len(d.text) for d in five_pages)


@pytest.mark.skipif(not BENCH_CORPUS.is_dir(), reason="bench corpus missing")
def test_corpus_search(benchmark: BenchmarkFixture) -> None:
    corpus = CorpusBackend(BENCH_CORPUS)
    results = benchmark(corpus.search, "who founded the company that built the bridge", 5)
    assert results

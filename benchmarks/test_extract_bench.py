"""HTML extraction throughput on a realistic ~300 KB page."""

from __future__ import annotations

import re

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ollamasearch.extract import extract


def test_extract_selectolax(benchmark: BenchmarkFixture, page_300k: str) -> None:
    assert 290_000 < len(page_300k) < 310_000
    page = benchmark(extract, page_300k)
    assert page.title == "Benchmark page"
    assert "Section 42" not in page.text  # navigation removed
    benchmark.extra_info["mb_per_s"] = round(len(page_300k) / 1e6 / benchmark.stats.stats.mean, 1)


def test_extract_beautifulsoup_baseline(benchmark: BenchmarkFixture, page_300k: str) -> None:
    """The prototype's approach (BeautifulSoup + html.parser), for comparison."""
    bs4 = pytest.importorskip("bs4")

    def clean_html(html: str) -> str:
        soup = bs4.BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

    text = benchmark(clean_html, page_300k)
    assert "Benchmark article" in text

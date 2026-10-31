"""Fixtures for performance benchmarks (run: uv run pytest benchmarks --benchmark-only)."""

from __future__ import annotations

import random

import pytest

WORDS = (
    "river town bridge company census population founded engineering museum festival "
    "mayor council railway station harbour market university research results measured "
    "steppe lake mountain season tourism history archive report annual growth"
).split()


def _sentence(rng: random.Random, n: int = 18) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def realistic_page(target_bytes: int = 300_000, seed: int = 1) -> str:
    """A news/wiki-like page: big nav, scripts, inline styles, sidebar, tables, long article."""
    rng = random.Random(seed)
    nav = "".join(f'<li><a href="/section/{i}">Section {i}</a></li>' for i in range(150))
    script = "<script>" + "var x=1;function f(a){return a*2};" * 300 + "</script>"
    style = "<style>" + ".c{color:#333;margin:0 auto;padding:4px}" * 200 + "</style>"
    sidebar = "".join(
        f'<div class="teaser"><a href="/n/{i}">{_sentence(rng, 8)}</a></div>' for i in range(60)
    )
    parts = [
        f"<!DOCTYPE html><html><head><title>Benchmark page</title>{style}{script}</head><body>",
        f'<header><nav><ul class="menu">{nav}</ul></nav></header>',
        '<div class="cookie-banner">We use cookies.</div>',
        f'<aside class="sidebar">{sidebar}</aside><main><article><h1>Benchmark article</h1>',
    ]
    size = sum(len(p) for p in parts)
    i = 0
    while size < target_bytes - 5000:
        if i % 12 == 0:
            chunk = f"<h2>Part {i}</h2>"
        elif i % 17 == 0:
            rows = "".join(
                f"<tr><td>{rng.choice(WORDS)}</td><td>{rng.randint(1, 99999):,}</td></tr>"
                for _ in range(10)
            )
            chunk = f"<table>{rows}</table>"
        else:
            chunk = "<p>" + " ".join(_sentence(rng) for _ in range(4)) + ' <a href="#">link</a></p>'
        parts.append(chunk)
        size += len(chunk)
        i += 1
    parts.append("</article></main><footer>" + nav + "</footer></body></html>")
    return "".join(parts)


@pytest.fixture(scope="session")
def page_300k() -> str:
    return realistic_page()

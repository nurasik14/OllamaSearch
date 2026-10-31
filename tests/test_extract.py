from __future__ import annotations

from ollamasearch.extract import extract, looks_like_html, normalize_whitespace

from .conftest import CORPUS

LONG = "This paragraph is long enough to make the main element count as real content. " * 4


def test_title_and_main_text() -> None:
    page = extract((CORPUS / "zarnitsa.html").read_text(encoding="utf-8"))
    assert page.title == "Zarnitsa-on-Ural – Town profile"
    assert "7,420" in page.text
    assert "Dinara Oskembay" in page.text


def test_boilerplate_is_dropped() -> None:
    page = extract((CORPUS / "zarnitsa.html").read_text(encoding="utf-8"))
    for junk in ("cookies", "Privacy policy", "About us", "99,999", "analytics", "font-family"):
        assert junk not in page.text


def test_prefers_main_over_body() -> None:
    html = f"<body><div>Outside noise</div><main><p>{LONG}</p></main></body>"
    assert "Outside noise" not in extract(html).text


def test_prefers_longest_article() -> None:
    html = f"<body><article><p>short teaser</p></article><article><p>{LONG}</p></article></body>"
    text = extract(html).text
    assert "short teaser" not in text
    assert "real content" in text


def test_falls_back_to_body_when_main_is_tiny() -> None:
    html = "<body><main><p>tiny</p></main><div><p>Body text here.</p></div></body>"
    text = extract(html).text
    assert "Body text here." in text
    assert "tiny" in text


def test_blocks_become_lines_and_inline_stays_joined() -> None:
    html = "<body><p>Hello <b>big</b> world</p><ul><li>one</li><li>two</li></ul>tail</body>"
    assert extract(html).text.split("\n") == ["Hello big world", "one", "two", "tail"]


def test_hidden_and_role_navigation_removed() -> None:
    html = (
        '<body><div hidden>secret</div><div role="navigation">menu links</div>'
        '<div aria-hidden="true">icon</div><p>visible</p></body>'
    )
    assert extract(html).text == "visible"


def test_boilerplate_classes_removed_but_main_kept() -> None:
    html = (
        f'<body><div class="share-buttons">Share on X</div>'
        f'<main class="main-menu-layout"><p>{LONG}</p></main></body>'
    )
    text = extract(html).text
    assert "Share on X" not in text
    assert "real content" in text


def test_title_fallbacks() -> None:
    assert extract('<head><meta property="og:title" content="OG"></head><body/>').title == "OG"
    assert extract("<body><h1>Heading  title</h1></body>").title == "Heading title"
    assert extract("<body><p>x</p></body>").title == ""


def test_entities_and_nbsp() -> None:
    page = extract("<body><p>Fish&nbsp;&amp;&nbsp;chips   &lt;3</p></body>")
    assert page.text == "Fish & chips <3"


def test_cyrillic_and_kazakh_text() -> None:
    page = extract((CORPUS / "aksuat.html").read_text(encoding="utf-8"))
    assert page.title == "Ақсуат ауылы"
    assert "2 150 адам" in page.text


def test_empty_and_broken_html() -> None:
    assert extract("").text == ""
    assert "unclosed" in extract("<div><p>unclosed <b>tags").text


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a \t b\n\n\n c \xa0 d​ ") == "a b\nc d"


def test_looks_like_html() -> None:
    assert looks_like_html("<!DOCTYPE html><html>")
    assert not looks_like_html("just some plain text")

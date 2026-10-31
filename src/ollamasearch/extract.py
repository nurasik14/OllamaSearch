"""HTML -> clean text, built on selectolax (lexbor), which is ~20x faster than BeautifulSoup."""

from __future__ import annotations

import re
from dataclasses import dataclass

from selectolax.lexbor import LexborHTMLParser, LexborNode

DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "select",
    "dialog",
)
DROP_SELECTORS = (
    "[hidden]",
    '[aria-hidden="true"]',
    '[role="navigation"]',
    '[role="banner"]',
    '[role="contentinfo"]',
    '[role="complementary"]',
    '[role="dialog"]',
)
BOILERPLATE_RE = re.compile(
    r"(?:^|[\s_-])(cookie|consent|gdpr|breadcrumbs?|sidebar|advert|ads|banner|newsletter|"
    r"share|social|menu|navbar|navigation|popup|modal|related|promo)(?:$|[\s_-])",
    re.IGNORECASE,
)
BLOCK_TAGS = (
    "p",
    "div",
    "section",
    "article",
    "main",
    "li",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "tr",
    "table",
    "caption",
    "blockquote",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "figcaption",
    "address",
    "br",
    "hr",
)
_INLINE_WS = re.compile(r"[^\S\n]+")
_MULTI_NL = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class Page:
    title: str
    text: str


def _is_boilerplate(node: LexborNode) -> bool:
    attrs = node.attributes
    marker = f"{attrs.get('class') or ''} {attrs.get('id') or ''}"
    return bool(marker.strip()) and BOILERPLATE_RE.search(marker) is not None


def _title(tree: LexborHTMLParser) -> str:
    for selector, attr in (
        ('meta[property="og:title"]', "content"),
        ("title", None),
        ("h1", None),
    ):
        node = tree.css_first(selector)
        if node is None:
            continue
        value = node.attributes.get(attr) if attr else node.text()
        if value and value.strip():
            return _INLINE_WS.sub(" ", value).strip()
    return ""


def _pick_root(tree: LexborHTMLParser) -> LexborNode | None:
    for selector in ("main", '[role="main"]'):
        node = tree.css_first(selector)
        if node is not None and len(node.text(strip=True)) > 200:
            return node
    articles = tree.css("article")
    if articles:
        best = max(articles, key=lambda n: len(n.text(strip=True)))
        if len(best.text(strip=True)) > 200:
            return best
    return tree.body


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    text = _INLINE_WS.sub(" ", text)
    lines = (line.strip() for line in text.split("\n"))
    return _MULTI_NL.sub("\n", "\n".join(line for line in lines if line)).strip()


def extract(html: str) -> Page:
    """Extract the title and the main readable text of an HTML document.

    Paragraph and other block boundaries become newlines, which :mod:`ollamasearch.rank`
    uses to split passages.
    """
    tree = LexborHTMLParser(html)
    title = _title(tree)

    for node in tree.css(", ".join(DROP_TAGS + DROP_SELECTORS)):
        node.remove()
    # Class/id heuristics, but never strip the structural containers themselves.
    for node in tree.css("[class], [id]"):
        if node.tag not in {"html", "body", "main", "article"} and _is_boilerplate(node):
            node.remove()

    root = _pick_root(tree)
    if root is None:
        return Page(title=title, text="")
    for node in root.css(", ".join(BLOCK_TAGS)):
        node.insert_before("\n")
        node.insert_after("\n")
    text = normalize_whitespace(root.text(separator=""))
    return Page(title=title, text=text)


def looks_like_html(body: str) -> bool:
    head = body[:2048].lower()
    return "<html" in head or "<!doctype html" in head or "<body" in head or "<p>" in head

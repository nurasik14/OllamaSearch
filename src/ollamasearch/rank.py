"""Passage splitting and BM25 ranking.

Small models answer far better from a few relevant passages than from whole pages,
so this module selects the best passages for a query within a character budget.
Pure Python, no extra dependencies; tokenisation is unicode-aware (en/ru/kk).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
STEM_LEN = 6

STOPWORDS = frozenset(
    """
    a an the and or of to in on at by for with from as is are was were be been being it its
    this that these those what which who whom whose when where why how did does do has have
    had not no but if than then so such can could will would shall should may might must
    into about over under after before between during i you he she we they me him her us
    them my your our their there here also just only very more most much many any some all
    each other s t и в во не что он на я с со как а то все она так его но да ты к у же вы за
    бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если
    уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может
    они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе
    под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой
    тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть
    после над больше тот через эти нас про всего них какая много разве три эту моя впрочем
    хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
    между сколько каком году какое және мен бен пен де та те ма ме ба бе па пе бұл сол осы
    ол олар неше қай қандай қашан кім қалай үшін деп болып еді жылы
    """.split()
)


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Casefold, split into ``\\w+`` tokens, drop stopwords, crude prefix stemming."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.casefold().replace("ё", "е")):
        if tok in STOPWORDS:
            continue
        if stem and len(tok) > STEM_LEN and not tok.isdigit():
            tok = tok[:STEM_LEN]
        out.append(tok)
    return out


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tfs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: Sequence[str], index: int) -> float:
        tf = self.tfs[index]
        dl = self.lengths[index]
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        s = 0.0
        for t in set(query):
            f = tf.get(t)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def scores(self, query: Sequence[str]) -> list[float]:
        return [self.score(query, i) for i in range(len(self.tfs))]

    def top(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: (-x[1], x[0]))
        return [(i, s) for i, s in ranked[:k] if s > 0]


def split_passages(text: str, target: int = 500) -> list[str]:
    """Split text into passages of roughly ``target`` chars along paragraph/sentence lines."""
    pieces: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target * 1.5:
            pieces.append(para)
            continue
        buf = ""
        for sent in _SENTENCE_RE.split(para):
            if buf and len(buf) + len(sent) + 1 > target:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
            while len(buf) > target * 2:  # no sentence boundaries: hard wrap on spaces
                cut = buf.rfind(" ", 0, target)
                cut = cut if cut > 0 else target
                pieces.append(buf[:cut].strip())
                buf = buf[cut:].strip()
        if buf:
            pieces.append(buf)

    passages: list[str] = []
    buf = ""
    for piece in pieces:  # merge short neighbours (list items, headings) into one passage
        if buf and len(buf) + len(piece) + 1 > target:
            passages.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n{piece}" if buf else piece
    if buf:
        passages.append(buf)
    return passages


@dataclass(frozen=True)
class Document:
    """A fetched source to rank passages from. ``source_id`` is the caller's handle."""

    source_id: int
    title: str
    text: str


@dataclass(frozen=True)
class Passage:
    source_id: int
    position: int  # index of the passage within its document
    text: str
    score: float


def select_passages(
    query: str,
    documents: Iterable[Document],
    *,
    budget_chars: int = 6000,
    per_source_chars: int = 1500,
    passage_chars: int = 500,
) -> list[Passage]:
    """Return the best passages across documents, in (source, position) order.

    Every document with any query overlap gets its best passage first (diversity),
    then remaining budget is filled greedily by score. Titles are prepended to
    every passage for scoring so passages that never repeat the entity name still match.
    """
    candidates: list[tuple[int, int, str, list[str]]] = []
    for doc in documents:
        title_tokens = tokenize(doc.title)
        for pos, text in enumerate(split_passages(doc.text, passage_chars)):
            candidates.append((doc.source_id, pos, text, title_tokens + tokenize(text)))
    if not candidates:
        return []

    bm25 = BM25([c[3] for c in candidates])
    q = tokenize(query)
    scored = [
        Passage(source_id=c[0], position=c[1], text=c[2], score=bm25.score(q, i))
        for i, c in enumerate(candidates)
    ]
    by_score = sorted(scored, key=lambda p: (-p.score, p.source_id, p.position))

    chosen: dict[tuple[int, int], Passage] = {}
    used_total = 0
    used_per_source: Counter[int] = Counter()
    source_cap = max(per_source_chars, passage_chars)

    def try_add(p: Passage) -> None:
        nonlocal used_total
        size = len(p.text)
        key = (p.source_id, p.position)
        if key in chosen or used_total + size > budget_chars:
            return
        if used_per_source[p.source_id] + size > source_cap:
            return
        chosen[key] = p
        used_total += size
        used_per_source[p.source_id] += size

    seen_sources: set[int] = set()
    for p in by_score:  # pass 1: best passage of each source
        if p.source_id not in seen_sources and (p.score > 0 or not q):
            seen_sources.add(p.source_id)
            try_add(p)
    for p in by_score:  # pass 2: fill by score
        if p.score > 0 or not q:
            try_add(p)
    if not chosen:  # no lexical overlap at all: fall back to document openings
        for p in sorted(scored, key=lambda p: (p.position, p.source_id)):
            try_add(p)

    return sorted(chosen.values(), key=lambda p: (p.source_id, p.position))

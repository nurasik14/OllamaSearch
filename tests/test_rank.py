from __future__ import annotations

from ollamasearch.rank import BM25, Document, select_passages, split_passages, tokenize


def test_tokenize_casefold_stopwords_and_stemming() -> None:
    assert tokenize("The Population of Velmora") == ["popula", "velmor"]
    assert tokenize("population populated") == ["popula", "popula"]
    assert tokenize("1250 people", stem=False) == ["1250", "people"]


def test_tokenize_cyrillic_and_kazakh() -> None:
    assert tokenize("Какова ГЛУБИНА озера Тенгизколь?") == ["какова", "глубин", "озера", "тенгиз"]
    assert tokenize("Ауылдың халқы қанша?") == ["ауылды", "халқы", "қанша"]
    assert tokenize("Ёлка") == tokenize("елка")


def test_tokenize_keeps_long_numbers_whole() -> None:
    assert tokenize("13050250 items") == ["13050250", "items"]


def test_bm25_ranks_relevant_doc_first() -> None:
    docs = [tokenize(t) for t in ("cats and dogs", "the bridge was built in 2004", "bridge bridge")]
    bm = BM25(docs)
    top = bm.top(tokenize("when was the bridge built"), 2)
    assert top[0][0] == 1
    assert all(score > 0 for _, score in top)


def test_bm25_no_match_and_empty() -> None:
    bm = BM25([tokenize("alpha beta")])
    assert bm.top(tokenize("gamma"), 3) == []
    assert BM25([]).top(["x"], 3) == []


def test_split_passages_merges_short_lines() -> None:
    text = "\n".join(f"line {i}" for i in range(10))
    passages = split_passages(text, target=40)
    assert all(len(p) <= 40 for p in passages)
    assert "".join(passages).replace("\n", "") == text.replace("\n", "")


def test_split_passages_splits_long_paragraph_on_sentences() -> None:
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    passages = split_passages(para, target=200)
    assert len(passages) > 5
    assert all(len(p) <= 400 for p in passages)
    assert passages[0].startswith("Sentence number 0")


def test_split_passages_hard_wraps_without_punctuation() -> None:
    passages = split_passages("word " * 1000, target=100)
    assert all(len(p) <= 200 for p in passages)


def test_select_passages_picks_relevant_and_respects_budget() -> None:
    filler = "\n".join(f"Unrelated filler paragraph {i} about weather and food." for i in range(30))
    docs = [
        Document(0, "Velmora town", filler + "\nVelmora has a population of 13,050 people."),
        Document(1, "Other", filler),
    ]
    chosen = select_passages("population of Velmora", docs, budget_chars=300, per_source_chars=300)
    assert sum(len(p.text) for p in chosen) <= 300
    assert any("13,050" in p.text for p in chosen)


def test_select_passages_diversity_and_per_source_cap() -> None:
    docs = [Document(i, f"doc {i}", "\n".join(["bridge facts here"] * 20)) for i in range(3)]
    chosen = select_passages(
        "bridge", docs, budget_chars=10_000, per_source_chars=100, passage_chars=50
    )
    assert {p.source_id for p in chosen} == {0, 1, 2}
    for sid in range(3):
        assert sum(len(p.text) for p in chosen if p.source_id == sid) <= 100


def test_select_passages_title_helps_matching() -> None:
    docs = [Document(0, "Orlek Bridge", "It opened in 2004."), Document(1, "Misc", "Nothing.")]
    chosen = select_passages("Orlek Bridge opening", docs, budget_chars=100)
    assert chosen[0].source_id == 0


def test_select_passages_fallback_without_overlap() -> None:
    docs = [Document(0, "A", "alpha text"), Document(1, "B", "beta text")]
    chosen = select_passages("zzz", docs, budget_chars=1000)
    assert {p.source_id for p in chosen} == {0, 1}


def test_select_passages_output_order() -> None:
    docs = [Document(1, "x", "one\ntwo bridge"), Document(0, "y", "bridge three")]
    chosen = select_passages("bridge", docs, budget_chars=1000, passage_chars=5)
    assert [(p.source_id, p.position) for p in chosen] == sorted(
        (p.source_id, p.position) for p in chosen
    )


def test_select_passages_empty() -> None:
    assert select_passages("q", []) == []

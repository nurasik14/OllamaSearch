from __future__ import annotations

import math

import pytest

from ollamasearch.bench.dataset import Question
from ollamasearch.bench.scoring import (
    is_abstention,
    match_alias,
    normalize,
    percentile,
    score_answer,
    summarize_rows,
)


def q(**kw: object) -> Question:
    base: dict[str, object] = {
        "id": "x",
        "question": "?",
        "lang": "en",
        "answers": ("13050",),
        "category": "single-hop",
        "gold_urls": ("http://corpus.local/a.html",),
    }
    base.update(kw)
    return Question(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("1,250", "1250"),
        ("1 250", "1250"),
        ("1 250", "1250"),
        ("13 050 человек", "13050 человек"),
        ("1,250,000", "1250000"),
        ("3,5 км", "3.5 км"),
        ("4.7 million.", "4.7 million"),
        ("Hello, World!", "hello world"),
        ("Ёлка", "елка"),
        ("ҚАЗАҚСТАН", "қазақстан"),
        ("“Astana”", "astana"),
    ],
)
def test_normalize(raw: str, norm: str) -> None:
    assert normalize(raw) == norm


def test_years_in_lists_are_not_merged() -> None:
    assert normalize("1999, 2000") == "1999 2000"


def test_match_alias_numbers_and_tokens() -> None:
    assert match_alias("The population is 13,050 people.", ["13050"]) == "13050"
    assert match_alias("About 113,050 people", ["13050"]) is None
    assert match_alias("It is Astana.", ["Nur-Sultan", "Astana"]) == "Astana"
    assert match_alias("Nur-Sultan was the name", ["nur sultan"]) == "nur sultan"
    assert match_alias("Astanaville", ["Astana"]) is None


def test_match_alias_cyrillic_case() -> None:
    assert match_alias("Столица — АСТАНА.", ["Астана"]) == "Астана"
    assert match_alias("Мұхтар Әуезов жазған", ["мұхтар әуезов"]) is not None


def test_match_alias_empty_alias_ignored() -> None:
    assert match_alias("anything", ["", "  "]) is None


@pytest.mark.parametrize(
    "text",
    [
        "I could not find this information.",
        "I couldn't find that in the sources.",
        "The sources do not contain the answer.",
        "Информация не найдена.",
        "К сожалению, в источниках нет информации об этом.",
        "Бұл туралы ақпарат табылмады.",
        "Мәлімет жоқ.",
    ],
)
def test_abstention_detected(text: str) -> None:
    assert is_abstention(text)


@pytest.mark.parametrize("text", ["The population is 13,050.", "Астана", "Ақсуат ауылы", ""])
def test_abstention_not_detected(text: str) -> None:
    assert not is_abstention(text)


def test_score_answerable_correct_with_citation() -> None:
    s = score_answer(
        q(),
        "It has 13 050 residents [1].",
        cited_urls=["http://corpus.local/a.html/"],
        context_urls=["http://corpus.local/a.html"],
    )
    assert s.correct and s.citation_hit and s.gold_in_context and not s.abstained


def test_score_wrong_answer_and_no_citation() -> None:
    s = score_answer(q(), "It has 12,400 residents.", context_urls=["http://other"])
    assert not s.correct
    assert s.citation_hit is False and s.gold_in_context is False


def test_score_stale_answer_on_conflicting() -> None:
    question = q(category="conflicting", outdated=("12400",))
    assert score_answer(question, "12,400 people").stale
    assert not score_answer(question, "Now 13,050, up from 12,400").stale


def test_score_unanswerable() -> None:
    question = q(category="unanswerable", answers=(), gold_urls=())
    assert score_answer(question, "I could not find this information.").correct
    s = score_answer(question, "The phone number is 555-0100.")
    assert not s.correct and s.citation_hit is None


def test_closed_book_has_no_citation_metric() -> None:
    s = score_answer(q(), "13050", mode="closed")
    assert s.correct and s.citation_hit is None and s.gold_in_context is None


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == pytest.approx(9.55)
    assert math.isnan(percentile([], 50))


def _row(model: str, mode: str, cat: str, correct: bool, **answer: object) -> dict[str, object]:
    ans: dict[str, object] = {
        "timings": {"total_s": 2.0},
        "tokens": {"eval_tokens": 20, "eval_s": 1.0, "prompt_tokens": 100},
        "tool_calls": [],
        "error": None,
        "steps": 1,
    }
    ans.update(answer)
    return {
        "model": model,
        "mode": mode,
        "category": cat,
        "lang": "en",
        "score": {
            "correct": correct,
            "abstained": False,
            "stale": False,
            "citation_hit": correct if mode != "closed" else None,
            "gold_in_context": True if mode != "closed" else None,
        },
        "answer": ans,
    }


def test_summarize_rows_groups_and_metrics() -> None:
    calls = [
        {"name": "web_search", "status": "ok", "from_text": False},
        {"name": "nope", "status": "invalid", "from_text": False},
    ]
    rows = [
        _row("m", "rag", "single-hop", True),
        _row("m", "rag", "unanswerable", False),
        _row("m", "agent", "single-hop", True, tool_calls=calls, steps=2),
        _row("m", "closed", "single-hop", False),
    ]
    groups = {(g["model"], g["mode"]): g for g in summarize_rows(rows)["groups"]}
    rag = groups[("m", "rag")]
    assert rag["accuracy"] == 0.5
    assert rag["answerable_accuracy"] == 1.0
    assert rag["unanswerable_accuracy"] == 0.0
    assert rag["tokens_per_s"] == 20
    assert rag["latency_p50_s"] == 2.0
    agent = groups[("m", "agent")]["agent"]
    assert agent["tool_calls_per_question"] == 2
    assert agent["invalid_call_rate"] == 0.5
    assert agent["searched_rate"] == 1.0
    assert groups[("m", "closed")]["citation_hit_rate"] is None

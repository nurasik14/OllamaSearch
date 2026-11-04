from __future__ import annotations

import math
import random
from typing import Any

import pytest

from ollamasearch.bench.external import Doc, ExternalItem
from ollamasearch.bench.external.scoring import (
    doc_name,
    doc_url,
    flags_factual_error,
    gold_doc_recall,
    hotpot_em,
    hotpot_f1,
    hotpot_normalize,
    is_rejection,
    rgb_checkanswer,
    span_scores,
    stem_contains,
    strip_citations,
    unicode_normalize,
)
from ollamasearch.bench.external.tasks.hotpotqa import score_text
from ollamasearch.bench.external.tasks.rgb import answer_correct, process
from ollamasearch.bench.external.tasks.simpleqa import (
    aliases,
    grade_metrics,
    parse_grade,
    string_grade,
)
from ollamasearch.pipelines import Answer

# --- HotpotQA: official normalisation / EM / F1 ------------------------------------------


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("The Quick, Brown Fox!", "quick brown fox"),
        ("an Apple a day", "apple day"),
        ("  Chief   of\tProtocol. ", "chief of protocol"),
        ("U.S.A.", "usa"),
        ("theatre", "theatre"),  # articles only as whole words
    ],
)
def test_hotpot_normalize(raw: str, norm: str) -> None:
    assert hotpot_normalize(raw) == norm


def test_hotpot_em_and_f1_match_official() -> None:
    assert hotpot_em("The Animorphs", "animorphs")
    assert not hotpot_em("Animorphs series", "Animorphs")
    f1, precision, recall = hotpot_f1("Chief of Protocol of the US", "Chief of Protocol")
    assert (precision, recall) == (3 / 5, 1.0)
    assert f1 == pytest.approx(2 * 0.6 / 1.6)
    # yes/no/noanswer: any mismatch scores zero even with token overlap
    assert hotpot_f1("yes", "no") == (0.0, 0.0, 0.0)
    assert hotpot_f1("no", "no way") == (0.0, 0.0, 0.0)
    assert hotpot_f1("yes", "yes")[0] == 1.0
    assert hotpot_f1("", "Paris") == (0.0, 0.0, 0.0)


def test_hotpot_sentence_answers_count_by_containment() -> None:
    s = score_text("She served as Chief of Protocol of the United States [2].", "Chief of Protocol")
    assert s["correct"] and not s["em"] and 0 < s["f1"] < 1
    s = score_text("Chief of Protocol [1]", "Chief of Protocol")
    assert s["correct"] and s["em"] and s["f1"] == 1.0  # citation marker stripped
    assert not score_text("He was born in 1924 [1].", "1942")["correct"]
    assert not score_text("Animorphsy", "Animorphs")["correct"]  # whole tokens only


@pytest.mark.parametrize(
    ("text", "gold", "correct"),
    [
        ("Yes, both are American [1][2].", "yes", True),
        ("No. Derrickson is American, Wood was British.", "no", True),
        ("No, they were not.", "yes", False),
        ("Yes — there is no doubt.", "yes", True),  # first yes/no token decides
        ("I could not find this information.", "yes", False),
    ],
)
def test_hotpot_yes_no(text: str, gold: str, correct: bool) -> None:
    assert score_text(text, gold)["correct"] is correct


# --- Kazakh-safe normalisation -------------------------------------------------------------


def test_unicode_normalize_kazakh() -> None:
    assert unicode_normalize("«Ұлтабарға»!") == "ұлтабарға"
    assert unicode_normalize("ＡＢＣ — the Апа") == "abc the апа"  # NFKC, no article removal
    assert unicode_normalize("Хан Тәңірі, 7010 м.") == "хан тәңірі 7010 м"


def test_span_scores_kazakh() -> None:
    s = span_scores("Бауыр өзегі ұлтабарға ашылады [1].", ["Ұлтабарға", "ултабарга"])
    assert s["matched"] == "Ұлтабарға" and not s["em"] and s["f1"] > 0 and s["stem_match"]
    s = span_scores("Ұлтабарға [2]", ["Ұлтабарға"])
    assert s["em"] and s["f1"] == 1.0
    s = span_scores("Ұлтабар.", ["Ұлтабарға"])  # different case ending
    assert s["matched"] is None and s["stem_match"]
    assert not span_scores("Қарын.", ["Ұлтабарға"])["stem_match"]


def test_stem_contains_needs_order() -> None:
    assert stem_contains("хан тәңірі шыңы", "хан тәңірінің")
    assert not stem_contains("тәңірі хан", "хан тәңірі")
    assert not stem_contains("anything", "")


def test_strip_citations() -> None:
    assert strip_citations("Paris [1][2, 3]. Also [10]") == "Paris . Also"


# --- RGB -------------------------------------------------------------------------------------


def test_rgb_checkanswer_like_evalue_py() -> None:
    # every string element is required
    assert rgb_checkanswer(
        "Ada Brightwater and Milo Quist [1]", ["Ada Brightwater", "Milo Quist"]
    ) == [1, 1]
    assert rgb_checkanswer("Ada Brightwater", ["Ada Brightwater", "Milo Quist"]) == [1, 0]
    # a list element is a set of aliases (any one)
    assert rgb_checkanswer("It premiered on Jan 2, 2022.", [["January 2 2022", "Jan 2, 2022"]]) == [
        1
    ]
    assert rgb_checkanswer("NORWAY", "Norway") == [1]  # case-insensitive substring


def test_rgb_answer_correct_and_rejection() -> None:
    gold = [["936,172", "936172"]]
    assert answer_correct("Voltra delivered 936,172 vehicles [1].", gold)
    assert not answer_correct("There is insufficient information. Maybe 936,172?", gold)
    assert not answer_correct("I could not find this information.", gold)
    assert not answer_correct("Ada Brightwater won.", ["Ada Brightwater", "Milo Quist"])


@pytest.mark.parametrize(
    ("text", "rejected"),
    [
        (
            "I can not answer the question because of the insufficient information in documents.",
            True,
        ),
        ("I could not find this information.", True),
        ("The documents do not contain the premiere date.", True),
        ("It premieres on September 21 [1].", False),
    ],
)
def test_is_rejection(text: str, rejected: bool) -> None:
    assert is_rejection(text) is rejected


@pytest.mark.parametrize(
    ("text", "flag"),
    [
        ("There are factual errors in the provided documents. The answer is Tampa.", True),
        ("The documents are incorrect: it was played in Tampa, Florida.", True),
        ("The provided information contains an error; Norway won.", True),
        ("The sources contradict what I know.", True),
        ("It was played in Glendale, Arizona [1].", False),
        ("Norway won the most medals [1].", False),
    ],
)
def test_flags_factual_error(text: str, flag: bool) -> None:
    assert flags_factual_error(text) is flag


def _reference_processdata(
    instance: dict[str, Any], noise_rate: float, passage_num: int, filename: str
) -> list[str]:
    """RGB evalue.py processdata (en / en_fact, correct_rate 0), uses the global RNG."""
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    if "_fact" in filename:
        correct_num = 0
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance["positive"])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [instance["positive_wrong"][i] for i in selected]
        if neg_num > 0:
            docs += instance["negative"][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance["negative"]):
                neg_num = len(instance["negative"])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance["positive"]):
                pos_num = len(instance["positive"])
                neg_num = passage_num - pos_num
        docs = instance["positive"][:pos_num] + instance["negative"][:neg_num]
    random.shuffle(docs)
    return docs


def _instance(n_pos: int, n_neg: int) -> dict[str, Any]:
    return {
        "positive": [f"pos {i}" for i in range(n_pos)],
        "positive_wrong": [f"wrong {i}" for i in range(n_pos)],
        "negative": [f"neg {i}" for i in range(n_neg)],
    }


@pytest.mark.parametrize("noise", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
@pytest.mark.parametrize(("n_pos", "n_neg"), [(5, 5), (2, 10), (10, 1), (7, 3)])
def test_rgb_process_matches_reference(noise: float, n_pos: int, n_neg: int) -> None:
    inst = _instance(n_pos, n_neg)
    random.seed(2333)
    assert process(inst, noise) == _reference_processdata(inst, noise, 5, "en")
    if noise < 1:
        random.seed(2333)
        assert process(inst, noise, fact=True) == _reference_processdata(inst, noise, 5, "en_fact")


def test_rgb_noise_rates_mix() -> None:
    inst = _instance(10, 10)
    assert sum(d.startswith("neg") for d in process(inst, 0.4)) == 2
    assert sum(d.startswith("neg") for d in process(inst, 0.8)) == 4
    assert all(d.startswith("neg") for d in process(inst, 1.0))
    assert all(d.startswith("wrong") for d in process(inst, 0.0, fact=True))


# --- SimpleQA ----------------------------------------------------------------------------------


def test_simpleqa_aliases() -> None:
    assert aliases("Season 7, Episode 20: White Wedding") == (
        "Season 7, Episode 20: White Wedding",
        "Season 7, Episode 20",
        "White Wedding",
    )
    assert aliases("San Francisco (California)") == ("San Francisco (California)", "San Francisco")


@pytest.mark.parametrize(
    ("text", "gold", "grade"),
    [
        ("The award went to Michio Sugeno in 2010 [1].", "Michio Sugeno", "CORRECT"),
        ("It was founded in 1586.", "1586", "CORRECT"),
        ("They married in White Wedding [3].", "Season 7, Episode 20: White Wedding", "CORRECT"),
        ("San Francisco, California.", "San Francisco (California)", "CORRECT"),
        ("I could not find this information.", "1586", "NOT_ATTEMPTED"),
        ("", "1586", "NOT_ATTEMPTED"),
        ("It was founded in 1856.", "1586", "INCORRECT"),
    ],
)
def test_simpleqa_string_grade(text: str, gold: str, grade: str) -> None:
    assert string_grade(text, gold) == grade


@pytest.mark.parametrize(
    ("response", "grade"),
    [
        ("A", "CORRECT"),
        ("B", "INCORRECT"),
        ("C", "NOT_ATTEMPTED"),
        (" B\n", "INCORRECT"),
        ("the grade is: A", "CORRECT"),
        ("no idea", "NOT_ATTEMPTED"),
    ],
)
def test_simpleqa_parse_grade(response: str, grade: str) -> None:
    assert parse_grade(response) == grade


def test_simpleqa_grade_metrics_like_official() -> None:
    m = grade_metrics(["CORRECT", "CORRECT", "INCORRECT", "NOT_ATTEMPTED"])
    assert m["is_correct"] == 0.5 and m["is_not_attempted"] == 0.25
    assert m["is_given_attempted"] == 0.75
    assert m["accuracy_given_attempted"] == pytest.approx(2 / 3)
    given, correct = 2 / 3, 0.5
    assert m["f1"] == pytest.approx(2 * given * correct / (given + correct))
    assert grade_metrics(["NOT_ATTEMPTED"])["f1"] == 0.0
    assert grade_metrics([]) == {}


# --- per-item corpus helpers ---------------------------------------------------------------


def test_doc_names_and_gold_recall() -> None:
    item = ExternalItem(
        id="x",
        question="q",
        answers=("a",),
        category="c",
        docs=(Doc("Scott Derrickson", "t"), Doc("Ұлтабар", "t"), Doc("Ed Wood (film)", "t")),
        extra={"gold_docs": (0, 2)},
    )
    assert doc_name(0, item.docs[0]) == "00-scott-derrickson.html"
    assert doc_name(1, item.docs[1]) == "01-doc.html"
    assert doc_url(item, 2) == "http://corpus.local/02-ed-wood-film.html"
    answer = Answer(question="q", text="", mode="rag", model="m", context_urls=[doc_url(item, 0)])
    assert gold_doc_recall(item, answer) == 0.5
    answer.mode = "closed"
    assert gold_doc_recall(item, answer) is None

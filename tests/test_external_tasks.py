from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ollamasearch.bench.external import available_tasks, get_task
from ollamasearch.bench.external.runner import ItemCorpora, item_pages, render_doc
from ollamasearch.bench.external.scoring import doc_url
from ollamasearch.bench.external.tasks import _common
from ollamasearch.bench.external.tasks import hotpotqa as hotpot_mod
from ollamasearch.bench.external.tasks._common import (
    DatasetUnavailableError,
    fetch_file,
    sample_indices,
)
from ollamasearch.bench.external.tasks.kazqad import NAME as KAZQAD_NAME
from ollamasearch.bench.external.tasks.kazqad import question_id
from ollamasearch.config import Settings
from ollamasearch.extract import extract
from ollamasearch.pipelines import Answer

from .conftest import DATA

EXTERNAL = DATA / "external"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaders must use the fixture files; any download attempt fails the test."""

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tests must not download datasets")

    monkeypatch.setattr(_common, "_new_client", boom)


@pytest.fixture
def hotpot_dir(tmp_path: Path) -> Path:
    """A data dir holding a tiny HotpotQA Parquet file built from rows.json."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = json.loads((EXTERNAL / "hotpotqa" / "rows.json").read_text(encoding="utf-8"))
    dest = tmp_path / hotpot_mod.FILE
    dest.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), dest)
    return tmp_path


def test_registry_has_all_tasks() -> None:
    assert available_tasks() == [
        "hotpotqa",
        "kazqad",
        "rgb-counterfactual",
        "rgb-negative",
        "rgb-noise",
        "simpleqa",
    ]
    with pytest.raises(ValueError, match="unknown external task"):
        get_task("nope")


def test_sample_indices_deterministic_and_prefix_stable() -> None:
    a = sample_indices(100, 10, 50, seed=1)
    assert a == sample_indices(100, 10, 50, seed=1)
    assert a != sample_indices(100, 10, 50, seed=2)
    assert sample_indices(100, 3, 50, seed=1) == a[:3]
    assert len(sample_indices(100, None, 50, seed=1)) == 50
    assert sample_indices(5, 0, 50, seed=1) == [0, 1, 2, 3, 4]
    assert sorted(sample_indices(5, 99, 50, seed=1)) == [0, 1, 2, 3, 4]


def test_fetch_file_downloads_once(tmp_path: Path) -> None:
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        if request.url.path == "/missing":
            return httpx.Response(404)
        return httpx.Response(200, content=b"hello")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dest = tmp_path / "d" / "f.txt"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    f = fetch_file("http://x/f.txt", dest, expected_sha256=sha, client=client)
    assert dest.read_bytes() == b"hello" and f.bytes == 5 and f.verified is True
    assert fetch_file("http://x/f.txt", dest, expected_sha256="0" * 64).verified is False
    assert len(hits) == 1  # cached
    with pytest.raises(DatasetUnavailableError, match="404"):
        fetch_file("http://x/missing", tmp_path / "m.txt", client=client)
    assert not list(tmp_path.glob("*.part"))


def test_hotpotqa_loader(hotpot_dir: Path) -> None:
    task = get_task("hotpotqa")
    items = task.load(hotpot_dir, 0, 42)
    assert items[0].id == "5a8b57f25542995d1e6f1371"
    first = items[0]
    assert first.category == "comparison" and first.answers == ("yes",)
    assert len(first.docs) == 4 and first.docs[1].title == "Scott Derrickson"
    assert (
        first.docs[1].text.startswith("Scott Derrickson (born") and "He lives" in first.docs[1].text
    )
    assert first.extra["gold_docs"] == (1, 3)
    sampled = task.load(hotpot_dir, 2, 7)
    assert [i.id for i in sampled] == [i.id for i in task.load(hotpot_dir, 2, 7)]
    assert len(sampled) == 2
    assert task.files[0].sha256 and task.files[0].verified is False  # fixture, not upstream


def test_hotpotqa_needs_pyarrow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_pyarrow(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(hotpot_mod.importlib, "import_module", no_pyarrow)
    with pytest.raises(RuntimeError, match="uv sync --extra datasets"):
        get_task("hotpotqa").load(tmp_path, 1, 42)


def test_hotpotqa_score_and_sp_recall(hotpot_dir: Path) -> None:
    task = get_task("hotpotqa")
    item = task.load(hotpot_dir, 0, 42)[1]  # Shirley Temple, bridge
    ctx = [doc_url(item, i) for i in item.extra["gold_docs"]]
    answer = Answer(
        question=item.question,
        text="She served as Chief of Protocol of the United States [1].",
        mode="rag",
        model="m",
        context_urls=ctx[:1],
    )
    s = task.score(item, answer)
    assert s["correct"] and not s["em"] and s["sp_recall"] == 0.5 and s["sp_both"] is False
    rows = [{"score": s}, {"score": {**s, "em": True, "sp_recall": 1.0, "sp_both": True}}]
    m = task.summarize(rows)
    assert m["em"] == 0.5 and m["sp_recall"] == 0.75 and m["sp_both"] == 0.5


def test_rgb_noise_loader() -> None:
    task = get_task("rgb-noise")
    items = task.load(EXTERNAL, 0, 42)
    assert len(items) == 4 * 3
    cats = [i.category for i in items]
    assert cats[:4] == ["noise-0.0"] * 4 and cats[-1] == "noise-0.8"
    by_rate = {c: [i.extra["rgb_id"] for i in items if i.category == c] for c in set(cats)}
    assert by_rate["noise-0.0"] == by_rate["noise-0.4"] == by_rate["noise-0.8"]
    voltra = next(i for i in items if i.id == "2@noise-0.8")
    # 3 negatives only: RGB tops up with positives (neg 3 + pos 2)
    assert len(voltra.docs) == 5 and len(voltra.extra["gold_docs"]) == 2
    assert all(len(i.docs) == 5 for i in items)
    sampled = task.load(EXTERNAL, 2, 42)
    assert len(sampled) == 6
    assert [i.id for i in sampled] == [i.id for i in task.load(EXTERNAL, 2, 42)]


def test_rgb_negative_and_counterfactual_loaders() -> None:
    raw = [json.loads(line) for line in (EXTERNAL / "rgb" / "en_refine.json").open()]
    neg = get_task("rgb-negative").load(EXTERNAL, 0, 42)
    assert len(neg) == 4 and all(not i.answerable and i.answers == () for i in neg)
    for item in neg:  # only negative passages, at most 5
        inst = next(r for r in raw if str(r["id"]) == item.id)
        assert {d.text for d in item.docs} <= set(inst["negative"])
        assert len(item.docs) == min(5, len(inst["negative"]))
    fact = get_task("rgb-counterfactual").load(EXTERNAL, 0, 42)
    assert len(fact) == 3 and fact[0].extra["fake"] == "Glendale, Arizona"
    assert all("Glendale" in d.text for d in fact[0].docs) and len(fact[0].docs) == 3


def test_rgb_scores() -> None:
    noise = get_task("rgb-noise")
    item = next(i for i in noise.load(EXTERNAL, 0, 42) if i.id == "1@noise-0.0")
    ans = Answer(item.question, "Ada Brightwater and Milo Quist [1].", "rag", "m")
    assert noise.score(item, ans)["correct"]
    ans.text = "Ada Brightwater [1]."
    assert not noise.score(item, ans)["correct"]

    neg = get_task("rgb-negative")
    item = neg.load(EXTERNAL, 0, 42)[0]
    s = neg.score(item, Answer(item.question, "I could not find this information.", "rag", "m"))
    assert s["correct"] and s["rejected"] and not s["exact_phrase"]
    s = neg.score(item, Answer(item.question, "It premieres on January 2, 2022.", "rag", "m"))
    assert not s["correct"] and s["answered_anyway"]
    judged = neg.apply_judge(s, "No, the question is not addressed by the documents.")
    assert judged["correct"] and judged["string_rejected"] is False

    fact = get_task("rgb-counterfactual")
    item = fact.load(EXTERNAL, 0, 42)[0]
    good = "The documents contain factual errors: Super Bowl LV was in Tampa, Florida."
    s = fact.score(item, Answer(item.question, good, "rag", "m"))
    assert s["correct"] and s["detected"] and s["true_answer"] and not s["misled"]
    s = fact.score(item, Answer(item.question, "It was in Glendale, Arizona [1].", "rag", "m"))
    assert not s["correct"] and s["misled"] and not s["detected"]
    assert fact.apply_judge(s, "Yes, the model has identified the factual errors.")["detected"]
    rows = [
        {"score": fact.score(item, Answer(item.question, t, "rag", "m"))}
        for t in (good, "There are factual errors in the documents.", "Glendale, Arizona.")
    ]
    m = fact.summarize(rows)
    assert m["error_detection_rate"] == pytest.approx(2 / 3)
    assert m["error_correction_rate"] == 0.5 and m["misled_rate"] == pytest.approx(1 / 3)


def test_kazqad_loader_groups_questions_and_adds_distractors() -> None:
    task = get_task("kazqad")
    items = task.load(EXTERNAL, 0, 42)
    assert task.files[0].path.name == f"{KAZQAD_NAME}.jsonl"
    assert [i.id for i in items] == ["bio0001", "bio0002", "bio0003", "geo0001", "geo0002"]
    liver = items[0]
    assert liver.lang == "kk" and liver.category == "biology"
    assert liver.answers == ("Ұлтабарға", "ултабарга")
    assert len(liver.extra["gold_docs"]) == 2
    titles = {d.title for d in liver.docs}
    assert {"Асқорыту мүшелері", "Ас қорыту жүйесі", "Жүрек", "Фотосинтез"} == titles
    assert all(
        "ұлтабар" not in d.text.casefold()
        for i, d in enumerate(liver.docs)
        if i not in liver.extra["gold_docs"]
    )
    assert items[3].category == "geography" and len(items[3].docs) == 2  # only other geo passage
    assert [i.id for i in task.load(EXTERNAL, 2, 5)] == [i.id for i in task.load(EXTERNAL, 2, 5)]
    assert question_id("kzh0123kzh#1_2") == "kzh0123"


def test_kazqad_prefers_huggingface_copy(tmp_path: Path) -> None:
    import gzip

    src = EXTERNAL / "kazqad" / f"{KAZQAD_NAME}.jsonl"
    dest = tmp_path / "kazqad" / f"{KAZQAD_NAME}.jsonl.gz"
    dest.parent.mkdir()
    dest.write_bytes(gzip.compress(src.read_bytes()))
    task = get_task("kazqad")
    assert len(task.load(tmp_path, 0, 42)) == 5
    assert task.files[0].url.startswith("https://huggingface.co/")


def test_kazqad_score() -> None:
    task = get_task("kazqad")
    item = task.load(EXTERNAL, 0, 42)[0]
    gold = doc_url(item, item.extra["gold_docs"][0])
    ans = Answer(
        item.question, "Бауыр өзегі ұлтабарға ашылады [1].", "rag", "m", context_urls=[gold]
    )
    s = task.score(item, ans)
    assert s["correct"] and s["gold_in_context"] and not s["abstained"]
    ans.text = "Ұлтабар."
    s = task.score(item, ans)
    assert not s["correct"] and s["stem_match"]


def test_simpleqa_loader() -> None:
    task = get_task("simpleqa")
    assert task.info.live
    items = task.load(EXTERNAL, 0, 42)
    assert [i.id for i in items] == [f"simpleqa-{n:04d}" for n in range(4)]
    assert items[0].category == "Science and technology" and items[0].docs == ()
    assert items[2].answers == ("Season 7, Episode 20: White Wedding",)
    assert items[1].question.startswith("What year was the municipality of Cómbita")


# --- per-item corpora ------------------------------------------------------------------------


def test_render_doc_round_trips_through_extract() -> None:
    from ollamasearch.bench.external import Doc

    page = extract(render_doc(Doc("Ed <Wood>", "First line & more.\nSecond line.")))
    assert page.title == "Ed <Wood>"
    assert "First line & more." in page.text and "Second line." in page.text


def test_item_corpora_search_fetch_and_isolation() -> None:
    items = get_task("rgb-noise").load(EXTERNAL, 0, 42)
    tallinn = next(i for i in items if i.id == "3@noise-0.4")
    voltra = next(i for i in items if i.id == "2@noise-0.0")
    corpora = ItemCorpora(Settings(cache_dir=None))
    try:
        backend = corpora.activate(tallinn)
        hits = backend.search("Which city hosted the 2023 Aurora Games?", 5)
        assert hits and all(h.url.startswith("http://corpus.local/") for h in hits)
        page = corpora.fetcher.fetch(hits[0].url)
        assert page.ok and "Aurora" in page.text
        assert corpora.activate(tallinn) is backend  # cached
        corpora.activate(voltra)
        # same file name, but now served from the active item's corpus
        other = corpora.fetcher.fetch(doc_url(tallinn, 0))
        assert other.ok and "Aurora" not in other.text and "Voltra" in other.text
        assert not corpora.fetcher.fetch("http://corpus.local/99-nope.html").ok
        assert len(item_pages(voltra)) == 5
    finally:
        corpora.close()


def test_kazqad_fixture_matches_real_schema() -> None:
    src = EXTERNAL / "kazqad" / f"{KAZQAD_NAME}.jsonl"
    for line in src.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {"id", "title", "context", "question", "answers"}
        assert set(row["answers"]) == {"text", "answer_start"}

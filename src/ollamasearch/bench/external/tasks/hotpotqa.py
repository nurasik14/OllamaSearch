"""HotpotQA, distractor setting, dev split (Yang et al., EMNLP 2018).

Each question comes with 10 Wikipedia paragraphs: the 2 gold paragraphs plus 8 distractors
retrieved by bigram TF-IDF. They become the item's private corpus, so rag/agent/multi must
find and combine the two gold paragraphs among look-alikes.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ....pipelines import Answer
from .. import Doc, ExternalItem, TaskInfo, register
from ..scoring import (
    contains_tokens,
    gold_doc_recall,
    hotpot_em,
    hotpot_f1,
    hotpot_normalize,
    rate,
    strip_citations,
)
from ._common import DatasetFile, fetch_file, sample_indices

log = logging.getLogger(__name__)

URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
FILE = "hotpotqa/distractor-validation-00000-of-00001.parquet"
COLUMNS = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
YES_NO = ("yes", "no")


def _parquet() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeError(
            "HotpotQA is published as Parquet and needs pyarrow: run `uv sync --extra datasets`"
        ) from exc


def item_from_row(row: Mapping[str, Any]) -> ExternalItem:
    titles: list[str] = list(row["context"]["title"])
    sentences: list[list[str]] = list(row["context"]["sentences"])
    docs = tuple(Doc(t, "".join(s).strip()) for t, s in zip(titles, sentences, strict=True))
    gold_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
    gold_docs = tuple(titles.index(t) for t in gold_titles if t in titles)
    return ExternalItem(
        id=str(row["id"]),
        question=str(row["question"]),
        answers=(str(row["answer"]),),
        category=str(row["type"]),
        lang="en",
        docs=docs,
        extra={"level": row.get("level"), "gold_titles": gold_titles, "gold_docs": gold_docs},
    )


def _yes_no(norm: str) -> str | None:
    """The first yes/no token of a normalised answer ("Yes, both are American" -> yes)."""
    return next((t for t in norm.split() if t in YES_NO), None)


def score_text(text: str, gold: str) -> dict[str, Any]:
    """Official EM/F1 on the citation-stripped answer, plus containment accuracy.

    The pipelines answer in sentences with citations ("He was born in 1948 [2]."), so the
    official EM almost never fires. ``accuracy`` counts the normalised gold answer appearing
    as a whole-token sequence in the normalised answer; for yes/no questions the first
    yes/no token of the answer must equal the gold one ("no" inside a sentence would match
    too easily otherwise).
    """
    clean = strip_citations(text)
    pred, truth = hotpot_normalize(clean), hotpot_normalize(gold)
    f1, precision, recall = hotpot_f1(clean, gold)
    correct = _yes_no(pred) == truth if truth in YES_NO else contains_tokens(pred, truth)
    return {
        "correct": correct,
        "em": hotpot_em(clean, gold),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


@register("hotpotqa")
class HotpotQA:
    info = TaskInfo(
        name="hotpotqa",
        title="HotpotQA (distractor, dev)",
        source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        license="CC BY-SA 4.0",
        citation="Yang et al., EMNLP 2018",
        default_sample=100,
    )
    protocol_note = (
        "Official: the model reads all 10 paragraphs and outputs a short answer span plus "
        "supporting sentences; EM/F1 via hotpot_evaluate_v1.py. Here the 10 paragraphs are a "
        "per-question corpus the pipelines search (BM25), fetch and rank themselves, so "
        "retrieval is part of the test. Answers are full sentences with citations: EM/F1 are "
        "computed on the citation-stripped answer and undercount; **accuracy** (normalised "
        "gold contained in the normalised answer; first yes/no token for yes/no questions) is "
        "the headline. Supporting-fact prediction is not scored; instead *gold paragraph "
        "recall* measures whether the two gold paragraphs reached the model."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "em": "EM",
        "f1": "F1",
        "sp_recall": "Gold paragraph recall",
        "sp_both": "Both gold paragraphs",
    }

    def __init__(self) -> None:
        self.files: list[DatasetFile] = []

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        pq = _parquet()
        f = fetch_file(URL, data_dir / FILE, expected_sha256=SHA256)
        self.files = [f]
        table = pq.read_table(f.path, columns=COLUMNS)
        indices = sample_indices(table.num_rows, sample, self.info.default_sample, seed)
        rows: list[dict[str, Any]] = table.take(indices).to_pylist()
        return [item_from_row(r) for r in rows]

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        out = score_text(answer.text, item.answers[0])
        recall = gold_doc_recall(item, answer)
        out["sp_recall"] = recall
        out["sp_both"] = None if recall is None else recall == 1.0
        return out

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scores = [r["score"] for r in rows]
        return {
            "em": rate(s["em"] for s in scores),
            "f1": rate(s["f1"] for s in scores),
            "sp_recall": rate(s.get("sp_recall") for s in scores),
            "sp_both": rate(s.get("sp_both") for s in scores),
        }

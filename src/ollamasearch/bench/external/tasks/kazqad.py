"""KazQAD reading comprehension, Kazakh test split (Yeshpanov et al., LREC-COLING 2024).

The file has one row per (question, relevant passage); a question can have several gold
passages with slightly different answer spans. Rows are grouped per question: the gold
passages plus a few seeded distractor passages from other questions of the same subject
form the question's corpus, so the pipelines must retrieve before they read.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ....pipelines import Answer
from ...scoring import is_abstention
from .. import Doc, ExternalItem, TaskInfo, register
from ..scoring import gold_doc_recall, rate, span_scores, unicode_normalize
from ._common import DatasetFile, fetch_file, sample_indices

NAME = "kazqad-reading-comprehension-v1.0-kk-test"
# The Hugging Face copy (issai/kazqad, gzipped) is gated behind a login; the authors' GitHub
# repository publishes the same v1.0 file openly, so that is what we download. A copy of the
# Hugging Face .jsonl.gz placed in <data_dir>/kazqad/ is used instead when present.
URL = f"https://raw.githubusercontent.com/IS2AI/KazQAD/main/data/reading-comprehension/{NAME}.jsonl"
SHA256 = "9bcf199095c745b14a8e4af704d488313720e51c7d476ebc9f4d97441fa29d36"
HF_URL = f"https://huggingface.co/datasets/issai/kazqad/resolve/main/kazqad/{NAME}.jsonl.gz"
DISTRACTORS = 6
SUBJECTS = {
    "bio": "biology",
    "geo": "geography",
    "kzh": "history-of-kazakhstan",
    "lit": "literature",
    "woh": "world-history",
}
_QID = re.compile(r"^([a-z]+)(\d+)")


def question_id(row_id: str) -> str:
    """``bio0001bio#85927_11_3`` -> ``bio0001`` (question id; the rest is the passage id)."""
    m = _QID.match(row_id)
    return m.group(0) if m else row_id


def read_rows(path: Path) -> list[dict[str, Any]]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def subject_of(qid: str) -> str:
    m = _QID.match(qid)
    return m.group(1) if m else ""


def group_questions(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Question id -> its rows (one per gold passage), in file order."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(question_id(str(row["id"])), []).append(row)
    return groups


def build_items(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    qids: Sequence[str],
    seed: int,
    distractors: int = DISTRACTORS,
) -> list[ExternalItem]:
    """Items for ``qids``: gold passages plus seeded same-subject distractor passages."""
    # subject -> unique passages (context text -> (row, normalised text, question id))
    pools: dict[str, dict[str, tuple[Mapping[str, Any], str, str]]] = {}
    for qid, group in groups.items():
        pool = pools.setdefault(subject_of(qid), {})
        for r in group:
            if r["context"] not in pool:
                pool[r["context"]] = (r, unicode_normalize(r["context"]), qid)

    items = []
    for qid in qids:
        group = groups[qid]
        answers = tuple(dict.fromkeys(a for r in group for a in r["answers"]["text"] if a))
        norm_answers = [n for n in (unicode_normalize(a) for a in answers) if n]
        gold_ctx = {r["context"] for r in group}
        subject = subject_of(qid)
        candidates = [
            row
            for ctx, (row, norm, owner) in pools.get(subject, {}).items()
            if owner != qid
            and ctx not in gold_ctx
            # a "distractor" must not contain this question's answer
            and not any(a in norm for a in norm_answers)
        ]
        rng = random.Random(f"{seed}:{qid}")
        docs_rows = [*group, *rng.sample(candidates, min(distractors, len(candidates)))]
        rng.shuffle(docs_rows)
        items.append(
            ExternalItem(
                id=qid,
                question=str(group[0]["question"]),
                answers=answers,
                category=SUBJECTS.get(subject, subject or "other"),
                lang="kk",
                docs=tuple(Doc(str(r["title"]), str(r["context"])) for r in docs_rows),
                extra={
                    "gold_docs": tuple(
                        i for i, r in enumerate(docs_rows) if r["context"] in gold_ctx
                    ),
                    "passage_ids": [r["id"] for r in group],
                },
            )
        )
    return items


@register("kazqad")
class KazQAD:
    info = TaskInfo(
        name="kazqad",
        title="KazQAD reading comprehension (kk, test)",
        source_url="https://github.com/IS2AI/KazQAD",
        license="CC BY-SA 4.0",
        citation="Yeshpanov et al., LREC-COLING 2024",
        default_sample=100,
    )
    protocol_note = (
        "Official reading comprehension gives an extractive reader one gold passage and scores "
        "the span with SQuAD-style EM/F1. Here each question's gold passage(s) plus "
        f"{DISTRACTORS} seeded passages of other questions from the same subject form a "
        "per-question corpus that the pipelines search and rank, and a generative model answers "
        "in a sentence. **accuracy** = some gold answer contained in the normalised answer "
        "(NFKC, casefold, Unicode punctuation removed, no article stripping); EM and token F1 "
        "are reported on the citation-stripped answer. Kazakh is agglutinative, so "
        "*stem match* (each gold token matched on its first 6 letters) is a lenient extra."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "em": "EM",
        "f1": "F1",
        "stem_match": "Stem match",
        "gold_in_context": "Gold passage in context",
        "abstained": "Abstained",
    }

    def __init__(self) -> None:
        self.files: list[DatasetFile] = []

    def _file(self, data_dir: Path) -> DatasetFile:
        manual = data_dir / "kazqad" / f"{NAME}.jsonl.gz"
        if manual.is_file():
            return fetch_file(HF_URL, manual)
        return fetch_file(URL, data_dir / "kazqad" / f"{NAME}.jsonl", expected_sha256=SHA256)

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        f = self._file(data_dir)
        self.files = [f]
        groups = group_questions(read_rows(f.path))
        qids = list(groups)
        picked = sample_indices(len(qids), sample, self.info.default_sample, seed)
        return build_items(groups, [qids[i] for i in picked], seed)

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        s = span_scores(answer.text, item.answers)
        recall = gold_doc_recall(item, answer)
        return {
            "correct": s["matched"] is not None,
            **s,
            "abstained": is_abstention(answer.text),
            "gold_in_context": None if recall is None else recall > 0,
        }

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        s = [r["score"] for r in rows]
        return {key: rate(x[key] for x in s) for key in self.metric_labels}

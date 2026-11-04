"""RGB: Retrieval-Augmented Generation Benchmark (Chen et al., AAAI 2024), English part.

Three of RGB's four abilities, built exactly like the authors' ``processdata`` in evalue.py
(``passage_num`` 5, ``random.seed(2333)`` per instance, first-N positive/negative passages,
then shuffled):

- ``rgb-noise``: noise robustness, the same questions at noise rates 0.0 / 0.4 / 0.8.
- ``rgb-negative``: negative rejection, noise rate 1.0 (only irrelevant passages).
- ``rgb-counterfactual``: counterfactual robustness, passages with a planted wrong fact.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ....pipelines import Answer
from ...scoring import is_abstention
from .. import Doc, ExternalItem, TaskInfo, register
from ..scoring import (
    RGB_REJECTION,
    flags_factual_error,
    gold_doc_recall,
    is_rejection,
    rate,
    rgb_checkanswer,
)
from ._common import DatasetFile, fetch_file, sample_indices

BASE = "https://raw.githubusercontent.com/chen700564/RGB/master/data/"
REFINE = ("en_refine.json", "1be6ec4b85364b5590c7683ed4df8d47535ae21bc021bbcd3f44a71579f64bcb")
FACT = ("en_fact.json", "92f4b2330ee407f74fbd923197028ef5140cfbc1f4b4092efec2d4d10ae6c9e5")
PASSAGE_NUM = 5
RGB_SEED = 2333  # evalue.py: random.seed(2333) before processdata() for every instance
NOISE_RATES = (0.0, 0.4, 0.8)

# reject_evalue.py / fact_evalue.py prompts (verbatim, Chinese demonstrations included), used
# when --judge-model is given. RGB ran them on gpt-3.5-turbo.
REJECT_JUDGE = """I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
    """  # noqa: E501

FACT_JUDGE = """Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response:  There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}
    """  # noqa: E501

_NOT_FOUND_RE = re.compile(r"could not find this information", re.IGNORECASE)


def _info(name: str, title: str) -> TaskInfo:
    return TaskInfo(
        name=name,
        title=title,
        source_url="https://github.com/chen700564/RGB",
        license="CC BY-NC-SA 4.0",
        citation="Chen et al., AAAI 2024",
        default_sample=100,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def process(instance: Mapping[str, Any], noise_rate: float, *, fact: bool = False) -> list[str]:
    """RGB's ``processdata`` for en / en_fact (correct_rate 0), same RNG call sequence."""
    rng = random.Random(RGB_SEED)
    neg_num = math.ceil(PASSAGE_NUM * noise_rate)
    pos_num = PASSAGE_NUM - neg_num
    positive: list[str] = list(instance["positive"])
    negative: list[str] = list(instance["negative"])
    if fact:
        indexes = list(range(len(positive)))
        selected = rng.sample(indexes, min(len(indexes), pos_num))
        docs = [instance["positive_wrong"][i] for i in selected]
        if neg_num > 0:
            docs += negative[:neg_num]
    else:
        if noise_rate == 1:
            neg_num, pos_num = PASSAGE_NUM, 0
        elif neg_num > len(negative):
            neg_num = len(negative)
            pos_num = PASSAGE_NUM - neg_num
        elif pos_num > len(positive):
            pos_num = len(positive)
            neg_num = PASSAGE_NUM - pos_num
        docs = positive[:pos_num] + negative[:neg_num]
    rng.shuffle(docs)
    return docs


def _flatten(answer: Any) -> tuple[str, ...]:
    groups = answer if isinstance(answer, list) else [answer]
    out: list[str] = []
    for g in groups:
        out.extend(str(a) for a in (g if isinstance(g, list) else [g]))
    return tuple(dict.fromkeys(out))


def _docs(passages: Sequence[str]) -> tuple[Doc, ...]:
    # RGB passages are untitled search snippets / news excerpts: neutral titles.
    return tuple(Doc(f"Document {i + 1}", p) for i, p in enumerate(passages))


def explicit_rejection(text: str) -> bool:
    """RGB's own rule (``"insufficient information" in prediction``) plus our NOT_FOUND."""
    return RGB_REJECTION in text.lower() or _NOT_FOUND_RE.search(text) is not None


def answer_correct(text: str, gold: Any) -> bool:
    """evalue.py: label [-1] on rejection, else every checkanswer label must be 1."""
    if explicit_rejection(text):
        return False
    labels = rgb_checkanswer(text, gold)
    return 0 not in labels and 1 in labels


class _RGB:
    info: TaskInfo
    file: tuple[str, str] = REFINE

    def __init__(self) -> None:
        self.files: list[DatasetFile] = []

    def _instances(self, data_dir: Path, sample: int | None, seed: int) -> list[dict[str, Any]]:
        name, sha = self.file
        f = fetch_file(BASE + name, data_dir / "rgb" / name, expected_sha256=sha)
        self.files = [f]
        rows = read_jsonl(f.path)
        return [rows[i] for i in sample_indices(len(rows), sample, self.info.default_sample, seed)]


@register("rgb-noise")
class RGBNoise(_RGB):
    info = _info("rgb-noise", "RGB noise robustness (en)")
    protocol_note = (
        "Same passages as RGB's evalue.py (5 passages, noise rates 0.0/0.4/0.8, seed 2333) and "
        "the same `checkanswer` (every gold element contained, case-insensitive), with a "
        "rejection (`insufficient information` or our not-found sentence) counted wrong. "
        "Differences: RGB pastes all 5 passages into its own prompt; here they form a "
        "per-question corpus that rag/agent/multi search (BM25) and rank with our prompts, so "
        "a passage can be missed by retrieval; RGB samples at temperature 0.7, we use 0."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "rejected": "Rejected",
        "gold_in_context": "Positive passage in context",
    }

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        instances = self._instances(data_dir, sample, seed)
        items = []
        for noise in NOISE_RATES:
            for inst in instances:
                passages = process(inst, noise)
                positives = set(inst["positive"])
                items.append(
                    ExternalItem(
                        id=f"{inst['id']}@noise-{noise}",
                        question=str(inst["query"]),
                        answers=_flatten(inst["answer"]),
                        category=f"noise-{noise}",
                        docs=_docs(passages),
                        extra={
                            "gold": inst["answer"],
                            "rgb_id": inst["id"],
                            "noise_rate": noise,
                            "gold_docs": tuple(i for i, p in enumerate(passages) if p in positives),
                        },
                    )
                )
        return items

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        recall = gold_doc_recall(item, answer)
        return {
            "correct": answer_correct(answer.text, item.extra["gold"]),
            "rejected": explicit_rejection(answer.text),
            "gold_in_context": None if recall is None else recall > 0,
        }

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "rejected": rate(r["score"]["rejected"] for r in rows),
            "gold_in_context": rate(r["score"]["gold_in_context"] for r in rows),
        }


@register("rgb-negative")
class RGBNegative(_RGB):
    info = _info("rgb-negative", "RGB negative rejection (en)")
    protocol_note = (
        "Noise rate 1.0: all 5 passages are irrelevant, so the right answer is to decline. "
        "RGB's evalue.py counts the exact phrase `insufficient information` that its prompt "
        "asks for, and reject_evalue.py additionally asks ChatGPT whether the answer addresses "
        "the question. Our prompts ask for a different sentence, so **rejection** = RGB's "
        "phrase, our not-found sentence or any phrase of our abstention detector; with "
        "`--judge-model` the reject_evalue.py prompt runs on that local model instead "
        "(`not addressed` -> rejected)."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "rejection_rate": "Rejection rate",
        "exact_phrase_rate": "RGB exact phrase",
        "answered_anyway": "Gave the gold answer anyway",
    }

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        return [
            ExternalItem(
                id=str(inst["id"]),
                question=str(inst["query"]),
                answers=(),
                category="negative",
                docs=_docs(process(inst, 1.0)),
                answerable=False,
                extra={"gold": inst["answer"], "rgb_id": inst["id"], "noise_rate": 1.0},
            )
            for inst in self._instances(data_dir, sample, seed)
        ]

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        rejected = is_rejection(answer.text)
        labels = rgb_checkanswer(answer.text, item.extra["gold"])
        return {
            "correct": rejected,
            "rejected": rejected,
            "exact_phrase": RGB_REJECTION in answer.text.lower(),
            "answered_anyway": 0 not in labels and 1 in labels,
        }

    def judge_prompt(self, item: ExternalItem, text: str) -> str:
        return REJECT_JUDGE.format(question=item.question, answer=text)

    def apply_judge(self, score: Mapping[str, Any], verdict: str) -> dict[str, Any]:
        rejected = "not addressed" in verdict  # reject_evalue.py's rule
        return {
            **score,
            "string_rejected": score.get("string_rejected", score["rejected"]),
            "rejected": rejected,
            "correct": rejected,
        }

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        s = [r["score"] for r in rows]
        return {
            "rejection_rate": rate(x["rejected"] for x in s),
            "exact_phrase_rate": rate(x["exact_phrase"] for x in s),
            "answered_anyway": rate(x["answered_anyway"] for x in s),
        }


@register("rgb-counterfactual")
class RGBCounterfactual(_RGB):
    info = _info("rgb-counterfactual", "RGB counterfactual robustness (en)")
    file = FACT
    protocol_note = (
        "en_fact.json with RGB's processdata for `_fact` (noise 0, correct_rate 0): up to 5 "
        "passages that all state the planted wrong answer. RGB's prompt for this ability "
        "(instruction_fact.yaml) warns that the documents may contain factual errors; our "
        "pipelines keep their normal prompts, so detection is spontaneous and rates are not "
        "directly comparable. **Error detection**: RGB checks `factual errors` in the output "
        "(evalue.py) and asks ChatGPT (fact_evalue.py); we accept that phrase plus common "
        "paraphrases (heuristic), or run the fact_evalue.py prompt on `--judge-model`. "
        "**Error correction** (official `correct_rate`) = detected and gives the true answer, "
        "divided by detected; a row is *correct* when it is corrected."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "error_detection_rate": "Error detection (ED)",
        "error_correction_rate": "Error correction (CR, of detected)",
        "true_answer_rate": "True answer given",
        "misled_rate": "Repeated the planted answer",
    }

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        return [
            ExternalItem(
                id=str(inst["id"]),
                question=str(inst["query"]),
                answers=_flatten(inst["answer"]),
                category="counterfactual",
                docs=_docs(process(inst, 0.0, fact=True)),
                extra={"gold": inst["answer"], "fake": inst["fakeanswer"], "rgb_id": inst["id"]},
            )
            for inst in self._instances(data_dir, sample, seed)
        ]

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        detected = flags_factual_error(answer.text)
        true_answer = answer_correct(answer.text, item.extra["gold"])
        fake = rgb_checkanswer(answer.text, item.extra["fake"])
        return {
            "correct": detected and true_answer,
            "detected": detected,
            "true_answer": true_answer,
            "misled": 0 not in fake and 1 in fake and not true_answer,
            "abstained": is_abstention(answer.text),
        }

    def judge_prompt(self, item: ExternalItem, text: str) -> str:
        return FACT_JUDGE.format(answer=text)

    def apply_judge(self, score: Mapping[str, Any], verdict: str) -> dict[str, Any]:
        detected = "has identified" in verdict or "Yes" in verdict  # fact_evalue.py's rule
        return {
            **score,
            "string_detected": score.get("string_detected", score["detected"]),
            "detected": detected,
            "correct": detected and score["true_answer"],
        }

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        s = [r["score"] for r in rows]
        detected = [x for x in s if x["detected"]]
        return {
            "error_detection_rate": rate(x["detected"] for x in s),
            "error_correction_rate": rate(x["true_answer"] for x in detected),
            "true_answer_rate": rate(x["true_answer"] for x in s),
            "misled_rate": rate(x["misled"] for x in s),
        }

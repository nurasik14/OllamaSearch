"""SimpleQA (Wei et al., 2024): short fact-seeking questions answered from the live web.

Official grading uses an LLM with three grades (CORRECT / INCORRECT / NOT_ATTEMPTED). Without
``--judge-model`` we grade with strings (gold contained -> CORRECT, abstention ->
NOT_ATTEMPTED, else INCORRECT); with it, the official grader prompt runs on that local model.
"""

from __future__ import annotations

import ast
import csv
import io
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ....pipelines import Answer
from ...scoring import is_abstention, match_alias
from .. import ExternalItem, TaskInfo, register
from ..scoring import strip_citations
from ._common import DatasetFile, fetch_file, sample_indices

URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SHA256 = "feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032"

# GRADER_TEMPLATE copied verbatim from openai/simple-evals, simpleqa_eval.py (MIT License,
# Copyright (c) 2024 OpenAI), https://github.com/openai/simple-evals
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".\x20
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.\x20
    - Predicted answers "100k" and "113k" are INCORRECT.\x20
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.\x20
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()  # noqa: E501

CHOICE_LETTER_TO_STRING = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
GRADES = tuple(CHOICE_LETTER_TO_STRING.values())
_LETTER = re.compile(r"(A|B|C)")
_PAREN = re.compile(r"\s*\([^)]*\)")


def parse_grade(response: str) -> str:
    """simpleqa_eval.py's ``re.search(r"(A|B|C)", ...)``, defaulting to NOT_ATTEMPTED."""
    m = _LETTER.search(response)
    return CHOICE_LETTER_TO_STRING[m.group(0) if m else "C"]


def aliases(gold: str) -> tuple[str, ...]:
    """The gold target plus forms the official grader would also accept by its rules:
    without a parenthetical ("Mumbai (Bombay)" -> "Mumbai"), and either side of a colon
    ("Season 7, Episode 20: White Wedding")."""
    out = [gold, _PAREN.sub("", gold).strip()]
    if ":" in gold:
        out += [part.strip() for part in gold.split(":")]
    return tuple(dict.fromkeys(a for a in out if len(a) > 1))


def string_grade(text: str, gold: str) -> str:
    clean = strip_citations(text)
    if clean and match_alias(clean, aliases(gold)) is not None:
        return "CORRECT"
    if not clean or is_abstention(clean):
        return "NOT_ATTEMPTED"
    return "INCORRECT"


def grade_metrics(grades: Sequence[str]) -> dict[str, float | None]:
    """SimpleQA's aggregate metrics (simpleqa_eval.py) over a list of grades."""
    if not grades:
        return {}
    n = len(grades)
    correct = grades.count("CORRECT") / n
    incorrect = grades.count("INCORRECT") / n
    attempted = correct + incorrect
    given = correct / attempted if attempted > 0 else 0.0
    return {
        "is_correct": correct,
        "is_incorrect": incorrect,
        "is_not_attempted": grades.count("NOT_ATTEMPTED") / n,
        "is_given_attempted": attempted,
        "accuracy_given_attempted": given,
        "f1": 2 * given * correct / (given + correct) if given + correct > 0 else 0.0,
    }


@register("simpleqa")
class SimpleQA:
    info = TaskInfo(
        name="simpleqa",
        title="SimpleQA (live web)",
        source_url="https://github.com/openai/simple-evals",
        license="MIT",
        citation="Wei et al., 2024",
        default_sample=100,
        live=True,
    )
    protocol_note = (
        "Official: the model answers from memory and an LLM grader (GPT-4-class) assigns "
        "CORRECT / INCORRECT / NOT_ATTEMPTED with the GRADER_TEMPLATE of simpleqa_eval.py. "
        "Here rag/agent/multi search the live web through SearXNG (results recorded in "
        "search_record.json so every model sees the same results) and closed is the official "
        "memory-only setting. Grading: with `--judge-model`, the official template runs on that "
        "local model (weaker than the official grader); otherwise a string grader: gold target "
        "(or its main part) contained -> CORRECT, abstention -> NOT_ATTEMPTED, else INCORRECT. "
        "The string grader misses paraphrases and numbers written differently, so it "
        "under-counts CORRECT. Sampling uses our seeded permutation, not simple-evals' "
        "`random.Random(0).sample`."
    )
    metric_labels: ClassVar[dict[str, str]] = {
        "is_correct": "Correct",
        "is_incorrect": "Incorrect",
        "is_not_attempted": "Not attempted",
        "accuracy_given_attempted": "Correct given attempted",
        "f1": "F-score",
    }

    def __init__(self) -> None:
        self.files: list[DatasetFile] = []

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        dest = data_dir / "simpleqa" / "simple_qa_test_set.csv"
        f = fetch_file(URL, dest, expected_sha256=SHA256)
        self.files = [f]
        rows = list(csv.DictReader(io.StringIO(f.path.read_text(encoding="utf-8"))))
        items = []
        for i in sample_indices(len(rows), sample, self.info.default_sample, seed):
            row = rows[i]
            meta = ast.literal_eval(row["metadata"])  # a Python dict literal, as published
            items.append(
                ExternalItem(
                    id=f"simpleqa-{i:04d}",
                    question=row["problem"],
                    answers=(row["answer"],),
                    category=str(meta.get("topic", "Other")),
                    extra={"answer_type": meta.get("answer_type"), "urls": meta.get("urls", [])},
                )
            )
        return items

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        grade = string_grade(answer.text, item.answers[0])
        return {"correct": grade == "CORRECT", "grade": grade, "string_grade": grade}

    def judge_prompt(self, item: ExternalItem, text: str) -> str:
        return GRADER_TEMPLATE.format(
            question=item.question, target=item.answers[0], predicted_answer=text
        )

    def apply_judge(self, score: Mapping[str, Any], verdict: str) -> dict[str, Any]:
        grade = parse_grade(verdict)
        return {**score, "grade": grade, "correct": grade == "CORRECT"}

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = dict(grade_metrics([r["score"]["grade"] for r in rows]))
        if any(r["score"].get("grader", "string") != "string" for r in rows):
            strings = grade_metrics([r["score"]["string_grade"] for r in rows])
            out["string_is_correct"] = strings.get("is_correct")
        return out

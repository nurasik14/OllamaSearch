"""SearchBench question sets (JSONL)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ("single-hop", "multi-hop", "conflicting", "multilingual", "unanswerable")
LANGS = ("en", "ru", "kk")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    lang: str
    answers: tuple[str, ...]
    category: str
    gold_urls: tuple[str, ...] = ()
    outdated: tuple[str, ...] = field(default=())

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


class DatasetError(ValueError):
    pass


def load_questions(path: str | Path) -> list[Question]:
    path = Path(path)
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            q = Question(
                id=str(raw["id"]),
                question=str(raw["question"]),
                lang=str(raw["lang"]),
                answers=tuple(str(a) for a in raw.get("answers", [])),
                category=str(raw["category"]),
                gold_urls=tuple(str(u) for u in raw.get("gold_urls", [])),
                outdated=tuple(str(u) for u in raw.get("outdated", [])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise DatasetError(f"{path}:{lineno}: {exc}") from exc
        if q.category not in CATEGORIES:
            raise DatasetError(f"{path}:{lineno}: unknown category {q.category!r}")
        if q.answerable and not q.answers:
            raise DatasetError(f"{path}:{lineno}: answerable question without answers")
        if q.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate id {q.id}")
        seen.add(q.id)
        questions.append(q)
    return questions


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

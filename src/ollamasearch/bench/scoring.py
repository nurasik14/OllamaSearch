"""SearchBench scoring: alias matching, abstention detection, citation and tool metrics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .dataset import Question

_THOUSANDS = re.compile(r"(?<=\d)[,    '](?=\d{3}(?!\d))")
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_LOOSE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_PUNCT = re.compile(r"[^\w\s.]|_")
_SPACES = re.compile(r"\s+")

# Phrases (already normalised) that signal "I could not find it", in en / ru / kk.
ABSTENTION_PHRASES = (
    # en
    "could not find",
    "couldn t find",
    "cannot find",
    "can t find",
    "unable to find",
    "did not find",
    "didn t find",
    "not found",
    "no information",
    "no relevant information",
    "not enough information",
    "insufficient",
    "do not contain",
    "does not contain",
    "don t contain",
    "doesn t contain",
    "do not mention",
    "does not mention",
    "not mentioned",
    "no mention",
    "not specified",
    "not provided",
    "not available",
    "don t know",
    "do not know",
    "unknown",
    "no data",
    # ru
    "не удалось найти",
    "не удалось",
    "не могу найти",
    "не нашел",
    "не найден",
    "нет информации",
    "нет данных",
    "нет сведений",
    "информации нет",
    "информация отсутствует",
    "отсутствует информация",
    "недостаточно",
    "не содерж",
    "не указан",
    "не упомина",
    "не знаю",
    "неизвестно",
    # kk
    "табылмады",
    "таба алмадым",
    "ақпарат жоқ",
    "мәлімет жоқ",
    "мәліметтер жоқ",
    "деректер жоқ",
    "білмеймін",
    "жеткіліксіз",
    "көрсетілмеген",
    "айтылмаған",
    "белгісіз",
)


def normalize(text: str) -> str:
    """Casefold, unify digits/separators, strip punctuation, collapse whitespace.

    ``"1,250"``, ``"1 250"`` and ``"1250"`` all become ``"1250"``; ``"3,5"`` becomes ``"3.5"``.
    """
    s = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    s = _THOUSANDS.sub("", s)
    s = _DECIMAL_COMMA.sub(".", s)
    s = _LOOSE_DOT.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def match_alias(answer: str, aliases: Iterable[str]) -> str | None:
    """Return the first alias that occurs in ``answer`` as a whole-token sequence."""
    haystack = f" {normalize(answer)} "
    for alias in aliases:
        needle = normalize(alias)
        if needle and f" {needle} " in haystack:
            return alias
    return None


def is_abstention(answer: str) -> bool:
    norm = normalize(answer)
    if not norm:
        return False
    return any(phrase in norm for phrase in ABSTENTION_PHRASES)


@dataclass
class Score:
    correct: bool
    matched: str | None
    abstained: bool
    stale: bool
    citation_hit: bool | None  # None when not applicable (closed-book / unanswerable)
    gold_in_context: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def score_answer(
    question: Question,
    text: str,
    *,
    cited_urls: Sequence[str] = (),
    context_urls: Sequence[str] = (),
    mode: str = "rag",
) -> Score:
    abstained = is_abstention(text)
    if question.answerable:
        matched = match_alias(text, question.answers)
        correct = matched is not None
    else:
        matched = None
        correct = abstained
    stale = (
        not correct and bool(question.outdated) and match_alias(text, question.outdated) is not None
    )
    gold = {_norm_url(u) for u in question.gold_urls}
    citation_hit: bool | None = None
    gold_in_context: bool | None = None
    if mode != "closed" and question.answerable and gold:
        citation_hit = any(_norm_url(u) in gold for u in cited_urls)
        gold_in_context = any(_norm_url(u) in gold for u in context_urls)
    return Score(correct, matched, abstained, stale, citation_hit, gold_in_context)


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0..100). Empty input -> nan."""
    if not values:
        return math.nan
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _rate(flags: Iterable[bool | None]) -> float | None:
    vals = [bool(f) for f in flags if f is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate scored rows (as written to runs.jsonl) by (model, mode)."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mode"])].append(row)

    out: list[dict[str, Any]] = []
    for (model, mode), items in sorted(groups.items()):
        scores = [r["score"] for r in items]
        answers = [r["answer"] for r in items]
        answerable = [r for r in items if r["category"] != "unanswerable"]
        unanswerable = [r for r in items if r["category"] == "unanswerable"]
        by_cat: dict[str, list[bool]] = defaultdict(list)
        by_lang: dict[str, list[bool]] = defaultdict(list)
        for r in items:
            by_cat[r["category"]].append(r["score"]["correct"])
            by_lang[r["lang"]].append(r["score"]["correct"])
        totals = [a["timings"].get("total_s", 0.0) for a in answers]
        eval_tokens = sum(a["tokens"].get("eval_tokens", 0) for a in answers)
        eval_s = sum(a["tokens"].get("eval_s", 0.0) for a in answers)
        entry: dict[str, Any] = {
            "model": model,
            "mode": mode,
            "n": len(items),
            "accuracy": _rate(s["correct"] for s in scores),
            "answerable_accuracy": _rate(r["score"]["correct"] for r in answerable),
            "unanswerable_accuracy": _rate(r["score"]["correct"] for r in unanswerable),
            "false_abstention_rate": _rate(r["score"]["abstained"] for r in answerable),
            "stale_rate": _rate(
                r["score"]["stale"] for r in items if r["category"] == "conflicting"
            ),
            "citation_hit_rate": _rate(s["citation_hit"] for s in scores),
            "gold_in_context_rate": _rate(s["gold_in_context"] for s in scores),
            "by_category": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
            "by_lang": {k: sum(v) / len(v) for k, v in sorted(by_lang.items())},
            "latency_p50_s": percentile(totals, 50),
            "latency_p95_s": percentile(totals, 95),
            "latency_mean_s": sum(totals) / len(totals) if totals else math.nan,
            "tokens_per_s": eval_tokens / eval_s if eval_s else None,
            "mean_prompt_tokens": sum(a["tokens"].get("prompt_tokens", 0) for a in answers)
            / len(answers),
            "mean_eval_tokens": eval_tokens / len(answers),
            "error_rate": _rate(a.get("error") is not None for a in answers),
        }
        if mode == "agent":
            calls = [c for a in answers for c in a["tool_calls"]]
            entry["agent"] = {
                "tool_calls_per_question": len(calls) / len(answers),
                "invalid_call_rate": (
                    sum(c["status"] == "invalid" for c in calls) / len(calls) if calls else None
                ),
                "searched_rate": _rate(
                    any(
                        c["name"] == "web_search" and c["status"] != "invalid"
                        for c in a["tool_calls"]
                    )
                    for a in answers
                ),
                "fetched_rate": _rate(
                    any(c["name"] == "web_fetch" and c["status"] == "ok" for c in a["tool_calls"])
                    for a in answers
                ),
                "text_tool_call_rate": (
                    sum(bool(c.get("from_text")) for c in calls) / len(calls) if calls else None
                ),
                "step_limit_rate": _rate(a.get("step_limit_reached") for a in answers),
                "mean_steps": sum(a.get("steps", 0) for a in answers) / len(answers),
            }
        out.append(entry)
    return {"groups": out}

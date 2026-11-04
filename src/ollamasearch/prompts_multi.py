"""Prompts and JSON schemas for the multi-agent research pipeline (persona Arika).

Every role gets a narrow job and answers in JSON constrained by a schema (Ollama structured
output), except the synthesizer, which writes the user-facing answer as plain text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .prompts import NOT_FOUND

# ---------------------------------------------------------------------------------------------
# JSON schemas


def _obj(properties: dict[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties if required is None else required),
    }


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}

# Key order matters: Ollama's grammar makes the model write keys in schema order, so the
# free-text fields come first and each yes/no decision is taken after writing them.
PLAN_SCHEMA = _obj(
    {
        "sub_questions": {
            "type": "array",
            "items": _obj({"question": _STR, "query": _STR, "needs_previous": _BOOL}),
            "minItems": 1,
            "maxItems": 3,
        },
        "simple": _BOOL,
    }
)

REPLAN_SCHEMA = _obj(
    {
        "sub_questions": {
            "type": "array",
            "items": _obj({"question": _STR, "query": _STR}),
            "maxItems": 2,
        }
    }
)

READ_SCHEMA = _obj(
    {
        "finding": _STR,
        "source_numbers": {"type": "array", "items": {"type": "integer"}},
        "date_or_version": {"type": ["string", "null"]},
        "found": _BOOL,
    }
)

VERIFY_SCHEMA = _obj(
    {
        "problems": {"type": "array", "items": _STR},
        "supported": _BOOL,
        "follow_up_query": {"type": ["string", "null"]},
    }
)

# ---------------------------------------------------------------------------------------------
# Planner

PLANNER_SYSTEM = """You are Arika's research planner. You plan web searches; you never answer.

Split the user's question into 1 to 3 sub-questions. Each sub-question gets one short search
query made of the key names and terms (no full sentences, keep names exactly as written).

- Most questions are simple: one fact about one named thing. Then set "simple": true and
  return exactly one sub-question (the question itself).
- A question that chains facts (e.g. "the company that built X", "the university where Y
  works") needs one sub-question per hop. A later hop that needs the answer of an earlier one
  gets "needs_previous": true; describe it in words, e.g. "When was the company that built X
  founded?".
- A question that compares or combines several independent things gets one sub-question
  for each thing ("needs_previous": false).
- Write sub-questions and queries in the language of the question.

Reply with JSON only."""

PLANNER_EXAMPLE = """Example:
Question: In what year was the company that built the Harnwick Tunnel founded?
{"sub_questions": [
 {"question": "Which company built the Harnwick Tunnel?",
  "query": "Harnwick Tunnel built by", "needs_previous": false},
 {"question": "In what year was the company that built the Harnwick Tunnel founded?",
  "query": "Harnwick Tunnel company founded", "needs_previous": true}], "simple": false}

Example:
Question: What is the population of Lysander Bay?
{"sub_questions": [
 {"question": "What is the population of Lysander Bay?",
  "query": "Lysander Bay population", "needs_previous": false}], "simple": true}"""


def planner_prompt(question: str, history_text: str = "") -> str:
    convo = f"Conversation so far (resolve words like 'it' or 'there'):\n{history_text}\n\n"
    return (
        f"{PLANNER_EXAMPLE}\n\n{convo if history_text else ''}Question: {question}\nPlan as JSON:"
    )


REPLANNER_SYSTEM = """You are Arika's research planner. Some searches are done; plan the next ones.

Given the question, the findings so far and the open sub-questions, write at most 2 new
sub-questions with short search queries that fill the remaining gaps. Put the concrete names
from the findings into the queries (e.g. the company name found in the first step).
If the findings already answer the question, or nothing useful is left to search, return an
empty list. Never repeat a query that was already searched. Reply with JSON only."""


def replanner_prompt(
    question: str,
    findings: Sequence[str],
    open_questions: Sequence[str],
    searched: Sequence[str],
) -> str:
    found = "\n".join(f"- {f}" for f in findings) or "- (nothing found yet)"
    pending = "\n".join(f"- {q}" for q in open_questions) or "- (none)"
    done = "\n".join(f"- {q}" for q in searched) or "- (none)"
    return (
        f"Question: {question}\n\nFindings so far:\n{found}\n\n"
        f"Open sub-questions:\n{pending}\n\nQueries already searched:\n{done}\n\n"
        "Next sub-questions as JSON:"
    )


# ---------------------------------------------------------------------------------------------
# Reader (one per sub-question)

READER_SYSTEM = """You are Arika's reader. You get numbered source passages and one sub-question.

- Answer the sub-question only from the passages. Ignore menus, ads, sidebars and passages
  about other entities with similar names.
- "finding": one or two sentences with the answer and the key names and numbers, written so
  that it makes sense without the passages. If the answer is not there, say briefly what the
  passages do say about the subject (useful partial facts), or leave it empty.
- "source_numbers": the numbers of the passages that state it.
- "date_or_version": the year, date or version the fact refers to (e.g. "2024 census"), or
  null if the passages give none.
- "found": true if your finding answers the sub-question (facts from several passages may
  be combined, e.g. the company named on one page and its founding year on another); false
  if the passages do not contain the answer.
- If passages disagree, report the most recent value and say that the older value is outdated.

Reply with JSON only."""


def reader_prompt(sub_question: str, sources_text: str) -> str:
    return f"Sources:\n\n{sources_text}\n\nSub-question: {sub_question}\n\nYour JSON:"


# ---------------------------------------------------------------------------------------------
# Synthesizer


@dataclass(frozen=True)
class FindingView:
    """A reader result as shown to the synthesizer and the re-planner."""

    sub_question: str
    finding: str
    source_numbers: Sequence[int]
    date_or_version: str | None = None

    def line(self) -> str:
        cites = "".join(f"[{n}]" for n in self.source_numbers)
        dated = f" (as of: {self.date_or_version})" if self.date_or_version else ""
        return f"{self.finding}{dated} {cites}".strip()


SYNTH_SYSTEM = f"""You are Arika, a web research assistant.

Write the final answer to the user's question using only the research findings below.
- Answer directly and concisely (one to three sentences), in the language of the question.
- Combine findings when the question chains facts (e.g. the company from one finding, its
  founding year from another).
- Cite the source numbers written after the findings you use, in square brackets, exactly
  as they appear (e.g. a finding ending in [6] is cited as [6]).
- If findings conflict, give the most recent value (see "as of") and say briefly that the
  older value is outdated.
- Do not add facts that are not in the findings.
- If the findings do not answer the question, reply exactly: "{NOT_FOUND}"
"""


def synth_prompt(question: str, findings: Sequence[FindingView], language: str) -> str:
    # Bullets, not "1." / "2.": numbered findings get confused with source numbers.
    lines = "\n".join(f"- {f.line()}" for f in findings)
    example = next((f"[{f.source_numbers[0]}]" for f in findings if f.source_numbers), "[1]")
    return (
        f"Research findings (source numbers in brackets):\n{lines}\n\nQuestion: {question}\n\n"
        f"Answer in {language}. Cite with the source numbers shown after the findings, "
        f"e.g. {example}:"
    )


# ---------------------------------------------------------------------------------------------
# Verifier

VERIFIER_SYSTEM = """You are Arika's fact checker. You get a question, a draft answer and the
source passages it cites.

- Check every factual claim of the draft (names, numbers, dates) against the passages.
- "supported": true if every claim is stated in the passages and the draft answers the
  question. Wording, language and citation format do not matter.
- "problems": short notes on claims that are missing from or contradict the passages
  (empty list if supported).
- "follow_up_query": if something is unsupported or missing, one short search query that
  could find it; otherwise null.

Reply with JSON only."""


def verifier_prompt(question: str, draft: str, sources_text: str) -> str:
    return (
        f"Sources:\n\n{sources_text}\n\nQuestion: {question}\n\nDraft answer: {draft}\n\nYour JSON:"
    )


# ---------------------------------------------------------------------------------------------
# Language hint

_KAZAKH_LETTERS = set("әғқңөұүһі")


def question_language(question: str) -> str:
    """Rough script-based language name, used to tell small models which language to answer in."""
    letters = [c for c in question.casefold() if c.isalpha()]
    if not letters:
        return "the language of the question"
    cyrillic = sum("а" <= c <= "я" or c in "ё" or c in _KAZAKH_LETTERS for c in letters)
    if cyrillic * 2 < len(letters):
        return "the language of the question"
    if any(c in _KAZAKH_LETTERS for c in letters):
        return "Kazakh"
    return "Russian"

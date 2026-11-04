"""Scoring helpers shared by the external-benchmark adapters.

Each function mirrors a published evaluation script where one exists (HotpotQA's
``hotpot_evaluate_v1.py``, RGB's ``checkanswer``) so numbers stay comparable; the deviations
needed for sentence-style answers are documented next to the code and in
``docs/external-benchmarks.md``.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ...pipelines import Answer
from ...prompts import NOT_FOUND
from ...rank import STEM_LEN
from ...search import CorpusBackend
from ..scoring import is_abstention
from ..scoring import normalize as bench_normalize
from . import Doc, ExternalItem

_CITATION = re.compile(r"\[\d{1,3}(?:\s*[,;]\s*\d{1,3})*\]")
_SPACES = re.compile(r"\s+")
_SLUG = re.compile(r"[^a-z0-9]+")


def strip_citations(text: str) -> str:
    """Drop ``[1]`` / ``[2, 3]`` markers so they don't count as answer tokens."""
    return _SPACES.sub(" ", _CITATION.sub(" ", text)).strip()


# --- HotpotQA (official hotpot_evaluate_v1.py) ---------------------------------------------

_HOTPOT_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def hotpot_normalize(s: str) -> str:
    """``normalize_answer`` from hotpot_evaluate_v1.py: lower, punctuation, articles, spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _HOTPOT_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def hotpot_em(prediction: str, gold: str) -> bool:
    return hotpot_normalize(prediction) == hotpot_normalize(gold)


def hotpot_f1(prediction: str, gold: str) -> tuple[float, float, float]:
    """(f1, precision, recall) exactly as ``f1_score`` in hotpot_evaluate_v1.py."""
    pred, truth = hotpot_normalize(prediction), hotpot_normalize(gold)
    zero = (0.0, 0.0, 0.0)
    special = ("yes", "no", "noanswer")
    if pred in special and pred != truth:
        return zero
    if truth in special and pred != truth:
        return zero
    return _token_f1(pred.split(), truth.split())


def _token_f1(pred: Sequence[str], truth: Sequence[str]) -> tuple[float, float, float]:
    common = Counter(pred) & Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0, 0.0, 0.0
    precision = same / len(pred)
    recall = same / len(truth)
    return 2 * precision * recall / (precision + recall), precision, recall


def contains_tokens(haystack: str, needle: str) -> bool:
    """``needle`` occurs in ``haystack`` as a whole-token sequence (both already normalised)."""
    return bool(needle) and f" {needle} " in f" {haystack} "


# --- language-neutral normalisation (KazQAD, SimpleQA) --------------------------------------


def unicode_normalize(s: str) -> str:
    """NFKC, casefold, every Unicode punctuation/symbol -> space, collapse whitespace.

    Safe for Kazakh/Cyrillic: no English article removal, no ASCII-only punctuation table.
    """
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)
    return " ".join(s.split())


def token_f1(prediction: str, gold: str, norm: Callable[[str], str] = unicode_normalize) -> float:
    return _token_f1(norm(prediction).split(), norm(gold).split())[0]


def stem_contains(answer_norm: str, gold_norm: str, stem: int = STEM_LEN) -> bool:
    """Lenient containment for agglutinative languages: every gold token matches an answer
    token on its first ``stem`` characters (``ұлтабарға`` ~ ``ұлтабар``), in order."""
    gold = [t[:stem] for t in gold_norm.split()]
    toks = [t[:stem] for t in answer_norm.split()]
    if not gold:
        return False
    return any(toks[i : i + len(gold)] == gold for i in range(len(toks) - len(gold) + 1))


def span_scores(text: str, golds: Iterable[str]) -> dict[str, Any]:
    """Containment / EM / token-F1 of an answer against gold aliases (best over aliases)."""
    clean = strip_citations(text)
    norm = unicode_normalize(clean)
    golds = [g for g in golds if unicode_normalize(g)]
    matched = next((g for g in golds if contains_tokens(norm, unicode_normalize(g))), None)
    stem = matched is not None or any(stem_contains(norm, unicode_normalize(g)) for g in golds)
    return {
        "matched": matched,
        "em": any(norm == unicode_normalize(g) for g in golds),
        "f1": max((token_f1(clean, g) for g in golds), default=0.0),
        "stem_match": stem,
    }


# --- RGB (chen700564/RGB evalue.py) ----------------------------------------------------------

RGB_REJECTION = "insufficient information"
RGB_FACT_ERROR = "factual errors"


def rgb_checkanswer(prediction: str, ground_truth: Any) -> list[int]:
    """Port of RGB's ``checkanswer``: one label per gold element.

    A string element must appear (lower-cased substring); a list element is a set of aliases
    of which any may appear. The answer is correct when every label is 1.
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        if isinstance(instance, list):
            flag = any(str(i).lower() in prediction for i in instance)
        else:
            flag = str(instance).lower() in prediction
        labels.append(int(flag))
    return labels


_NOT_FOUND_NORM = bench_normalize(NOT_FOUND)


def is_rejection(text: str) -> bool:
    """RGB's rejection phrase, our pipelines' NOT_FOUND sentence, or any abstention phrase."""
    return (
        RGB_REJECTION in text.lower()
        or _NOT_FOUND_NORM in bench_normalize(text)
        or is_abstention(text)
    )


_FACT_FLAGS = re.compile(
    r"factual(?:ly)? (?:errors?|incorrect|inaccura)"
    r"|(?:contain|contains|has|have|with) (?:an? )?(?:errors?|mistakes?|inaccurac|incorrect)"
    r"|(?:is|are|seems?|appears?(?: to be)?) (?:incorrect|inaccurate|wrong|erroneous|mistaken)"
    r"|(?:incorrect|inaccurate|wrong|erroneous|false|conflicting) "
    r"(?:information|claim|statement|date|documents?|sources?|passages?)"
    r"|misinformation|contradict|not (?:accurate|correct)",
    re.IGNORECASE,
)


def flags_factual_error(text: str) -> bool:
    """Heuristic stand-in for RGB's ChatGPT error-detection judge (fact_evalue.py).

    RGB's own string check is ``"factual errors" in prediction`` (evalue.py); we accept that
    and common paraphrases ("the documents are incorrect", "contains an error", ...).
    """
    return RGB_FACT_ERROR in text.lower() or _FACT_FLAGS.search(text) is not None


# --- per-item corpora --------------------------------------------------------------------------


def doc_name(index: int, doc: Doc) -> str:
    """Stable file name of an item's ``index``-th document (``03-scott-derrickson.html``)."""
    slug = _SLUG.sub("-", unicodedata.normalize("NFKD", doc.title).lower()).strip("-")[:48]
    return f"{index:02d}-{slug or 'doc'}.html"


def doc_url(item: ExternalItem, index: int) -> str:
    return CorpusBackend.url_for(doc_name(index, item.docs[index]))


def gold_doc_recall(item: ExternalItem, answer: Answer) -> float | None:
    """Share of the item's gold documents (``extra["gold_docs"]``) that reached the model.

    "Reached" means the URL is in ``answer.context_urls``: RAG context sources, or agent
    search results and fetched pages. ``None`` for closed-book or items without gold docs.
    """
    gold: Sequence[int] = item.extra.get("gold_docs", ())
    if answer.mode == "closed" or not gold or not item.docs:
        return None
    seen = {u.rstrip("/") for u in answer.context_urls}
    return sum(doc_url(item, i) in seen for i in gold) / len(gold)


# --- aggregation helpers ------------------------------------------------------------------------


def rate(flags: Iterable[Any]) -> float | None:
    """Mean of the non-``None`` values (booleans or numbers); ``None`` when there are none."""
    vals = [float(f) for f in flags if f is not None]
    return sum(vals) / len(vals) if vals else None


def scores_of(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r["score"] for r in rows]

"""Third-party benchmarks run through the same pipelines as SearchBench.

This module is the contract between the runner and the per-dataset adapters. An adapter turns
a published dataset into :class:`ExternalItem` objects. Each item carries its own documents
(``docs``) that become a tiny per-item corpus (BM25 search + served pages), or none when the
task searches the live web. The adapter also scores answers.

Datasets are downloaded at run time into ``data_dir`` and never committed. Every adapter
records the source URL, license and citation so reports can credit the authors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ...pipelines import Answer


@dataclass(frozen=True)
class Doc:
    """One retrievable document for an item (a Wikipedia paragraph, a news passage, ...)."""

    title: str
    text: str


@dataclass(frozen=True)
class ExternalItem:
    """One benchmark question in a dataset-neutral shape."""

    id: str
    question: str
    answers: tuple[str, ...]  # accepted answers (aliases); empty when not answerable
    category: str  # dataset-specific bucket, e.g. "bridge", "noise-0.4", "negative"
    lang: str = "en"
    docs: tuple[Doc, ...] = ()  # per-item corpus; empty = live web search (needs SearXNG)
    answerable: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)  # e.g. counterfactual wrong answer


@dataclass(frozen=True)
class TaskInfo:
    name: str  # registry key, e.g. "hotpotqa"
    title: str  # e.g. "HotpotQA (distractor, dev)"
    source_url: str
    license: str
    citation: str  # short "Author et al., Venue Year"
    default_sample: int
    live: bool = False  # True: items have no docs and use SearXNG


@runtime_checkable
class ExternalTask(Protocol):
    info: TaskInfo

    def load(self, data_dir: Path, sample: int | None, seed: int) -> list[ExternalItem]:
        """Download (once, cached in ``data_dir``) and return a deterministic sample.

        ``sample=None`` means ``info.default_sample``; ``sample=0`` means all items.
        Same ``seed`` and file -> same items in the same order.
        """
        ...

    def score(self, item: ExternalItem, answer: Answer) -> dict[str, Any]:
        """Score one answer. Must include ``"correct": bool``; may add task metrics."""
        ...

    def summarize(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Task-specific aggregate metrics over rows of one model x mode (may be ``{}``)."""
        ...


#: name -> factory. Adapters register themselves here (filled in by the dataset modules).
REGISTRY: dict[str, Callable[[], ExternalTask]] = {}


def register(name: str) -> Callable[[Callable[[], ExternalTask]], Callable[[], ExternalTask]]:
    def deco(factory: Callable[[], ExternalTask]) -> Callable[[], ExternalTask]:
        REGISTRY[name] = factory
        return factory

    return deco


def available_tasks() -> list[str]:
    """Registered task names (importing the adapter modules registers them)."""
    from . import tasks  # noqa: F401  (side effect: registration)

    return sorted(REGISTRY)


def get_task(name: str) -> ExternalTask:
    names = available_tasks()
    if name not in REGISTRY:
        raise ValueError(f"unknown external task {name!r}; choose from {names}")
    return REGISTRY[name]()


def default_data_dir() -> Path:
    return Path.home() / ".cache" / "ollamasearch" / "datasets"

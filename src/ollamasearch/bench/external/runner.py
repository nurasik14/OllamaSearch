"""Runner for third-party benchmarks: items x models x modes -> runs.jsonl + summary + report."""

from __future__ import annotations

import html
import json
import logging
import math
import platform
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from ... import __version__
from ...config import Settings
from ...fetch import Fetcher
from ...llm import LLM, ChatClient
from ...pipelines import MODES, Pipeline, build_pipeline
from ...search import CorpusBackend, RecordingSearch, SearchBackend, SearchResult, SearxngBackend
from ..runner import _tilde, read_rows, row_key
from ..scoring import agent_metrics, multi_metrics, percentile
from . import Doc, ExternalItem, ExternalTask, default_data_dir, get_task
from .report import write_report
from .scoring import doc_name, rate

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, dict[str, Any]], None]


@dataclass
class ExternalConfig:
    task: str  # registry name, e.g. "hotpotqa"
    models: Sequence[str]
    modes: Sequence[str]  # any of pipelines.MODES ("closed", "rag", "agent", "multi")
    out_dir: Path
    data_dir: Path = field(default_factory=default_data_dir)
    sample: int | None = None  # None = task default, 0 = all
    seed: int = 42
    repeats: int = 1
    judge_model: str | None = None  # optional local LLM grader (e.g. SimpleQA's grader prompt)
    settings: Settings = field(default_factory=Settings)


@runtime_checkable
class JudgedTask(Protocol):
    """A task whose official protocol grades answers with an LLM (``--judge-model``)."""

    def judge_prompt(self, item: ExternalItem, text: str) -> str: ...

    def apply_judge(self, score: Mapping[str, Any], verdict: str) -> dict[str, Any]: ...


# --- per-item corpora ----------------------------------------------------------------------


def render_doc(doc: Doc) -> str:
    """A minimal page for one document: ``<title>`` + ``<main>`` with one ``<p>`` per line."""
    title = html.escape(doc.title)
    paras = "".join(f"<p>{html.escape(p)}</p>" for p in doc.text.split("\n") if p.strip())
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
        f"<body><main><h1>{title}</h1>{paras}</main></body></html>"
    )


def item_pages(item: ExternalItem) -> dict[str, str]:
    """``{file name: html}`` for the item's documents (URLs ``http://corpus.local/<name>``)."""
    return {doc_name(i, d): render_doc(d) for i, d in enumerate(item.docs)}


class _NoResults:
    name = "corpus"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return []


class ItemCorpora:
    """Per-item corpora behind one shared fetcher.

    :meth:`activate` builds (or reuses, LRU-cached) the item's :class:`CorpusBackend` and
    points the fetcher's transport at it, so pages of other items are never served.
    """

    def __init__(self, settings: Settings, cache_size: int = 512) -> None:
        self._cache: OrderedDict[str, SearchBackend] = OrderedDict()
        self._cache_size = cache_size
        self._current: CorpusBackend | None = None
        self.fetcher = Fetcher(
            transport=httpx.MockTransport(self._handle),
            concurrency=settings.fetch_concurrency,
            use_truststore=False,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self._current is None:
            return httpx.Response(404, text="no corpus")
        return self._current.transport().handle_request(request)

    def activate(self, item: ExternalItem) -> SearchBackend:
        backend = self._cache.get(item.id)
        if backend is None:
            backend = CorpusBackend.from_html(item_pages(item)) if item.docs else _NoResults()
            self._cache[item.id] = backend
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(item.id)
        self._current = backend if isinstance(backend, CorpusBackend) else None
        return backend

    def close(self) -> None:
        self.fetcher.close()


@dataclass
class _Resources:
    corpora: ItemCorpora | None = None
    live: tuple[SearchBackend, Fetcher] | None = None


@contextmanager
def _resources(cfg: ExternalConfig, task: ExternalTask) -> Iterator[_Resources]:
    res = _Resources()
    needs_search = any(m != "closed" for m in cfg.modes)
    with ExitStack() as stack:
        if needs_search and task.info.live:
            # Same query -> same results for every model and on resume (see bench/runner.py).
            backend = RecordingSearch(
                SearxngBackend(cfg.settings.searxng_url, timeout=cfg.settings.search_timeout),
                cfg.out_dir / "search_record.json",
            )
            fetcher = stack.enter_context(Fetcher.from_settings(cfg.settings))
            res.live = (backend, fetcher)
        elif needs_search:
            res.corpora = ItemCorpora(cfg.settings)
            stack.callback(res.corpora.close)
        yield res


def _pipeline(mode: str, llm: LLM, item: ExternalItem, res: _Resources, s: Settings) -> Pipeline:
    if mode == "closed":
        return build_pipeline(mode, llm, None, None, s)
    if res.live is not None:
        return build_pipeline(mode, llm, res.live[0], res.live[1], s)
    assert res.corpora is not None
    backend = res.corpora.activate(item)
    return build_pipeline(mode, llm, backend, res.corpora.fetcher, s)


# --- run ----------------------------------------------------------------------------------


def _llm(client: ChatClient, model: str, s: Settings) -> LLM:
    return LLM(
        client,
        model,
        think=s.think,
        temperature=s.temperature,
        seed=s.seed,
        num_ctx=s.num_ctx,
        max_tokens=s.max_tokens,
    )


def _grader(cfg: ExternalConfig, task: ExternalTask) -> str:
    if cfg.judge_model and isinstance(task, JudgedTask):
        return f"judge:{cfg.judge_model}"
    return "string"


def _meta(
    cfg: ExternalConfig, task: ExternalTask, n_items: int, server_version: str | None
) -> dict[str, Any]:
    settings = asdict(cfg.settings)
    settings.pop("ollama_api_key", None)
    settings["cache_dir"] = _tilde(settings["cache_dir"]) if settings["cache_dir"] else None
    info = task.info
    files = [
        {
            "url": f.url,
            "path": _tilde(f.path),
            "bytes": f.bytes,
            "sha256": f.sha256,
            "matches_pinned_sha256": f.verified,
        }
        for f in getattr(task, "files", [])
    ]
    return {
        "suite": info.name,
        "title": info.title,
        "source_url": info.source_url,
        "license": info.license,
        "citation": info.citation,
        "live": info.live,
        "models": list(cfg.models),
        "modes": list(cfg.modes),
        "repeats": cfg.repeats,
        "sample": cfg.sample,
        "default_sample": info.default_sample,
        "n_items": n_items,
        "seed": cfg.seed,
        "data_dir": _tilde(cfg.data_dir),
        "dataset_files": files,
        "grader": _grader(cfg, task),
        "judge_model": cfg.judge_model if isinstance(task, JudgedTask) else None,
        "ollamasearch_version": __version__,
        "ollama_server_version": server_version,
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": settings,
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def _update_meta(meta_path: Path, fresh: dict[str, Any]) -> None:
    """Create meta.json, or on resume merge in new models/modes and the grader."""
    if not meta_path.exists():
        _write_json(meta_path, fresh)
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("suite") != fresh["suite"]:
        raise ValueError(
            f"{meta_path.parent} holds a {meta.get('suite')!r} run; use another --out for "
            f"{fresh['suite']!r}"
        )
    for key in ("models", "modes"):
        meta[key] = list(dict.fromkeys([*meta.get(key, []), *fresh[key]]))
    meta["n_items"] = max(int(meta.get("n_items") or 0), fresh["n_items"])
    if fresh["grader"] != "string":
        meta["grader"], meta["judge_model"] = fresh["grader"], fresh["judge_model"]
    _write_json(meta_path, meta)


def run_external(
    cfg: ExternalConfig,
    client: ChatClient,
    *,
    progress: ProgressFn | None = None,
    server_version: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) one external task; returns the summary dict.

    Same output layout as SearchBench (``meta.json``, ``runs.jsonl``, ``summary.json``,
    ``report.md``). Rows carry ``score`` from the task's ``score()`` and the full answer trace.
    With ``cfg.judge_model`` and a task that supports it, a grading pass runs after all
    answers (one model in VRAM at a time) and rewrites the affected rows' scores.
    """
    for mode in cfg.modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    task = get_task(cfg.task)
    items = task.load(cfg.data_dir, cfg.sample, cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = cfg.out_dir / "runs.jsonl"
    _update_meta(cfg.out_dir / "meta.json", _meta(cfg, task, len(items), server_version))
    if cfg.judge_model and not isinstance(task, JudgedTask):
        log.warning("%s has no LLM grader; ignoring --judge-model", cfg.task)

    done = {r["key"] for r in read_rows(runs_path)}
    total = len(cfg.models) * len(cfg.modes) * len(items) * cfg.repeats
    count = 0
    s = cfg.settings
    with _resources(cfg, task) as res, runs_path.open("a", encoding="utf-8") as out:
        for model in cfg.models:
            llm = _llm(client, model, s)
            ran_any = False
            for mode in cfg.modes:
                for item in items:
                    for repeat in range(cfg.repeats):
                        count += 1
                        key = row_key(model, mode, item.id, repeat)
                        if key in done:
                            continue
                        ran_any = True
                        answer = _pipeline(mode, llm, item, res, s).run(item.question)
                        row = {
                            "key": key,
                            "suite": cfg.task,
                            "model": model,
                            "mode": mode,
                            "question_id": item.id,
                            "repeat": repeat,
                            "category": item.category,
                            "lang": item.lang,
                            "question": item.question,
                            "answers": list(item.answers),
                            "score": {**task.score(item, answer), "grader": "string"},
                            "answer": answer.to_dict(),
                            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                        }
                        out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                        out.flush()
                        if progress is not None:
                            progress(count, total, row)
            if ran_any:
                llm.unload()  # free VRAM before the next model (6 GB cards)

    if cfg.judge_model and isinstance(task, JudgedTask):
        judge_rows(cfg.out_dir, task, items, _llm(client, cfg.judge_model, s))
    return finalize_external(cfg.out_dir)


def judge_rows(out_dir: Path, task: JudgedTask, items: Sequence[ExternalItem], judge: LLM) -> int:
    """Grade every row of ``items`` not yet graded by ``judge``; rewrite runs.jsonl.

    Returns the number of rows graded. Failed grader calls leave the row's string grade.
    """
    runs_path = out_dir / "runs.jsonl"
    rows = read_rows(runs_path)
    by_id = {it.id: it for it in items}
    grader = f"judge:{judge.model}"
    graded = 0
    for row in rows:
        item = by_id.get(row.get("question_id", ""))
        if item is None or row["score"].get("grader") == grader:
            continue
        prompt = task.judge_prompt(item, row["answer"]["text"])
        try:
            verdict = judge.chat([{"role": "user", "content": prompt}]).content.strip()
        except Exception as exc:
            log.warning("grader call failed for %s: %s", row["key"], exc)
            continue
        row["score"] = {
            **task.apply_judge(row["score"], verdict),
            "grader": grader,
            "judge_verdict": verdict[:200],
        }
        graded += 1
    if graded:
        judge.unload()
        tmp = runs_path.with_suffix(".tmp")
        tmp.write_text(
            "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
            encoding="utf-8",
        )
        tmp.replace(runs_path)
    log.info("graded %d rows with %s", graded, judge.model)
    return graded


# --- summary --------------------------------------------------------------------------------


def summarize_external(
    rows: Sequence[Mapping[str, Any]], task: ExternalTask | None
) -> dict[str, Any]:
    """Aggregate rows by (model, mode): accuracy, latency, throughput, task metrics."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mode"])].append(row)
    out: list[dict[str, Any]] = []
    for (model, mode), items in sorted(groups.items()):
        answers = [r["answer"] for r in items]
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for r in items:
            by_cat[r["category"]].append(bool(r["score"]["correct"]))
        totals = [a["timings"].get("total_s", 0.0) for a in answers]
        eval_tokens = sum(a["tokens"].get("eval_tokens", 0) for a in answers)
        eval_s = sum(a["tokens"].get("eval_s", 0.0) for a in answers)
        entry: dict[str, Any] = {
            "model": model,
            "mode": mode,
            "n": len(items),
            "accuracy": rate(bool(r["score"]["correct"]) for r in items),
            "by_category": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
            "latency_p50_s": percentile(totals, 50),
            "latency_p95_s": percentile(totals, 95),
            "latency_mean_s": sum(totals) / len(totals) if totals else math.nan,
            "tokens_per_s": eval_tokens / eval_s if eval_s else None,
            "mean_prompt_tokens": sum(a["tokens"].get("prompt_tokens", 0) for a in answers)
            / len(answers),
            "mean_eval_tokens": eval_tokens / len(answers),
            "error_rate": rate(a.get("error") is not None for a in answers),
            "metrics": task.summarize(items) if task is not None else {},
        }
        if mode == "agent":
            entry["agent"] = agent_metrics(answers)
        elif mode == "multi":
            entry["multi"] = multi_metrics(answers)
        out.append(entry)
    return {"groups": out}


def finalize_external(out_dir: Path) -> dict[str, Any]:
    """(Re)build summary.json and report.md from runs.jsonl."""
    rows = read_rows(out_dir / "runs.jsonl")
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    task: ExternalTask | None = None
    try:
        task = get_task(str(meta.get("suite", "")))
    except ValueError:
        log.warning("unknown suite in %s; summarising without task metrics", meta_path)
    summary = {"meta": meta, **summarize_external(rows, task)}
    _write_json(out_dir / "summary.json", summary)
    write_report(
        summary,
        out_dir / "report.md",
        metric_labels=getattr(task, "metric_labels", {}),
        protocol_note=getattr(task, "protocol_note", ""),
    )
    return summary

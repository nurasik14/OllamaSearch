"""SearchBench runner: questions x models x modes x repeats -> runs.jsonl + summary + report."""

from __future__ import annotations

import json
import logging
import platform
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import Settings
from ..fetch import Fetcher
from ..llm import LLM, ChatClient
from ..pipelines import MODES, build_pipeline
from ..search import (
    CorpusBackend,
    PageFetcher,
    RecordingSearch,
    SearchBackend,
    SearxngBackend,
)
from .dataset import Question, file_sha256, load_questions
from .report import write_report
from .scoring import score_answer, summarize_rows

log = logging.getLogger(__name__)

TRACKS = ("offline", "live")


@dataclass
class BenchConfig:
    track: str
    models: Sequence[str]
    modes: Sequence[str]
    out_dir: Path
    bench_dir: Path = Path("bench")
    limit: int | None = None
    repeats: int = 1
    categories: Sequence[str] = ()
    settings: Settings = field(default_factory=Settings)

    @property
    def dataset_path(self) -> Path:
        return self.bench_dir / "datasets" / f"{self.track}.jsonl"


def row_key(model: str, mode: str, question_id: str, repeat: int) -> str:
    return f"{model}|{mode}|{question_id}|{repeat}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                log.warning("skipping corrupt line in %s", path)
    return rows


def select_questions(cfg: BenchConfig) -> list[Question]:
    questions = load_questions(cfg.dataset_path)
    if cfg.categories:
        questions = [q for q in questions if q.category in cfg.categories]
    if cfg.limit:
        questions = questions[: cfg.limit]
    return questions


@contextmanager
def track_resources(
    cfg: BenchConfig,
) -> Iterator[tuple[SearchBackend, PageFetcher]]:
    if cfg.track == "offline":
        corpus = CorpusBackend(cfg.bench_dir / "corpus")
        fetcher = Fetcher(transport=corpus.transport(), concurrency=cfg.settings.fetch_concurrency)
        with fetcher:
            yield corpus, fetcher
    elif cfg.track == "live":
        # Same query -> same results for every model in the run (and on resume); empty
        # results from a rate-limited SearXNG are retried instead of being scored.
        backend = RecordingSearch(
            SearxngBackend(cfg.settings.searxng_url, timeout=cfg.settings.search_timeout),
            cfg.out_dir / "search_record.json",
        )
        with Fetcher.from_settings(cfg.settings) as fetcher:
            yield backend, fetcher
    else:
        raise ValueError(f"unknown track {cfg.track!r}; choose from {TRACKS}")


def _tilde(path: object) -> str:
    """Don't publish the local username in committed result files."""
    text, home = str(path), str(Path.home())
    return "~" + text[len(home) :] if text.startswith(home) else text


def _meta(cfg: BenchConfig, server_version: str | None) -> dict[str, Any]:
    settings = asdict(cfg.settings)
    settings.pop("ollama_api_key", None)
    settings["cache_dir"] = _tilde(settings["cache_dir"]) if settings["cache_dir"] else None
    return {
        "track": cfg.track,
        "models": list(cfg.models),
        "modes": list(cfg.modes),
        "repeats": cfg.repeats,
        "limit": cfg.limit,
        "categories": list(cfg.categories),
        "dataset": _tilde(cfg.dataset_path),
        "dataset_sha256": file_sha256(cfg.dataset_path),
        "ollamasearch_version": __version__,
        "ollama_server_version": server_version,
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": settings,
    }


ProgressFn = Callable[[int, int, dict[str, Any]], None]


def run_bench(
    cfg: BenchConfig,
    client: ChatClient,
    *,
    progress: ProgressFn | None = None,
    server_version: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) a benchmark; returns the summary dict."""
    for mode in cfg.modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    questions = select_questions(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = cfg.out_dir / "runs.jsonl"
    meta_path = cfg.out_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(_meta(cfg, server_version), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    done = {r["key"] for r in read_rows(runs_path)}
    total = len(cfg.models) * len(cfg.modes) * len(questions) * cfg.repeats
    count = 0
    s = cfg.settings

    with track_resources(cfg) as (backend, fetcher), runs_path.open("a", encoding="utf-8") as out:
        for model in cfg.models:
            llm = LLM(
                client,
                model,
                think=s.think,
                temperature=s.temperature,
                seed=s.seed,
                num_ctx=s.num_ctx,
                max_tokens=s.max_tokens,
            )
            ran_any = False
            for mode in cfg.modes:
                pipeline = build_pipeline(mode, llm, backend, fetcher, s)
                for q in questions:
                    for repeat in range(cfg.repeats):
                        count += 1
                        key = row_key(model, mode, q.id, repeat)
                        if key in done:
                            continue
                        ran_any = True
                        answer = pipeline.run(q.question)
                        score = score_answer(
                            q,
                            answer.text,
                            cited_urls=answer.cited_urls,
                            context_urls=answer.context_urls,
                            mode=mode,
                        )
                        row = {
                            "key": key,
                            "track": cfg.track,
                            "model": model,
                            "mode": mode,
                            "question_id": q.id,
                            "repeat": repeat,
                            "category": q.category,
                            "lang": q.lang,
                            "question": q.question,
                            "answers": list(q.answers),
                            "score": score.to_dict(),
                            "answer": answer.to_dict(),
                            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        if progress is not None:
                            progress(count, total, row)
            if ran_any:
                llm.unload()  # free VRAM before the next model (6 GB cards)

    return finalize(cfg.out_dir)


def finalize(out_dir: Path) -> dict[str, Any]:
    """(Re)build summary.json and report.md from runs.jsonl."""
    rows = read_rows(out_dir / "runs.jsonl")
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summary = {"meta": meta, **summarize_rows(rows)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    write_report(summary, out_dir / "report.md")
    return summary

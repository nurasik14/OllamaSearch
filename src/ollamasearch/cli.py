"""Command line interface: ask, chat, doctor, bench."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .config import Settings
from .fetch import Fetcher
from .llm import LLM, ChatClient, Message
from .pipelines import MODES, Answer, build_pipeline
from .search import (
    CorpusBackend,
    OllamaWebBackend,
    PageFetcher,
    SearchBackend,
    SearxngBackend,
)

app = typer.Typer(
    help="Private web search for local Ollama models, plus the SearchBench benchmark.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)
BACKENDS = ("searxng", "ollama", "corpus")


class State:
    settings: Settings = Settings()


state = State()


def default_bench_dir() -> Path:
    local = Path("bench")
    if (local / "datasets").is_dir():
        return local
    return Path(__file__).resolve().parents[2] / "bench"


def make_client(settings: Settings) -> ChatClient:
    """Create the Ollama client (tests monkeypatch this with a fake)."""
    import ollama

    # The ollama client waits forever by default; a stalled stream would hang the CLI/bench.
    timeout = httpx.Timeout(settings.llm_timeout, connect=5.0)
    return ollama.Client(host=settings.ollama_host, timeout=timeout)


def ollama_base_url(settings: Settings) -> str:
    host = settings.ollama_host or "http://localhost:11434"
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def http_get(
    url: str, params: dict[str, str] | None = None, timeout: float = 5.0
) -> httpx.Response:
    """Plain GET used by ``doctor`` (tests monkeypatch this)."""
    return httpx.get(url, params=params, timeout=timeout)


def make_search(
    settings: Settings, backend: str, corpus: Path | None = None
) -> tuple[SearchBackend, PageFetcher]:
    if backend == "searxng":
        return (
            SearxngBackend(settings.searxng_url, timeout=settings.search_timeout),
            Fetcher.from_settings(settings),
        )
    if backend == "ollama":
        if not settings.ollama_api_key:
            raise typer.BadParameter("--backend ollama needs OLLAMA_API_KEY (env or .env)")
        web = OllamaWebBackend(settings.ollama_api_key)
        return web, web
    if backend == "corpus":
        corpus_backend = CorpusBackend(corpus or default_bench_dir() / "corpus")
        return corpus_backend, Fetcher(transport=corpus_backend.transport())
    raise typer.BadParameter(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")


def _check_choice(value: str, choices: tuple[str, ...], name: str) -> str:
    if value not in choices:
        raise typer.BadParameter(f"{name} must be one of: {', '.join(choices)}")
    return value


def _print_answer_footer(answer: Answer) -> None:
    if answer.error:
        err.print(f"[red]error:[/red] {escape(answer.error)}")
    for call in answer.tool_calls:
        args = json.dumps(call.arguments, ensure_ascii=False)
        colour = {"ok": "green", "error": "yellow"}.get(call.status, "red")
        console.print(
            f"[dim]  step {call.step}: {call.name}({escape(args)}) -> "
            f"[{colour}]{call.status}[/{colour}] {escape(call.detail)}[/dim]"
        )
    cited = set(answer.cited)
    shown = answer.sources
    if answer.mode in ("agent", "multi"):  # many search hits; list the cited ones if any
        shown = [s for s in answer.sources if s.number in cited] or answer.sources[:5]
    if shown:
        console.print("\n[bold]Sources[/bold]")
        for s in shown:
            mark = "*" if s.number in cited else " "
            console.print(
                f" {mark}[{s.number}] {escape(s.title or s.url)}\n     [blue]{s.url}[/blue]"
            )
    t = answer.tokens
    console.print(
        f"[dim]{answer.mode} · {answer.model} · {answer.timings.get('total_s', 0):.1f}s total · "
        f"{int(t.get('prompt_tokens', 0))} prompt / {int(t.get('eval_tokens', 0))} gen tokens · "
        f"{t.get('tokens_per_s', 0):.0f} tok/s[/dim]"
    )


def _make_llm(settings: Settings, client: ChatClient) -> LLM:
    return LLM(
        client,
        settings.model,
        think=settings.think,
        temperature=settings.temperature,
        seed=settings.seed,
        num_ctx=settings.num_ctx,
        max_tokens=settings.max_tokens,
    )


ModeOpt = Annotated[str, typer.Option("--mode", "-m", help="rag | agent | multi | closed")]
ModelOpt = Annotated[str | None, typer.Option("--model", help="Ollama model (default from env)")]
BackendOpt = Annotated[
    str | None, typer.Option("--backend", "-b", help="searxng | ollama | corpus")
]
CorpusOpt = Annotated[
    Path | None, typer.Option("--corpus", help="Folder of .html pages for --backend corpus")
]
ThinkOpt = Annotated[
    bool | None,
    typer.Option("--think/--no-think", help="Enable model thinking (default: off)"),
]


@app.callback()
def main(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v info, -vv debug")
    ] = 0,
    env_file: Annotated[Path, typer.Option("--env-file", help="dotenv file to read")] = Path(
        ".env"
    ),
) -> None:
    level = logging.WARNING - 10 * min(verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    state.settings = Settings.from_env(dotenv_path=env_file)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"ollamasearch {__version__}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question")],
    mode: ModeOpt = "rag",
    model: ModelOpt = None,
    backend: BackendOpt = None,
    corpus: CorpusOpt = None,
    think: ThinkOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full Answer as JSON")] = False,
) -> None:
    """Answer one question."""
    _check_choice(mode, MODES, "--mode")
    settings = state.settings.with_(model=model, think=think)
    backend_name = _check_choice(backend or settings.backend, BACKENDS, "--backend")
    client = make_client(settings)
    llm = _make_llm(settings, client)
    search, fetcher = (
        (None, None) if mode == "closed" else make_search(settings, backend_name, corpus)
    )
    pipeline = build_pipeline(mode, llm, search, fetcher, settings)

    if as_json:
        answer = pipeline.run(question)
        sys.stdout.write(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2) + "\n")
    else:
        with console.status("searching..." if mode != "closed" else "thinking...") as status:

            def on_token(tok: str) -> None:
                status.stop()
                console.out(tok, end="", highlight=False)

            answer = pipeline.run(question, on_token=on_token)
        console.out("")
        _print_answer_footer(answer)
    if answer.error and not answer.text:
        raise typer.Exit(1)


@app.command()
def chat(
    mode: ModeOpt = "rag",
    model: ModelOpt = None,
    backend: BackendOpt = None,
    corpus: CorpusOpt = None,
    think: ThinkOpt = None,
) -> None:
    """Interactive chat with streaming answers. Commands: /mode X, /clear, /quit."""
    _check_choice(mode, MODES, "--mode")
    settings = state.settings.with_(model=model, think=think)
    backend_name = _check_choice(backend or settings.backend, BACKENDS, "--backend")
    client = make_client(settings)
    llm = _make_llm(settings, client)
    search, fetcher = make_search(settings, backend_name, corpus)
    history: list[Message] = []
    console.print(
        f"[bold]Arika[/bold] · {settings.model} · mode={mode} · backend={backend_name}. "
        "Type /mode rag|agent|closed, /clear or /quit."
    )
    while True:
        try:
            line = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/clear":
            history.clear()
            console.print("[dim]history cleared[/dim]")
            continue
        if line.startswith("/mode"):
            new_mode = line.removeprefix("/mode").strip()
            if new_mode in MODES:
                mode = new_mode
                console.print(f"[dim]mode={mode}[/dim]")
            else:
                console.print(f"[red]modes: {', '.join(MODES)}[/red]")
            continue
        pipeline = build_pipeline(mode, llm, search, fetcher, settings)
        console.print("[bold magenta]arika>[/bold magenta] ", end="")
        try:
            answer = pipeline.run(
                line,
                history=history[-6:],
                on_token=lambda t: console.out(t, end="", highlight=False),
            )
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted[/dim]")
            continue
        console.out("")
        _print_answer_footer(answer)
        history += [
            {"role": "user", "content": line},
            {"role": "assistant", "content": answer.text},
        ]


def _model_names(client: ChatClient) -> list[str]:
    listing: Any = client.list()  # type: ignore[attr-defined]
    models = listing["models"] if isinstance(listing, dict) else listing.models
    names = []
    for m in models:
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            names.append(str(name))
    return names


def _has_model(names: list[str], model: str) -> bool:
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in names or model in names


@app.command()
def doctor() -> None:
    """Check Ollama, the model, SearXNG and optional features; print fixes."""
    s = state.settings
    rows: list[tuple[str, str, str]] = []
    failed = False

    def add(status: str, check: str, detail: str) -> None:
        nonlocal failed
        failed |= status == "fail"
        rows.append((status, check, detail))

    base = ollama_base_url(s)
    try:
        version_info = http_get(f"{base}/api/version").json().get("version", "?")
        client = make_client(s)
        names = _model_names(client)
        add("ok", "Ollama server", f"{base} (version {version_info}, {len(names)} models)")
        if _has_model(names, s.model):
            add("ok", "Model", s.model)
        else:
            add("fail", "Model", f"{s.model} not pulled. Fix: ollama pull {s.model}")
    except Exception as exc:
        add(
            "fail",
            "Ollama server",
            f"{base} unreachable ({exc}). Fix: start it with `ollama serve`",
        )

    try:
        resp = http_get(
            f"{s.searxng_url}/search", params={"q": "ollamasearch", "format": "json"}, timeout=10
        )
        if resp.status_code == 403:
            add(
                "fail",
                "SearXNG",
                "JSON API disabled. Fix: add json to search.formats in settings.yml",
            )
        elif resp.status_code != 200:
            add("fail", "SearXNG", f"HTTP {resp.status_code} from {s.searxng_url}")
        else:
            n = len(resp.json().get("results", []))
            status = "ok" if n else "warn"
            add(status, "SearXNG", f"{s.searxng_url} ({n} results for a test query)")
    except Exception as exc:
        add(
            "fail",
            "SearXNG",
            f"{s.searxng_url} unreachable ({type(exc).__name__}). Fix: "
            "cd deploy/searxng && podman compose up -d "
            "(or set SEARXNG_URL, or use --backend ollama)",
        )

    if s.ollama_api_key:
        add("ok", "OLLAMA_API_KEY", "set: --backend ollama available")
    else:
        add("info", "OLLAMA_API_KEY", "not set (only needed for --backend ollama)")

    try:
        import truststore  # noqa: F401

        add(
            "ok",
            "truststore",
            "OS certificate store used for TLS" if s.use_truststore else "disabled",
        )
    except ImportError:
        add("warn", "truststore", "not installed; falling back to certifi")

    if s.cache_dir:
        try:
            s.cache_dir.mkdir(parents=True, exist_ok=True)
            add("ok", "Page cache", str(s.cache_dir))
        except OSError as exc:
            add("warn", "Page cache", f"{s.cache_dir} not writable: {exc}")
    else:
        add("info", "Page cache", "disabled")

    corpus = default_bench_dir() / "corpus"
    n_pages = len(list(corpus.glob("*.html"))) if corpus.is_dir() else 0
    add("ok" if n_pages else "info", "SearchBench corpus", f"{corpus} ({n_pages} pages)")

    table = Table(title=f"ollamasearch {__version__} doctor", show_lines=False)
    table.add_column("")
    table.add_column("Check")
    table.add_column("Details / fix")
    icon = {
        "ok": "[green]✓[/green]",
        "fail": "[red]✗[/red]",
        "warn": "[yellow]![/yellow]",
        "info": "[blue]i[/blue]",
    }
    for status, check, detail in rows:
        table.add_row(icon[status], check, escape(detail))
    console.print(table)
    if failed:
        raise typer.Exit(1)


@app.command()
def bench(
    suite: Annotated[
        str,
        typer.Option(
            help="searchbench (own) | an external task (hotpotqa, rgb-*, kazqad, simpleqa) | bfcl"
        ),
    ] = "searchbench",
    track: Annotated[str, typer.Option(help="SearchBench track: offline | live")] = "offline",
    models: Annotated[str | None, typer.Option(help="Comma-separated models")] = None,
    modes: Annotated[
        str, typer.Option(help="Comma-separated: closed,rag,agent,multi")
    ] = "closed,rag,agent",
    limit: Annotated[int | None, typer.Option(help="Only the first N questions")] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Runs per question")] = 1,
    categories: Annotated[str | None, typer.Option(help="Comma-separated category filter")] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir (reuse it to resume)")] = None,
    bench_dir: Annotated[
        Path | None, typer.Option(help="Folder with datasets/ and corpus/")
    ] = None,
    think: ThinkOpt = None,
    num_ctx: Annotated[int | None, typer.Option(help="Context window passed to Ollama")] = None,
    report_only: Annotated[
        bool, typer.Option("--report-only", help="Rebuild summary/report from --out")
    ] = False,
    sample: Annotated[
        int | None,
        typer.Option(
            help="External/BFCL: questions to sample (per category for BFCL; rgb-noise runs each "
            "question at 3 noise rates); 0 = all"
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="External/BFCL: sampling seed")] = 42,
    data_dir: Annotated[
        Path | None, typer.Option(help="External/BFCL: dataset download cache")
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(help="External tasks: Ollama model used as the answer grader, if supported"),
    ] = None,
) -> None:
    """Run a benchmark (resumable) and write runs.jsonl, summary.json and report.md.

    SearchBench is this project's own benchmark. External tasks and BFCL are published
    third-party benchmarks; their datasets are downloaded on first use.
    """
    from .bench.runner import TRACKS, BenchConfig, run_bench

    if report_only:
        if out is None:
            raise typer.BadParameter("--report-only needs --out")
        _finalize_any(out)
        console.print(f"rebuilt {out / 'report.md'}")
        return
    if suite != "searchbench":
        _bench_other(
            suite=suite,
            models=models,
            modes=modes,
            out=out,
            repeats=repeats,
            categories=categories,
            sample=sample,
            seed=seed,
            data_dir=data_dir,
            judge_model=judge_model,
            think=think,
            num_ctx=num_ctx,
        )
        return
    _check_choice(track, TRACKS, "--track")
    settings = state.settings.with_(think=think, num_ctx=num_ctx)
    model_list = [m.strip() for m in (models or settings.model).split(",") if m.strip()]
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    for m in mode_list:
        _check_choice(m, MODES, "--modes")
    out_dir = out or Path("bench/results") / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = BenchConfig(
        track=track,
        models=model_list,
        modes=mode_list,
        out_dir=out_dir,
        bench_dir=bench_dir or default_bench_dir(),
        limit=limit,
        repeats=repeats,
        categories=[c.strip() for c in categories.split(",")] if categories else [],
        settings=settings,
    )
    client, server_version = _bench_client(settings, model_list)

    def progress(i: int, total: int, row: dict[str, Any]) -> None:
        ok = "[green]✓[/green]" if row["score"]["correct"] else "[red]✗[/red]"
        secs = row["answer"]["timings"].get("total_s", 0.0)
        text = row["answer"]["text"].replace("\n", " ")[:70]
        console.print(
            f"[dim]{i:>4}/{total}[/dim] {ok} {row['model']} {row['mode']:<6} "
            f"{row['question_id']} {secs:5.1f}s  [dim]{escape(text)}[/dim]"
        )

    console.print(f"SearchBench {track}: {model_list} x {mode_list} -> {out_dir}")
    summary = run_bench(cfg, client, progress=progress, server_version=server_version)
    table = Table(title="SearchBench summary")
    for col in ("model", "mode", "n", "accuracy", "cite hit", "p50 s", "tok/s"):
        table.add_column(col)
    for g in summary["groups"]:
        table.add_row(
            g["model"],
            g["mode"],
            str(g["n"]),
            f"{100 * (g['accuracy'] or 0):.0f}%",
            "–" if g["citation_hit_rate"] is None else f"{100 * g['citation_hit_rate']:.0f}%",
            f"{g['latency_p50_s']:.1f}",
            "–" if not g["tokens_per_s"] else f"{g['tokens_per_s']:.0f}",
        )
    console.print(table)
    console.print(f"report: {out_dir / 'report.md'}")


def _bench_client(settings: Settings, model_list: list[str]) -> tuple[Any, str | None]:
    """Ollama client for a benchmark run: check reachability and models, read the version."""
    client = make_client(settings)
    try:
        names = _model_names(client)
    except Exception as exc:
        err.print(f"[red]cannot reach Ollama: {exc}[/red]  (run `ollamasearch doctor`)")
        raise typer.Exit(1) from exc
    missing = [m for m in model_list if not _has_model(names, m)]
    if missing:
        err.print("[red]missing models:[/red] " + ", ".join(missing))
        err.print("Fix: " + " && ".join(f"ollama pull {m}" for m in missing))
        raise typer.Exit(1)
    server_version = None
    try:
        server_version = http_get(f"{ollama_base_url(settings)}/api/version").json().get("version")
    except Exception:
        logging.getLogger(__name__).debug("could not read Ollama version", exc_info=True)
    return client, server_version


def _finalize_any(out: Path) -> dict[str, Any]:
    """Rebuild the report of any run folder, dispatching on ``meta.json``'s ``suite``."""
    import json

    from .bench.runner import finalize

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    suite = meta.get("suite", "searchbench")
    if suite == "searchbench":
        return finalize(out)
    if suite == "bfcl":
        from .bench.bfcl import finalize_bfcl

        return finalize_bfcl(out)
    from .bench.external.runner import finalize_external

    return finalize_external(out)


def _progress_line(i: int, total: int, row: dict[str, Any]) -> None:
    ok = "[green]✓[/green]" if row.get("score", {}).get("correct") else "[red]✗[/red]"
    label = row.get("category", "") if row.get("mode") in (None, "bfcl") else row["mode"]
    console.print(
        f"[dim]{i:>4}/{total}[/dim] {ok} {row.get('model', '')} {label:<8} "
        f"{row.get('question_id') or row.get('id', '')}"
    )


def _bench_other(
    *,
    suite: str,
    models: str | None,
    modes: str,
    out: Path | None,
    repeats: int,
    categories: str | None,
    sample: int | None,
    seed: int,
    data_dir: Path | None,
    judge_model: str | None,
    think: bool | None,
    num_ctx: int | None,
) -> None:
    """Third-party benchmarks: external QA tasks (through the pipelines) and BFCL."""
    from .bench.external import available_tasks, default_data_dir

    settings = state.settings.with_(think=think, num_ctx=num_ctx)
    model_list = [m.strip() for m in (models or settings.model).split(",") if m.strip()]
    out_dir = out or Path("bench/results") / f"{suite}-{datetime.now():%Y%m%d-%H%M%S}"
    data = data_dir or default_data_dir()
    if suite == "bfcl":
        from .bench.bfcl import BFCL_CATEGORIES, BfclConfig, run_bfcl

        cats = [c.strip() for c in categories.split(",")] if categories else list(BFCL_CATEGORIES)
        for c in cats:
            _check_choice(c, BFCL_CATEGORIES, "--categories")
        client, server_version = _bench_client(settings, model_list)
        console.print(f"BFCL {cats}: {model_list} -> {out_dir}")
        run_bfcl(
            BfclConfig(
                models=model_list,
                out_dir=out_dir,
                categories=cats,
                data_dir=data,
                sample=sample,
                seed=seed,
                settings=settings,
            ),
            client,
            progress=_progress_line,
            server_version=server_version,
        )
        console.print(f"report: {out_dir / 'report.md'}")
        return

    _check_choice(suite, (*available_tasks(), "searchbench", "bfcl"), "--suite")
    from .bench.external.runner import ExternalConfig, run_external

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    for m in mode_list:
        _check_choice(m, MODES, "--modes")
    client, server_version = _bench_client(
        settings, model_list + ([judge_model] if judge_model else [])
    )
    console.print(f"{suite}: {model_list} x {mode_list} -> {out_dir}")
    run_external(
        ExternalConfig(
            task=suite,
            models=model_list,
            modes=mode_list,
            out_dir=out_dir,
            data_dir=data,
            sample=sample,
            seed=seed,
            repeats=repeats,
            judge_model=judge_model,
            settings=settings,
        ),
        client,
        progress=_progress_line,
        server_version=server_version,
    )
    console.print(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":  # pragma: no cover
    app()

# Architecture

`ollamasearch` is one package (`src/ollamasearch/`) with no global state and no work at import time. Every network-facing piece sits behind a small interface, so tests and the offline benchmark swap in fakes without monkeypatching internals.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` frozen dataclass, read from env vars and an optional `.env` (`Settings.from_env`). Real env vars win. |
| `search.py` | `SearchBackend` protocol → `list[SearchResult(title, url, snippet)]`. `SearxngBackend` (pooled httpx client, retries with backoff, clear error for a disabled JSON API), `OllamaWebBackend` (hosted `web_search`/`web_fetch`, needs `OLLAMA_API_KEY`), `CorpusBackend` (BM25 over a folder of HTML, served at `http://corpus.local/<file>` via an `httpx.MockTransport`). Also the `PageFetcher` protocol. |
| `fetch.py` | `Fetcher`: thread pool with global + per-host concurrency limits, connect/read timeouts plus a total deadline, streaming size cap, content-type allow-list, charset detection, optional on-disk cache (extracted text, keyed by URL hash, TTL), OS trust store via `truststore`. Returns `FetchResult(error=...)` and never raises for one bad page. Only `http(s)` URLs are allowed. |
| `extract.py` | HTML → `Page(title, text)` with selectolax/lexbor. Drops script/style/nav/header/footer/aside/form, hidden elements, ARIA landmarks and boilerplate class names (cookie, sidebar, share, ...). Prefers `<main>`, then the largest `<article>`, then `<body>`. Block elements become newlines. |
| `rank.py` | Unicode tokeniser (en/ru/kk stopwords, prefix stemming), pure-Python `BM25`, `split_passages` (~500 chars along paragraph/sentence lines) and `select_passages`, which takes each source's best passage first, then fills by score within total and per-source char budgets. |
| `prompts.py` | Arika system prompts (RAG, closed-book, agent), the step-limit nudge and source formatting. The fixed "not found" sentence makes abstention measurable. |
| `llm.py` | `LLM` wraps any `ChatClient` (the `ollama.Client.chat` shape). It sets temperature/seed/num_ctx and a `num_predict` cap (1024; without it Ollama keeps generating past a full context, so a looping small model never stops), `think=False` by default (retried without the flag for models that reject it), streaming, and `keep_alive=0` unload. `Usage` sums `prompt_eval_count`, `eval_count` and the durations across calls. |
| `pipelines.py` | `ClosedBookPipeline`, `RagPipeline` and `AgentPipeline` share `run(question, history=, on_token=) -> Answer`. |
| `cli.py` | Typer app: `ask`, `chat`, `doctor`, `bench`, `version`. `make_client` builds the Ollama client with a read timeout (the library default is none). `make_client`/`http_get` are the injection points used by the CLI tests. |
| `bench/` | `dataset.py` (JSONL loader + validation), `scoring.py` (normalisation, alias match, abstention, aggregation), `runner.py` (resumable runs), `report.py` (Markdown). |

## Data flow

```
RAG
question ─► SearchBackend.search(k) ─► Fetcher.fetch_many(urls)   (concurrent, per-host limited)
                                            │  extract() per page; failed page → search snippet
                                            ▼
                         select_passages(question, pages, 6000 chars total, 1500/source)
                                            ▼
                prompt "[1] title / URL / passages ..." ─► LLM.stream ─► Answer
                                                                          (text, sources, cited [n],
                                                                           timings, tokens)

Agent (max_steps, default 4)
messages ─► LLM.chat(tools=[web_search, web_fetch])
              ├─ no tool call ──────────────────────────────► final answer
              ├─ native tool_calls, or JSON printed as text ─► validate name + args
              │     web_search(query) → numbered results (global source numbers)
              │     web_fetch(url)    → page's BM25-ranked passages for the question
              │     unknown tool / bad args / duplicate call → error text back to the model
              └─ step limit reached ─► one last call without tools + "answer now" nudge
```

Each tool call becomes a `ToolCallRecord` (step, name, arguments, status `ok|error|invalid`, timing, `from_text`). Source numbers stay stable across the whole agent run, so `[n]` citations map back to URLs.

## Design choices

- **Passages, not pages.** Small models lose the answer in 30 KB of page text. A 6,000-char budget of BM25-selected passages keeps prompts at about 1.5-2.5k tokens.
- **Same code path offline and live.** The offline benchmark only replaces the search backend and the HTTP transport. Extraction, ranking, prompts and scoring are the production code.
- **Robust to small-model tool use.** The agent accepts argument aliases (`q`, `link`, ...), recovers `{"name": ..., "arguments": ...}` printed as text, short-circuits repeated identical calls, and always produces a final answer.
- **Deterministic by default.** temperature 0, seed 42, `think=False`.

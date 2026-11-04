# SearchBench

SearchBench measures **how well a local model uses web search**, not how much it already knows. It runs the same questions through four pipelines and scores the answers automatically:

| Mode | What the model gets |
|---|---|
| `closed` | The question only (baseline: memory and hallucination). |
| `rag` | Top-5 search results, fetched concurrently, cut down to BM25-ranked passages (≤ 6,000 chars, ≤ 1,500 per source), then one answer call. |
| `agent` | Native tool calling with `web_search(query)` and `web_fetch(url)`, up to 4 steps, then a forced final answer. |
| `multi` | A team of single-purpose calls on the same model: planner → searcher + reader per sub-question → re-planner (second hops) → synthesizer → verifier. JSON outputs are schema-constrained, and no tool calls are needed. See [multi-agent.md](multi-agent.md). |

## Tracks

### Offline (deterministic, the headline numbers)

- `bench/corpus/`: 49 small HTML pages about **fictional** towns, companies, people, scientific results, events and institutions. Models can't answer from memory, so the score reflects search and reading.
- `bench/datasets/offline.jsonl`: 42 questions (33 en, 5 ru, 4 kk).
- Search is BM25 over the corpus (`CorpusBackend`). Pages are served at `http://corpus.local/<file>` through an injected `httpx.MockTransport`, so the production fetch → extract → rank path runs with no network. Agent fetches of any other URL get a 404.
- The traps are deliberate:
  - **near-duplicate names** (a town and a company sharing a name, people with similar names in different fields);
  - **outdated vs updated pages**, where the question asks for the current value and the newer page supersedes the older one;
  - **boilerplate-heavy pages**, where big nav menus, cookie banners and sidebars full of other entities' (wrong) numbers surround the real content;
  - **multilingual pages**, one in Kazakh and several in Russian, with questions in kk/ru/en.

Question categories:

| Category | Count | What it tests |
|---|---|---|
| `single-hop` | 14 | Find one page, read one fact. |
| `multi-hop` | 9 | The answer is on a page about an entity that is only named on the first page, so the model must chain two facts. |
| `conflicting` | 6 | Choose the current value over an outdated page. The old value is listed in `outdated` and counted as a *stale* answer. |
| `multilingual` | 8 | ru/kk questions over ru/kk pages. |
| `unanswerable` | 5 | The entity exists (or not) but the fact is nowhere in the corpus. The right answer is to say it wasn't found. |

### Live (noisy, real web)

`bench/datasets/live.jsonl` has 20 time-stable real-world questions (en/ru/kk) answered via SearXNG and real page fetches. Results depend on SearXNG's upstream engines, the day's rankings, site availability and the page cache (`~/.cache/ollamasearch`), so compare models within one run, never across days. Closed-book scores are high here because the facts are well known. The interesting number is whether search *hurts* a model that already knows the answer.

Two infrastructure failures are recorded instead of being scored as model behaviour:

- **Empty search results.** SearXNG returns an empty list when its upstream engines rate-limit it, which tends to happen after a few hundred queries in one run. RAG answers then carry `error: "search: no results"` and agent tool calls get status `error`, so they show up in the report's error rate instead of looking like abstentions.
- **Hung DNS lookups.** httpx timeouts don't cover name resolution, and some hosts made `getaddrinfo` block for ~120 s. `Fetcher.fetch_many` has a hard per-batch deadline (`connect + 2 × read` timeout per concurrency wave) and abandons any page still pending, so one bad host can't stall a question.

## Scoring (`ollamasearch.bench.scoring`)

- **Normalisation**: NFKC, casefold, `ё→е`, thousands separators removed (`1,250` = `1 250` = `1250`), decimal comma → dot (`3,5` = `3.5`), punctuation stripped, whitespace collapsed.
- **Correct (answerable)**: any accepted alias appears in the normalised answer as a whole-token sequence (`13050` does not match `113050`). Aliases include Latin and Cyrillic forms of names.
- **Correct (unanswerable)**: the answer abstains. Abstention is detected with a multilingual phrase list (en/ru/kk: "could not find", "не удалось найти", "табылмады", ...). The prompts ask for an exact not-found sentence, but paraphrases also count.
- **Citation hit** (rag/agent, answerable): the answer cites `[n]` (or prints the URL) of at least one gold page.
- **Gold in context** (rag/agent): a gold page reached the model at all (RAG context, agent search results or fetches). This separates *retrieval* failures from *reading* failures.
- **Stale rate** (conflicting): the answer gives the outdated value and not the current one.
- **False abstention rate**: the model said "not found" on an answerable question.
- **Agent tool metrics**: calls per question, invalid-call rate (unknown tool or bad arguments), searched-at-all rate, fetched-a-page rate, tool calls printed as JSON text instead of native calls, step-limit rate, mean steps.
- **Multi-agent trace** (`multi` only): LLM calls, searches and sub-questions per question, how often the verifier rejected a draft and how often a follow-up round ran. The agent tool metrics are not computed for `multi`.
- **Latency**: p50/p95 of wall-clock seconds per question (search + fetch + generation). **Throughput**: generated tokens / generation time, from Ollama's `eval_count` and `eval_duration`.

## Reproducing

```bash
uv sync --all-extras
uv run ollamasearch doctor

# offline track, all local models, all modes (resumable: re-run with the same --out)
uv run ollamasearch bench --track offline \
  --models qwen3.5:2b,qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b,qwen3.5:latest \
  --modes closed,rag,agent,multi --out bench/results/offline-full

# live track (needs SearXNG)
uv run ollamasearch bench --track live --models qwen3.5:2b,llama3.2:3b --out bench/results/live

# rebuild report.md / summary.json from runs.jsonl, then plot
uv run ollamasearch bench --report-only --out bench/results/offline-full
uv run python bench/plot.py bench/results/offline-full/summary.json --out docs/img
```

Fixed for reproducibility: temperature 0, seed 42, `think=False`, `num_ctx=8192`, dataset SHA-256 recorded in `meta.json`. Models are unloaded (`keep_alive=0`) between models so a 6 GB GPU never holds two at once. Useful flags: `--limit N`, `--categories multi-hop,conflicting`, `--repeats 3`, `--think` (thinking models), `--num-ctx`.

## Results

The committed run is [`bench/results/offline-full/`](../bench/results/offline-full/report.md) (5 models × 4 modes, 840 rows) and [`bench/results/live/`](../bench/results/live/report.md) (closed/rag/agent). The README's [SearchBench results](../README.md#searchbench-results) section summarises both. The same pipelines on published datasets are in [external-benchmarks.md](external-benchmarks.md#results).

## Limitations

- **Small and synthetic.** About 40 offline questions means ±7-8 points of sampling noise on accuracy. Treat differences under about 10 points as ties unless they repeat across `--repeats` or tracks.
- **Multi-hop is a reading test more than a retrieval test.** In a 49-page corpus, BM25 on the question alone ranks the second-hop page in the top 3 for all 9 chains, because generic words like "founded" or "stadium" overlap. One-shot RAG usually sees both pages, so this category mostly measures whether a model chains two facts. Use the agent's `searched`/`fetched` traces to see whether it actually followed the chain.
- **"Hit step limit"** counts runs where the agent used all tool rounds (default 4) and got the forced final call. A search → fetch → search → fetch chain reaches it legitimately.
- **Lexical retrieval.** BM25 with prefix stemming is a fair, cheap baseline but no semantic search. Cross-script questions (Latin question, Cyrillic page) are mostly out of reach by design.
- **String-match scoring.** Alias matching can miss a correct paraphrase ("thirteen thousand fifty") and can reward an answer that lists several candidates including the right one. The stale metric and the trace in `runs.jsonl` help catch the latter.
- **Abstention is phrase-based.** Unusual phrasings may be missed, and an answer that hedges with "not specified" while still guessing counts as an abstention.
- **One prompt per mode.** Results reflect these prompts and tool descriptions. Other prompts can rank models differently.
- **Seeded is not bit-identical.** Ollama's seed and temperature 0 make runs repeatable on one machine and version. GPU kernels and Ollama upgrades can still change outputs.

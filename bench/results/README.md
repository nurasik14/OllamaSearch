# Benchmark results

`ollamasearch bench` writes one folder per run here (`bench/results/<name>/`):

- `meta.json`: models, modes, dataset hash (or dataset URLs and hashes for published suites), settings, Ollama version
- `runs.jsonl`: one row per question × model × mode × repeat, with the full answer trace (sources, tool calls, timings, tokens) and its score
- `summary.json`: aggregated metrics per model × mode
- `report.md`: Markdown tables (rebuild with `ollamasearch bench --report-only --out <folder>`)

Pass the same `--out` folder again to resume an interrupted run.

Committed runs, the evidence behind the README's tables (RTX 4050 6 GB, temperature 0, seed 42, Ollama 0.18.1):

| Folder | Suite | Models × modes | Rows |
|---|---|---|---|
| `offline-full/` | SearchBench offline track, 42 questions | 5 × closed/rag/agent/multi | 840 |
| `live/` | SearchBench live track, 20 questions; `search_record.json` holds the SearXNG results every model saw | 5 × closed/rag/agent | 300 |
| `hotpotqa/` | HotpotQA distractor dev, 60 questions | 4 × closed/rag/agent/multi | 960 |
| `kazqad/` | KazQAD reading comprehension (kk), 60 questions | 4 × closed/rag/agent/multi | 960 |
| `rgb-noise/` | RGB noise robustness, 40 questions × noise 0 / 0.4 / 0.8 | 4 × rag/agent/multi | 1,440 |
| `rgb-negative/` | RGB negative rejection, 40 questions | 4 × rag/agent/multi | 480 |
| `rgb-counterfactual/` | RGB counterfactual robustness, 40 questions | 4 × rag/agent/multi | 480 |
| `simpleqa/` | SimpleQA on the live web, 60 questions; `search_record.json` as above | 4 × closed/rag/agent/multi | 960 |
| `bfcl/` | BFCL v3 non-live Python, 50 items × 5 categories | 5 models | 1,250 |

Two sets of rows were re-run after infrastructure failures (details in `docs/multi-agent.md` and
`docs/external-benchmarks.md`): 9 `offline-full` multi rows that hit an Ollama outage, and 16 `simpleqa` rows that got
empty SearXNG results. Pre-rerun copies are kept locally as `*.bak*` files, which git ignores.

Other run folders (smoke tests, experiments) stay ignored by git.

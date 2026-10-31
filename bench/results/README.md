# Benchmark results

`ollamasearch bench` writes one folder per run here (`bench/results/<name>/`):

- `meta.json`: models, modes, dataset hash, settings, Ollama version
- `runs.jsonl`: one row per question × model × mode × repeat, with the full answer trace (sources, tool calls, timings, tokens) and its score
- `summary.json`: aggregated metrics per model × mode
- `report.md`: Markdown tables

Pass the same `--out` folder again to resume an interrupted run.

Two runs are committed as the evidence behind the README's tables (5 local models, closed/rag/agent, RTX 4050 6 GB):

- `offline-full/`: the offline track (42 questions, 630 runs).
- `live/`: the live track (20 questions, 300 runs). `search_record.json` holds the SearXNG results every model was given, so the RAG inputs can be inspected or replayed.

Other run folders stay ignored by git.

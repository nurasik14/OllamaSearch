# SearchBench report: offline track

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b, qwen3.5:latest
- Modes: closed, rag, agent
- Dataset: `bench/datasets/offline.jsonl` (sha256 c7fd092a2718)
- Repeats: 1, limit: none
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-10-31T18:52:02+00:00

## Overall

| Model | Mode | N | Accuracy | Answerable | Abstains when absent | Citation hit | Gold in context | p50 s | p95 s | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 42 | 45% | 38% | 100% | 8% | 86% | 1.2 | 1.8 | 77 |
| llama3.2:3b | closed | 42 | 12% | 0% | 100% | – | – | 0.2 | 0.6 | 86 |
| llama3.2:3b | multi | 42 | 71% | 70% | 80% | 78% | 97% | 9.5 | 14.7 | 81 |
| llama3.2:3b | rag | 42 | 98% | 100% | 80% | 89% | 100% | 1.0 | 1.5 | 76 |
| qwen2.5:1.5b | agent | 42 | 12% | 0% | 100% | 0% | 0% | 0.2 | 1.8 | 151 |
| qwen2.5:1.5b | closed | 42 | 12% | 0% | 100% | – | – | 0.1 | 1.5 | 159 |
| qwen2.5:1.5b | multi | 42 | 57% | 65% | 0% | 76% | 84% | 11.2 | 16.2 | 151 |
| qwen2.5:1.5b | rag | 42 | 86% | 97% | 0% | 59% | 100% | 0.6 | 1.0 | 149 |
| qwen2.5:3b | agent | 42 | 60% | 59% | 60% | 49% | 92% | 1.5 | 2.5 | 80 |
| qwen2.5:3b | closed | 42 | 12% | 0% | 100% | – | – | 0.2 | 0.2 | 90 |
| qwen2.5:3b | multi | 42 | 88% | 89% | 80% | 95% | 97% | 4.7 | 6.5 | 84 |
| qwen2.5:3b | rag | 42 | 98% | 100% | 80% | 95% | 100% | 1.2 | 2.2 | 78 |
| qwen3.5:2b | agent | 42 | 79% | 81% | 60% | 81% | 95% | 2.7 | 6.8 | 67 |
| qwen3.5:2b | closed | 42 | 2% | 0% | 20% | – | – | 0.7 | 3.4 | 68 |
| qwen3.5:2b | multi | 42 | 79% | 76% | 100% | 78% | 100% | 3.9 | 6.7 | 69 |
| qwen3.5:2b | rag | 42 | 100% | 100% | 100% | 100% | 100% | 1.0 | 1.8 | 68 |
| qwen3.5:latest | agent | 42 | 90% | 89% | 100% | 49% | 100% | 10.0 | 17.3 | 11 |
| qwen3.5:latest | closed | 42 | 7% | 0% | 60% | – | – | 3.0 | 13.0 | 9 |
| qwen3.5:latest | multi | 42 | 100% | 100% | 100% | 100% | 100% | 22.4 | 39.5 | 12 |
| qwen3.5:latest | rag | 42 | 100% | 100% | 100% | 100% | 100% | 4.5 | 6.3 | 12 |

## Accuracy by category

| Model | Mode | single-hop | multi-hop | conflicting | multilingual | unanswerable |
|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 43% | 44% | 17% | 38% | 100% |
| llama3.2:3b | closed | 0% | 0% | 0% | 0% | 100% |
| llama3.2:3b | multi | 71% | 67% | 83% | 62% | 80% |
| llama3.2:3b | rag | 100% | 100% | 100% | 100% | 80% |
| qwen2.5:1.5b | agent | 0% | 0% | 0% | 0% | 100% |
| qwen2.5:1.5b | closed | 0% | 0% | 0% | 0% | 100% |
| qwen2.5:1.5b | multi | 93% | 78% | 67% | 0% | 0% |
| qwen2.5:1.5b | rag | 100% | 100% | 83% | 100% | 0% |
| qwen2.5:3b | agent | 71% | 44% | 50% | 62% | 60% |
| qwen2.5:3b | closed | 0% | 0% | 0% | 0% | 100% |
| qwen2.5:3b | multi | 100% | 100% | 50% | 88% | 80% |
| qwen2.5:3b | rag | 100% | 100% | 100% | 100% | 80% |
| qwen3.5:2b | agent | 100% | 78% | 50% | 75% | 60% |
| qwen3.5:2b | closed | 0% | 0% | 0% | 0% | 20% |
| qwen3.5:2b | multi | 79% | 78% | 67% | 75% | 100% |
| qwen3.5:2b | rag | 100% | 100% | 100% | 100% | 100% |
| qwen3.5:latest | agent | 100% | 89% | 50% | 100% | 100% |
| qwen3.5:latest | closed | 0% | 0% | 0% | 0% | 60% |
| qwen3.5:latest | multi | 100% | 100% | 100% | 100% | 100% |
| qwen3.5:latest | rag | 100% | 100% | 100% | 100% | 100% |

## Accuracy by question language

| Model | Mode | en | kk | ru |
|---|---|---|---|---|
| llama3.2:3b | agent | 48% | 25% | 40% |
| llama3.2:3b | closed | 15% | 0% | 0% |
| llama3.2:3b | multi | 76% | 50% | 60% |
| llama3.2:3b | rag | 97% | 100% | 100% |
| qwen2.5:1.5b | agent | 15% | 0% | 0% |
| qwen2.5:1.5b | closed | 15% | 0% | 0% |
| qwen2.5:1.5b | multi | 73% | 0% | 0% |
| qwen2.5:1.5b | rag | 82% | 100% | 100% |
| qwen2.5:3b | agent | 61% | 75% | 40% |
| qwen2.5:3b | closed | 15% | 0% | 0% |
| qwen2.5:3b | multi | 91% | 100% | 60% |
| qwen2.5:3b | rag | 97% | 100% | 100% |
| qwen3.5:2b | agent | 79% | 100% | 60% |
| qwen3.5:2b | closed | 3% | 0% | 0% |
| qwen3.5:2b | multi | 79% | 75% | 80% |
| qwen3.5:2b | rag | 100% | 100% | 100% |
| qwen3.5:latest | agent | 91% | 100% | 80% |
| qwen3.5:latest | closed | 9% | 0% | 0% |
| qwen3.5:latest | multi | 100% | 100% | 100% |
| qwen3.5:latest | rag | 100% | 100% | 100% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched at all | Fetched a page | Text-JSON calls | Hit step limit |
|---|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | – | 0% |
| qwen2.5:3b | 1.3 | 0% | 90% | 29% | 0% | 2% |
| qwen3.5:2b | 1.9 | 0% | 95% | 60% | 0% | 17% |
| qwen3.5:latest | 1.7 | 0% | 100% | 48% | 0% | 0% |

## Multi-agent trace

| Model | LLM calls / question | Searches / question | Sub-questions | Verifier rejected | Follow-up round |
|---|---|---|---|---|---|
| llama3.2:3b | 6.0 | 3.3 | 2.3 | 10% | 0% |
| qwen2.5:1.5b | 6.5 | 3.5 | 2.5 | 2% | 2% |
| qwen2.5:3b | 4.0 | 2.0 | 1.0 | 2% | 2% |
| qwen3.5:2b | 4.0 | 1.8 | 1.0 | 5% | 0% |
| qwen3.5:latest | 4.8 | 2.4 | 1.4 | 5% | 5% |

## Other

| Model | Mode | Stale answers (conflicting) | False abstentions | Errors | Prompt tok | Gen tok |
|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 17% | 32% | 0% | 1064 | 58 |
| llama3.2:3b | closed | 0% | 92% | 0% | 90 | 10 |
| llama3.2:3b | multi | 17% | 22% | 0% | 4648 | 303 |
| llama3.2:3b | rag | 0% | 0% | 0% | 1372 | 33 |
| qwen2.5:1.5b | agent | 0% | 73% | 0% | 416 | 36 |
| qwen2.5:1.5b | closed | 0% | 76% | 0% | 79 | 23 |
| qwen2.5:1.5b | multi | 17% | 0% | 0% | 5493 | 407 |
| qwen2.5:1.5b | rag | 17% | 0% | 0% | 1416 | 47 |
| qwen2.5:3b | agent | 33% | 19% | 0% | 1590 | 70 |
| qwen2.5:3b | closed | 0% | 100% | 0% | 79 | 8 |
| qwen2.5:3b | multi | 33% | 3% | 0% | 3026 | 163 |
| qwen2.5:3b | rag | 0% | 0% | 0% | 1416 | 47 |
| qwen3.5:2b | agent | 50% | 0% | 0% | 3114 | 139 |
| qwen3.5:2b | closed | 0% | 11% | 0% | 83 | 74 |
| qwen3.5:2b | multi | 0% | 22% | 21% | 2501 | 135 |
| qwen3.5:2b | rag | 0% | 0% | 0% | 1396 | 36 |
| qwen3.5:latest | agent | 50% | 0% | 0% | 2655 | 95 |
| qwen3.5:latest | closed | 0% | 27% | 0% | 83 | 34 |
| qwen3.5:latest | multi | 0% | 0% | 0% | 3620 | 239 |
| qwen3.5:latest | rag | 0% | 0% | 0% | 1396 | 29 |

# SearchBench report: live track

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b, qwen3.5:latest
- Modes: closed, rag, agent
- Dataset: `bench/datasets/live.jsonl` (sha256 e237d6a69f75)
- Repeats: 1, limit: none
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-10-31T20:09:56+00:00

## Overall

| Model | Mode | N | Accuracy | Answerable | Abstains when absent | Citation hit | Gold in context | p50 s | p95 s | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 20 | 75% | 75% | – | 0% | 83% | 2.0 | 22.0 | 83 |
| llama3.2:3b | closed | 20 | 75% | 75% | – | – | – | 0.3 | 0.6 | 89 |
| llama3.2:3b | rag | 20 | 95% | 95% | – | 58% | 83% | 2.2 | 36.7 | 75 |
| qwen2.5:1.5b | agent | 20 | 60% | 60% | – | 0% | 0% | 0.2 | 1.8 | 158 |
| qwen2.5:1.5b | closed | 20 | 50% | 50% | – | – | – | 0.2 | 1.1 | 163 |
| qwen2.5:1.5b | rag | 20 | 100% | 100% | – | 8% | 83% | 2.0 | 38.6 | 151 |
| qwen2.5:3b | agent | 20 | 55% | 55% | – | 17% | 75% | 1.7 | 16.6 | 87 |
| qwen2.5:3b | closed | 20 | 40% | 40% | – | – | – | 0.3 | 1.9 | 86 |
| qwen2.5:3b | rag | 20 | 85% | 85% | – | 75% | 83% | 3.0 | 36.9 | 80 |
| qwen3.5:2b | agent | 20 | 85% | 85% | – | 42% | 92% | 3.6 | 26.8 | 70 |
| qwen3.5:2b | closed | 20 | 85% | 85% | – | – | – | 0.5 | 1.6 | 72 |
| qwen3.5:2b | rag | 20 | 100% | 100% | – | 58% | 83% | 2.5 | 38.4 | 71 |
| qwen3.5:latest | agent | 20 | 100% | 100% | – | 8% | 42% | 5.5 | 11.7 | 12 |
| qwen3.5:latest | closed | 20 | 100% | 100% | – | – | – | 2.2 | 6.1 | 13 |
| qwen3.5:latest | rag | 20 | 95% | 95% | – | 67% | 83% | 6.8 | 39.3 | 12 |

## Accuracy by category

| Model | Mode | single-hop | multi-hop |
|---|---|---|---|
| llama3.2:3b | agent | 76% | 67% |
| llama3.2:3b | closed | 71% | 100% |
| llama3.2:3b | rag | 94% | 100% |
| qwen2.5:1.5b | agent | 65% | 33% |
| qwen2.5:1.5b | closed | 47% | 67% |
| qwen2.5:1.5b | rag | 100% | 100% |
| qwen2.5:3b | agent | 53% | 67% |
| qwen2.5:3b | closed | 41% | 33% |
| qwen2.5:3b | rag | 88% | 67% |
| qwen3.5:2b | agent | 82% | 100% |
| qwen3.5:2b | closed | 82% | 100% |
| qwen3.5:2b | rag | 100% | 100% |
| qwen3.5:latest | agent | 100% | 100% |
| qwen3.5:latest | closed | 100% | 100% |
| qwen3.5:latest | rag | 100% | 67% |

## Accuracy by question language

| Model | Mode | en | kk | ru |
|---|---|---|---|---|
| llama3.2:3b | agent | 100% | 25% | 50% |
| llama3.2:3b | closed | 92% | 25% | 75% |
| llama3.2:3b | rag | 100% | 75% | 100% |
| qwen2.5:1.5b | agent | 75% | 25% | 50% |
| qwen2.5:1.5b | closed | 67% | 0% | 50% |
| qwen2.5:1.5b | rag | 100% | 100% | 100% |
| qwen2.5:3b | agent | 67% | 25% | 50% |
| qwen2.5:3b | closed | 42% | 25% | 50% |
| qwen2.5:3b | rag | 100% | 75% | 50% |
| qwen3.5:2b | agent | 100% | 50% | 75% |
| qwen3.5:2b | closed | 92% | 50% | 100% |
| qwen3.5:2b | rag | 100% | 100% | 100% |
| qwen3.5:latest | agent | 100% | 100% | 100% |
| qwen3.5:latest | closed | 100% | 100% | 100% |
| qwen3.5:latest | rag | 100% | 100% | 75% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched at all | Fetched a page | Text-JSON calls | Hit step limit |
|---|---|---|---|---|---|---|
| llama3.2:3b | 1.1 | 4% | 90% | 5% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | – | 0% |
| qwen2.5:3b | 0.9 | 5% | 80% | 10% | 0% | 0% |
| qwen3.5:2b | 1.9 | 0% | 75% | 15% | 0% | 35% |
| qwen3.5:latest | 0.5 | 0% | 45% | 0% | 0% | 0% |

## Other

| Model | Mode | Stale answers (conflicting) | False abstentions | Errors | Prompt tok | Gen tok |
|---|---|---|---|---|---|---|
| llama3.2:3b | agent | – | 10% | 0% | 1058 | 65 |
| llama3.2:3b | closed | – | 5% | 0% | 87 | 14 |
| llama3.2:3b | rag | – | 0% | 0% | 2304 | 37 |
| qwen2.5:1.5b | agent | – | 15% | 0% | 413 | 57 |
| qwen2.5:1.5b | closed | – | 10% | 0% | 76 | 27 |
| qwen2.5:1.5b | rag | – | 0% | 0% | 2399 | 41 |
| qwen2.5:3b | agent | – | 25% | 0% | 1275 | 51 |
| qwen2.5:3b | closed | – | 30% | 0% | 76 | 26 |
| qwen2.5:3b | rag | – | 10% | 0% | 2399 | 39 |
| qwen3.5:2b | agent | – | 0% | 0% | 3833 | 384 |
| qwen3.5:2b | closed | – | 0% | 0% | 79 | 37 |
| qwen3.5:2b | rag | – | 0% | 0% | 2168 | 59 |
| qwen3.5:latest | agent | – | 0% | 0% | 1174 | 55 |
| qwen3.5:latest | closed | – | 0% | 0% | 79 | 27 |
| qwen3.5:latest | rag | – | 5% | 0% | 2168 | 36 |

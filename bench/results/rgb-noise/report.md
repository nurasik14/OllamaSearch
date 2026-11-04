# RGB noise robustness (en): ollamasearch report

> **RGB noise robustness (en)**, Chen et al., AAAI 2024. Source: <https://github.com/chen700564/RGB>, license CC BY-NC-SA 4.0. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: rag, agent, multi
- Items: 120 (sample 40, seed 42), repeats 1
- Dataset file: <https://raw.githubusercontent.com/chen700564/RGB/master/data/en_refine.json> (7229738 bytes, sha256 1be6ec4b8536)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T15:52:50+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 120 | 81% | 1.0 | 1.5 | 80 | 0% | 49 |
| llama3.2:3b | multi | 120 | 83% | 10.5 | 15.8 | 80 | 0% | 328 |
| llama3.2:3b | rag | 120 | 97% | 1.0 | 2.4 | 76 | 0% | 49 |
| qwen2.5:1.5b | agent | 120 | 2% | 0.1 | 0.2 | 166 | 0% | 11 |
| qwen2.5:1.5b | multi | 120 | 81% | 8.4 | 11.9 | 153 | 0% | 342 |
| qwen2.5:1.5b | rag | 120 | 93% | 0.6 | 0.9 | 146 | 0% | 43 |
| qwen2.5:3b | agent | 120 | 75% | 1.2 | 2.3 | 79 | 0% | 69 |
| qwen2.5:3b | multi | 120 | 95% | 5.4 | 8.8 | 83 | 0% | 159 |
| qwen2.5:3b | rag | 120 | 96% | 1.2 | 1.8 | 77 | 0% | 51 |
| qwen3.5:2b | agent | 120 | 86% | 2.0 | 21.3 | 67 | 1% | 234 |
| qwen3.5:2b | multi | 120 | 95% | 4.3 | 5.3 | 67 | 0% | 179 |
| qwen3.5:2b | rag | 120 | 97% | 1.2 | 2.2 | 69 | 0% | 61 |

## Accuracy by category

| Model | Mode | noise-0.0 | noise-0.4 | noise-0.8 |
|---|---|---|---|---|
| llama3.2:3b | agent | 98% | 92% | 52% |
| llama3.2:3b | multi | 90% | 90% | 70% |
| llama3.2:3b | rag | 100% | 100% | 90% |
| qwen2.5:1.5b | agent | 2% | 2% | 2% |
| qwen2.5:1.5b | multi | 90% | 85% | 68% |
| qwen2.5:1.5b | rag | 100% | 95% | 85% |
| qwen2.5:3b | agent | 90% | 85% | 50% |
| qwen2.5:3b | multi | 100% | 100% | 85% |
| qwen2.5:3b | rag | 100% | 98% | 90% |
| qwen3.5:2b | agent | 95% | 88% | 75% |
| qwen3.5:2b | multi | 100% | 98% | 88% |
| qwen3.5:2b | rag | 100% | 100% | 90% |

## Task metrics

| Model | Mode | Rejected | Positive passage in context |
|---|---|---|---|
| llama3.2:3b | agent | 2% | 100% |
| llama3.2:3b | multi | 14% | 100% |
| llama3.2:3b | rag | 1% | 100% |
| qwen2.5:1.5b | agent | 90% | 0% |
| qwen2.5:1.5b | multi | 4% | 100% |
| qwen2.5:1.5b | rag | 0% | 100% |
| qwen2.5:3b | agent | 14% | 90% |
| qwen2.5:3b | multi | 2% | 100% |
| qwen2.5:3b | rag | 2% | 100% |
| qwen3.5:2b | agent | 2% | 98% |
| qwen3.5:2b | multi | 0% | 100% |
| qwen3.5:2b | rag | 1% | 100% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 1.3 | 2% | 90% | 32% | 0% |
| qwen3.5:2b | 1.8 | 0% | 98% | 30% | 19% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.3 | 3.6 | 2% | 0% |
| qwen2.5:1.5b | 5.9 | 3.2 | 0% | 0% |
| qwen2.5:3b | 4.2 | 2.0 | 5% | 5% |
| qwen3.5:2b | 4.1 | 2.0 | 2% | 2% |

## How this differs from the official protocol

Same passages as RGB's evalue.py (5 passages, noise rates 0.0/0.4/0.8, seed 2333) and the same `checkanswer` (every gold element contained, case-insensitive), with a rejection (`insufficient information` or our not-found sentence) counted wrong. Differences: RGB pastes all 5 passages into its own prompt; here they form a per-question corpus that rag/agent/multi search (BM25) and rank with our prompts, so a passage can be missed by retrieval; RGB samples at temperature 0.7, we use 0.

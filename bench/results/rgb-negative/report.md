# RGB negative rejection (en): ollamasearch report

> **RGB negative rejection (en)**, Chen et al., AAAI 2024. Source: <https://github.com/chen700564/RGB>, license CC BY-NC-SA 4.0. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: rag, agent, multi
- Items: 40 (sample 40, seed 42), repeats 1
- Dataset file: <https://raw.githubusercontent.com/chen700564/RGB/master/data/en_refine.json> (7229738 bytes, sha256 1be6ec4b8536)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T17:20:32+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 40 | 45% | 0.9 | 1.7 | 81 | 0% | 57 |
| llama3.2:3b | multi | 40 | 57% | 10.2 | 12.2 | 81 | 0% | 298 |
| llama3.2:3b | rag | 40 | 75% | 0.6 | 1.3 | 80 | 0% | 22 |
| qwen2.5:1.5b | agent | 40 | 90% | 0.1 | 0.2 | 167 | 0% | 11 |
| qwen2.5:1.5b | multi | 40 | 10% | 7.9 | 10.5 | 155 | 0% | 343 |
| qwen2.5:1.5b | rag | 40 | 15% | 0.5 | 1.0 | 148 | 0% | 40 |
| qwen2.5:3b | agent | 40 | 60% | 1.3 | 2.9 | 82 | 0% | 75 |
| qwen2.5:3b | multi | 40 | 48% | 4.3 | 6.4 | 84 | 0% | 150 |
| qwen2.5:3b | rag | 40 | 68% | 0.8 | 1.6 | 82 | 0% | 37 |
| qwen3.5:2b | agent | 40 | 52% | 3.8 | 8.9 | 70 | 0% | 227 |
| qwen3.5:2b | multi | 40 | 42% | 4.0 | 12.2 | 70 | 0% | 276 |
| qwen3.5:2b | rag | 40 | 55% | 1.4 | 3.3 | 70 | 0% | 75 |

## Task metrics

| Model | Mode | Rejection rate | RGB exact phrase | Gave the gold answer anyway |
|---|---|---|---|---|
| llama3.2:3b | agent | 45% | 0% | 12% |
| llama3.2:3b | multi | 57% | 0% | 10% |
| llama3.2:3b | rag | 75% | 0% | 5% |
| qwen2.5:1.5b | agent | 90% | 0% | 2% |
| qwen2.5:1.5b | multi | 10% | 0% | 2% |
| qwen2.5:1.5b | rag | 15% | 0% | 2% |
| qwen2.5:3b | agent | 60% | 0% | 0% |
| qwen2.5:3b | multi | 48% | 0% | 0% |
| qwen2.5:3b | rag | 68% | 0% | 0% |
| qwen3.5:2b | agent | 52% | 0% | 2% |
| qwen3.5:2b | multi | 42% | 0% | 5% |
| qwen3.5:2b | rag | 55% | 0% | 0% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 1.4 | 2% | 90% | 45% | 0% |
| qwen3.5:2b | 3.1 | 0% | 98% | 78% | 52% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.0 | 3.7 | 28% | 2% |
| qwen2.5:1.5b | 5.9 | 3.2 | 0% | 0% |
| qwen2.5:3b | 4.1 | 2.0 | 12% | 5% |
| qwen3.5:2b | 4.7 | 2.2 | 30% | 12% |

## How this differs from the official protocol

Noise rate 1.0: all 5 passages are irrelevant, so the right answer is to decline. RGB's evalue.py counts the exact phrase `insufficient information` that its prompt asks for, and reject_evalue.py additionally asks ChatGPT whether the answer addresses the question. Our prompts ask for a different sentence, so **rejection** = RGB's phrase, our not-found sentence or any phrase of our abstention detector; with `--judge-model` the reject_evalue.py prompt runs on that local model instead (`not addressed` -> rejected).

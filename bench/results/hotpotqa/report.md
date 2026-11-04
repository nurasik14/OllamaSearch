# HotpotQA (distractor, dev): ollamasearch report

> **HotpotQA (distractor, dev)**, Yang et al., EMNLP 2018. Source: <https://huggingface.co/datasets/hotpotqa/hotpot_qa>, license CC BY-SA 4.0. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: closed, rag, agent, multi
- Items: 60 (sample 60, seed 42), repeats 1
- Dataset file: <https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet> (27452575 bytes, sha256 c20b638ca82b)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T10:48:25+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 60 | 48% | 1.1 | 1.6 | 81 | 0% | 60 |
| llama3.2:3b | closed | 60 | 8% | 0.2 | 0.6 | 88 | 0% | 12 |
| llama3.2:3b | multi | 60 | 42% | 10.2 | 13.2 | 81 | 0% | 338 |
| llama3.2:3b | rag | 60 | 62% | 0.8 | 1.5 | 79 | 0% | 39 |
| qwen2.5:1.5b | agent | 60 | 8% | 0.1 | 0.7 | 155 | 0% | 22 |
| qwen2.5:1.5b | closed | 60 | 15% | 0.1 | 0.4 | 163 | 0% | 14 |
| qwen2.5:1.5b | multi | 60 | 48% | 8.4 | 11.7 | 155 | 0% | 378 |
| qwen2.5:1.5b | rag | 60 | 67% | 0.6 | 0.9 | 148 | 0% | 55 |
| qwen2.5:3b | agent | 60 | 32% | 1.0 | 2.1 | 83 | 0% | 69 |
| qwen2.5:3b | closed | 60 | 2% | 0.2 | 0.5 | 89 | 0% | 10 |
| qwen2.5:3b | multi | 60 | 42% | 5.5 | 13.0 | 84 | 0% | 227 |
| qwen2.5:3b | rag | 60 | 50% | 0.9 | 1.8 | 81 | 0% | 48 |
| qwen3.5:2b | agent | 60 | 68% | 4.0 | 20.1 | 70 | 2% | 305 |
| qwen3.5:2b | closed | 60 | 27% | 1.1 | 3.3 | 71 | 0% | 89 |
| qwen3.5:2b | multi | 60 | 68% | 3.9 | 9.1 | 70 | 0% | 246 |
| qwen3.5:2b | rag | 60 | 77% | 1.3 | 2.4 | 70 | 0% | 68 |

## Accuracy by category

| Model | Mode | bridge | comparison |
|---|---|---|---|
| llama3.2:3b | agent | 50% | 42% |
| llama3.2:3b | closed | 6% | 17% |
| llama3.2:3b | multi | 38% | 58% |
| llama3.2:3b | rag | 60% | 67% |
| qwen2.5:1.5b | agent | 2% | 33% |
| qwen2.5:1.5b | closed | 8% | 42% |
| qwen2.5:1.5b | multi | 48% | 50% |
| qwen2.5:1.5b | rag | 62% | 83% |
| qwen2.5:3b | agent | 33% | 25% |
| qwen2.5:3b | closed | 2% | 0% |
| qwen2.5:3b | multi | 42% | 42% |
| qwen2.5:3b | rag | 54% | 33% |
| qwen3.5:2b | agent | 62% | 92% |
| qwen3.5:2b | closed | 15% | 75% |
| qwen3.5:2b | multi | 67% | 75% |
| qwen3.5:2b | rag | 73% | 92% |

## Task metrics

| Model | Mode | EM | F1 | Gold paragraph recall | Both gold paragraphs |
|---|---|---|---|---|---|
| llama3.2:3b | agent | 0% | 13% | 87% | 73% |
| llama3.2:3b | closed | 3% | 8% | – | – |
| llama3.2:3b | multi | 0% | 8% | 98% | 95% |
| llama3.2:3b | rag | 5% | 15% | 92% | 83% |
| qwen2.5:1.5b | agent | 0% | 2% | 0% | 0% |
| qwen2.5:1.5b | closed | 8% | 11% | – | – |
| qwen2.5:1.5b | multi | 0% | 11% | 95% | 90% |
| qwen2.5:1.5b | rag | 0% | 11% | 92% | 83% |
| qwen2.5:3b | agent | 2% | 10% | 60% | 52% |
| qwen2.5:3b | closed | 0% | 1% | – | – |
| qwen2.5:3b | multi | 0% | 11% | 91% | 82% |
| qwen2.5:3b | rag | 0% | 8% | 92% | 83% |
| qwen3.5:2b | agent | 0% | 10% | 92% | 83% |
| qwen3.5:2b | closed | 0% | 5% | – | – |
| qwen3.5:2b | multi | 0% | 16% | 94% | 88% |
| qwen3.5:2b | rag | 0% | 11% | 92% | 83% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 1.3 | 16% | 68% | 32% | 2% |
| qwen3.5:2b | 2.6 | 0% | 100% | 63% | 40% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.5 | 3.9 | 18% | 3% |
| qwen2.5:1.5b | 6.2 | 3.4 | 0% | 0% |
| qwen2.5:3b | 4.9 | 2.4 | 17% | 17% |
| qwen3.5:2b | 4.4 | 2.2 | 8% | 8% |

## How this differs from the official protocol

Official: the model reads all 10 paragraphs and outputs a short answer span plus supporting sentences; EM/F1 via hotpot_evaluate_v1.py. Here the 10 paragraphs are a per-question corpus the pipelines search (BM25), fetch and rank themselves, so retrieval is part of the test. Answers are full sentences with citations: EM/F1 are computed on the citation-stripped answer and undercount; **accuracy** (normalised gold contained in the normalised answer; first yes/no token for yes/no questions) is the headline. Supporting-fact prediction is not scored; instead *gold paragraph recall* measures whether the two gold paragraphs reached the model.

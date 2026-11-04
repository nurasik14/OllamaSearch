# SimpleQA (live web): ollamasearch report

> **SimpleQA (live web)**, Wei et al., 2024. Source: <https://github.com/openai/simple-evals>, license MIT. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: closed, rag, agent, multi
- Items: 60 (sample 60, seed 42), repeats 1
- Dataset file: <https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv> (2012910 bytes, sha256 feee3f7e7db3)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T18:10:56+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 60 | 42% | 3.0 | 20.1 | 82 | 0% | 51 |
| llama3.2:3b | closed | 60 | 0% | 0.2 | 0.4 | 87 | 0% | 10 |
| llama3.2:3b | multi | 60 | 40% | 27.3 | 68.5 | 79 | 0% | 362 |
| llama3.2:3b | rag | 60 | 57% | 2.1 | 6.9 | 77 | 0% | 33 |
| qwen2.5:1.5b | agent | 60 | 0% | 0.1 | 0.5 | 166 | 0% | 15 |
| qwen2.5:1.5b | closed | 60 | 0% | 0.1 | 0.4 | 162 | 0% | 13 |
| qwen2.5:1.5b | multi | 60 | 53% | 25.3 | 71.0 | 154 | 3% | 371 |
| qwen2.5:1.5b | rag | 60 | 47% | 5.2 | 20.2 | 155 | 23% | 41 |
| qwen2.5:3b | agent | 60 | 22% | 2.5 | 19.9 | 85 | 0% | 59 |
| qwen2.5:3b | closed | 60 | 0% | 0.2 | 0.2 | 90 | 0% | 9 |
| qwen2.5:3b | multi | 60 | 60% | 11.2 | 28.8 | 82 | 0% | 215 |
| qwen2.5:3b | rag | 60 | 60% | 2.1 | 5.3 | 80 | 0% | 41 |
| qwen3.5:2b | agent | 60 | 58% | 13.6 | 61.4 | 71 | 0% | 182 |
| qwen3.5:2b | closed | 60 | 7% | 0.9 | 1.7 | 71 | 0% | 70 |
| qwen3.5:2b | multi | 60 | 67% | 8.6 | 27.7 | 70 | 0% | 240 |
| qwen3.5:2b | rag | 60 | 57% | 2.6 | 8.3 | 71 | 0% | 64 |

## Accuracy by category

| Model | Mode | Art | Geography | History | Music | Other | Politics | Science and technology | Sports | TV shows | Video games |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 57% | 57% | 100% | 60% | 0% | 43% | 27% | 50% | 0% | 0% |
| llama3.2:3b | closed | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| llama3.2:3b | multi | 43% | 71% | 0% | 80% | 25% | 29% | 55% | 17% | 0% | 0% |
| llama3.2:3b | rag | 57% | 71% | 50% | 60% | 50% | 64% | 55% | 50% | 0% | 50% |
| qwen2.5:1.5b | agent | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| qwen2.5:1.5b | closed | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| qwen2.5:1.5b | multi | 57% | 57% | 50% | 80% | 50% | 57% | 55% | 33% | 50% | 0% |
| qwen2.5:1.5b | rag | 57% | 43% | 100% | 60% | 50% | 36% | 36% | 50% | 50% | 50% |
| qwen2.5:3b | agent | 14% | 29% | 100% | 20% | 0% | 21% | 27% | 0% | 50% | 0% |
| qwen2.5:3b | closed | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| qwen2.5:3b | multi | 86% | 57% | 50% | 100% | 50% | 57% | 45% | 67% | 0% | 50% |
| qwen2.5:3b | rag | 71% | 71% | 50% | 60% | 50% | 79% | 45% | 50% | 0% | 50% |
| qwen3.5:2b | agent | 29% | 86% | 100% | 80% | 50% | 64% | 45% | 67% | 0% | 50% |
| qwen3.5:2b | closed | 14% | 0% | 0% | 0% | 0% | 14% | 0% | 17% | 0% | 0% |
| qwen3.5:2b | multi | 100% | 86% | 50% | 80% | 50% | 64% | 55% | 50% | 50% | 50% |
| qwen3.5:2b | rag | 57% | 71% | 50% | 60% | 50% | 57% | 55% | 50% | 50% | 50% |

## Task metrics

| Model | Mode | Correct | Incorrect | Not attempted | Correct given attempted | F-score | is_given_attempted |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 42% | 15% | 43% | 74% | 53% | 57% |
| llama3.2:3b | closed | 0% | 12% | 88% | 0% | 0% | 12% |
| llama3.2:3b | multi | 40% | 22% | 38% | 65% | 49% | 62% |
| llama3.2:3b | rag | 57% | 23% | 20% | 71% | 63% | 80% |
| qwen2.5:1.5b | agent | 0% | 20% | 80% | 0% | 0% | 20% |
| qwen2.5:1.5b | closed | 0% | 30% | 70% | 0% | 0% | 30% |
| qwen2.5:1.5b | multi | 53% | 42% | 5% | 56% | 55% | 95% |
| qwen2.5:1.5b | rag | 47% | 43% | 10% | 52% | 49% | 90% |
| qwen2.5:3b | agent | 22% | 17% | 62% | 57% | 31% | 38% |
| qwen2.5:3b | closed | 0% | 0% | 100% | 0% | 0% | 0% |
| qwen2.5:3b | multi | 60% | 28% | 12% | 68% | 64% | 88% |
| qwen2.5:3b | rag | 60% | 22% | 18% | 73% | 66% | 82% |
| qwen3.5:2b | agent | 58% | 18% | 23% | 76% | 66% | 77% |
| qwen3.5:2b | closed | 7% | 93% | 0% | 7% | 7% | 100% |
| qwen3.5:2b | multi | 67% | 32% | 2% | 68% | 67% | 98% |
| qwen3.5:2b | rag | 57% | 28% | 15% | 67% | 61% | 85% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 1.0 | 2% | 75% | 12% | 2% |
| qwen3.5:2b | 2.7 | 0% | 100% | 33% | 48% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.4 | 4.2 | 3% | 0% |
| qwen2.5:1.5b | 6.1 | 3.6 | 2% | 2% |
| qwen2.5:3b | 4.5 | 2.2 | 12% | 12% |
| qwen3.5:2b | 4.4 | 2.2 | 10% | 8% |

## How this differs from the official protocol

Official: the model answers from memory and an LLM grader (GPT-4-class) assigns CORRECT / INCORRECT / NOT_ATTEMPTED with the GRADER_TEMPLATE of simpleqa_eval.py. Here rag/agent/multi search the live web through SearXNG (results recorded in search_record.json so every model sees the same results) and closed is the official memory-only setting. Grading: with `--judge-model`, the official template runs on that local model (weaker than the official grader); otherwise a string grader: gold target (or its main part) contained -> CORRECT, abstention -> NOT_ATTEMPTED, else INCORRECT. The string grader misses paraphrases and numbers written differently, so it under-counts CORRECT. Sampling uses our seeded permutation, not simple-evals' `random.Random(0).sample`.

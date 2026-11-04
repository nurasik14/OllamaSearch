# KazQAD reading comprehension (kk, test): ollamasearch report

> **KazQAD reading comprehension (kk, test)**, Yeshpanov et al., LREC-COLING 2024. Source: <https://github.com/IS2AI/KazQAD>, license CC BY-SA 4.0. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: closed, rag, agent, multi
- Items: 60 (sample 60, seed 42), repeats 1
- Dataset file: <https://raw.githubusercontent.com/IS2AI/KazQAD/main/data/reading-comprehension/kazqad-reading-comprehension-v1.0-kk-test.jsonl> (13213212 bytes, sha256 9bcf199095c7)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T11:32:52+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 60 | 8% | 2.9 | 7.4 | 79 | 0% | 243 |
| llama3.2:3b | closed | 60 | 5% | 0.5 | 2.4 | 82 | 0% | 55 |
| llama3.2:3b | multi | 60 | 30% | 12.9 | 29.9 | 79 | 2% | 553 |
| llama3.2:3b | rag | 60 | 52% | 1.3 | 2.6 | 76 | 2% | 58 |
| qwen2.5:1.5b | agent | 60 | 0% | 1.7 | 7.9 | 149 | 0% | 383 |
| qwen2.5:1.5b | closed | 60 | 0% | 1.0 | 1.9 | 151 | 0% | 146 |
| qwen2.5:1.5b | multi | 60 | 8% | 7.8 | 16.2 | 153 | 2% | 442 |
| qwen2.5:1.5b | rag | 60 | 33% | 1.3 | 2.6 | 146 | 2% | 137 |
| qwen2.5:3b | agent | 60 | 18% | 1.2 | 4.1 | 82 | 0% | 104 |
| qwen2.5:3b | closed | 60 | 2% | 0.2 | 1.5 | 84 | 0% | 31 |
| qwen2.5:3b | multi | 60 | 40% | 6.5 | 21.8 | 83 | 2% | 260 |
| qwen2.5:3b | rag | 60 | 35% | 1.5 | 4.0 | 79 | 2% | 83 |
| qwen3.5:2b | agent | 60 | 27% | 4.0 | 14.6 | 66 | 0% | 290 |
| qwen3.5:2b | closed | 60 | 8% | 2.3 | 11.4 | 70 | 0% | 223 |
| qwen3.5:2b | multi | 60 | 50% | 4.8 | 12.8 | 68 | 2% | 306 |
| qwen3.5:2b | rag | 60 | 60% | 1.5 | 5.0 | 68 | 2% | 116 |

## Accuracy by category

| Model | Mode | biology | geography | history-of-kazakhstan | literature | world-history |
|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 18% | 0% | 10% | 0% | 9% |
| llama3.2:3b | closed | 9% | 0% | 0% | 0% | 18% |
| llama3.2:3b | multi | 36% | 44% | 15% | 22% | 45% |
| llama3.2:3b | rag | 73% | 44% | 60% | 22% | 45% |
| qwen2.5:1.5b | agent | 0% | 0% | 0% | 0% | 0% |
| qwen2.5:1.5b | closed | 0% | 0% | 0% | 0% | 0% |
| qwen2.5:1.5b | multi | 27% | 0% | 10% | 0% | 0% |
| qwen2.5:1.5b | rag | 55% | 56% | 35% | 22% | 0% |
| qwen2.5:3b | agent | 18% | 22% | 20% | 11% | 18% |
| qwen2.5:3b | closed | 0% | 11% | 0% | 0% | 0% |
| qwen2.5:3b | multi | 64% | 22% | 30% | 44% | 45% |
| qwen2.5:3b | rag | 64% | 33% | 25% | 33% | 27% |
| qwen3.5:2b | agent | 27% | 33% | 25% | 44% | 9% |
| qwen3.5:2b | closed | 9% | 11% | 5% | 0% | 18% |
| qwen3.5:2b | multi | 73% | 44% | 50% | 33% | 45% |
| qwen3.5:2b | rag | 73% | 56% | 55% | 44% | 73% |

## Task metrics

| Model | Mode | EM | F1 | Stem match | Gold passage in context | Abstained |
|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 2% | 3% | 8% | 42% | 33% |
| llama3.2:3b | closed | 2% | 4% | 5% | – | 2% |
| llama3.2:3b | multi | 0% | 10% | 33% | 98% | 10% |
| llama3.2:3b | rag | 10% | 27% | 55% | 98% | 3% |
| qwen2.5:1.5b | agent | 0% | 0% | 0% | 0% | 10% |
| qwen2.5:1.5b | closed | 0% | 0% | 0% | – | 3% |
| qwen2.5:1.5b | multi | 0% | 3% | 8% | 97% | 3% |
| qwen2.5:1.5b | rag | 0% | 7% | 38% | 98% | 2% |
| qwen2.5:3b | agent | 0% | 4% | 22% | 53% | 32% |
| qwen2.5:3b | closed | 0% | 0% | 2% | – | 78% |
| qwen2.5:3b | multi | 0% | 12% | 42% | 98% | 12% |
| qwen2.5:3b | rag | 0% | 7% | 37% | 98% | 25% |
| qwen3.5:2b | agent | 0% | 4% | 28% | 52% | 7% |
| qwen3.5:2b | closed | 0% | 1% | 8% | – | 3% |
| qwen3.5:2b | multi | 0% | 16% | 53% | 98% | 8% |
| qwen3.5:2b | rag | 0% | 13% | 63% | 98% | 8% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 2.2 | 0% | 97% | 12% | 30% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 0.7 | 0% | 63% | 7% | 0% |
| qwen3.5:2b | 0.7 | 0% | 53% | 12% | 2% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.3 | 4.6 | 3% | 0% |
| qwen2.5:1.5b | 6.0 | 3.9 | 2% | 2% |
| qwen2.5:3b | 4.2 | 2.1 | 8% | 7% |
| qwen3.5:2b | 4.5 | 2.3 | 13% | 12% |

## How this differs from the official protocol

Official reading comprehension gives an extractive reader one gold passage and scores the span with SQuAD-style EM/F1. Here each question's gold passage(s) plus 6 seeded passages of other questions from the same subject form a per-question corpus that the pipelines search and rank, and a generative model answers in a sentence. **accuracy** = some gold answer contained in the normalised answer (NFKC, casefold, Unicode punctuation removed, no article stripping); EM and token F1 are reported on the citation-stripped answer. Kazakh is agglutinative, so *stem match* (each gold token matched on its first 6 letters) is a lenient extra.

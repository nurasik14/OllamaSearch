# RGB counterfactual robustness (en): ollamasearch report

> **RGB counterfactual robustness (en)**, Chen et al., AAAI 2024. Source: <https://github.com/chen700564/RGB>, license CC BY-NC-SA 4.0. Downloaded at run time, not redistributed.

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b
- Modes: rag, agent, multi
- Items: 40 (sample 40, seed 42), repeats 1
- Dataset file: <https://raw.githubusercontent.com/chen700564/RGB/master/data/en_fact.json> (241320 bytes, sha256 92f4b2330ee4)
- Grader: string
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T17:46:03+00:00

## Overall

| Model | Mode | N | Accuracy | p50 s | p95 s | tok/s | Errors | Gen tok |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | agent | 40 | 0% | 0.9 | 1.2 | 81 | 0% | 46 |
| llama3.2:3b | multi | 40 | 0% | 9.5 | 12.3 | 82 | 0% | 307 |
| llama3.2:3b | rag | 40 | 0% | 0.5 | 1.4 | 81 | 0% | 29 |
| qwen2.5:1.5b | agent | 40 | 0% | 0.1 | 0.3 | 162 | 0% | 11 |
| qwen2.5:1.5b | multi | 40 | 0% | 7.3 | 9.4 | 155 | 0% | 312 |
| qwen2.5:1.5b | rag | 40 | 0% | 0.4 | 0.9 | 150 | 0% | 42 |
| qwen2.5:3b | agent | 40 | 0% | 1.1 | 1.6 | 82 | 0% | 62 |
| qwen2.5:3b | multi | 40 | 0% | 4.4 | 5.2 | 84 | 0% | 149 |
| qwen2.5:3b | rag | 40 | 0% | 0.5 | 1.4 | 82 | 0% | 34 |
| qwen3.5:2b | agent | 40 | 0% | 2.0 | 21.8 | 69 | 0% | 492 |
| qwen3.5:2b | multi | 40 | 0% | 3.6 | 4.4 | 70 | 0% | 177 |
| qwen3.5:2b | rag | 40 | 0% | 0.8 | 2.5 | 70 | 0% | 52 |

## Task metrics

| Model | Mode | Error detection (ED) | Error correction (CR, of detected) | True answer given | Repeated the planted answer |
|---|---|---|---|---|---|
| llama3.2:3b | agent | 0% | – | 2% | 80% |
| llama3.2:3b | multi | 0% | – | 12% | 60% |
| llama3.2:3b | rag | 8% | 0% | 5% | 88% |
| qwen2.5:1.5b | agent | 0% | – | 10% | 0% |
| qwen2.5:1.5b | multi | 0% | – | 2% | 90% |
| qwen2.5:1.5b | rag | 0% | – | 0% | 98% |
| qwen2.5:3b | agent | 0% | – | 0% | 90% |
| qwen2.5:3b | multi | 0% | – | 0% | 92% |
| qwen2.5:3b | rag | 0% | – | 0% | 88% |
| qwen3.5:2b | agent | 0% | – | 12% | 72% |
| qwen3.5:2b | multi | 0% | – | 0% | 92% |
| qwen3.5:2b | rag | 0% | – | 0% | 98% |

## Agent tool use

| Model | Calls / question | Invalid calls | Searched | Fetched | Step limit |
|---|---|---|---|---|---|
| llama3.2:3b | 1.0 | 0% | 100% | 0% | 0% |
| qwen2.5:1.5b | 0.0 | – | 0% | 0% | 0% |
| qwen2.5:3b | 1.2 | 0% | 98% | 20% | 0% |
| qwen3.5:2b | 2.3 | 0% | 100% | 42% | 42% |

## Multi-agent trace

| Model | LLM calls / q | Searches / q | Verifier rejected | Follow-up round |
|---|---|---|---|---|
| llama3.2:3b | 6.5 | 3.9 | 8% | 0% |
| qwen2.5:1.5b | 5.5 | 2.8 | 0% | 0% |
| qwen2.5:3b | 4.0 | 2.0 | 0% | 0% |
| qwen3.5:2b | 4.1 | 2.0 | 2% | 2% |

## How this differs from the official protocol

en_fact.json with RGB's processdata for `_fact` (noise 0, correct_rate 0): up to 5 passages that all state the planted wrong answer. RGB's prompt for this ability (instruction_fact.yaml) warns that the documents may contain factual errors; our pipelines keep their normal prompts, so detection is spontaneous and rates are not directly comparable. **Error detection**: RGB checks `factual errors` in the output (evalue.py) and asks ChatGPT (fact_evalue.py); we accept that phrase plus common paraphrases (heuristic), or run the fact_evalue.py prompt on `--judge-model`. **Error correction** (official `correct_rate`) = detected and gives the true answer, divided by detected; a row is *correct* when it is corrected.

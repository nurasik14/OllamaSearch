# BFCL report: Berkeley Function Calling Leaderboard (v3, non-live, Python)

Data and checker: Berkeley Function Calling Leaderboard, Patil, Mao, Yan, Ji, Suresh, Stoica, Gonzalez: The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025. License Apache-2.0. Data: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard · Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html · Code: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

- Models: qwen2.5:1.5b, qwen3.5:2b, qwen2.5:3b, llama3.2:3b, qwen3.5:latest
- Categories: simple, multiple, parallel, parallel_multiple, irrelevance
- Sample: 50 per category, seed 42
- Files: `BFCL_v3_irrelevance.json` (975f51c51f68), `BFCL_v3_multiple.json` (aef168155ebd), `BFCL_v3_parallel.json` (fad375d68295), `BFCL_v3_parallel_multiple.json` (8863ea843323), `BFCL_v3_simple.json` (fbc37b2ad252), `possible_answer/BFCL_v3_multiple.json` (244e00ce9395), `possible_answer/BFCL_v3_parallel.json` (3af54bf04e5b), `possible_answer/BFCL_v3_parallel_multiple.json` (df4d2c16238c), `possible_answer/BFCL_v3_simple.json` (2911a2bc00df)
- Temperature 0, think=False, num_ctx=8192, max_tokens=1024
- ollamasearch 0.2.0, Ollama 0.18.1, started 2026-11-03T09:30:28+00:00

## Overall

| Model | Overall | Native only | AST (non-live) | simple | multiple | parallel | parallel_multiple | irrelevance | Text calls | No call | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | 62% | 62% | 60% | 68% | 78% | 42% | 50% | 70% | 1% | 20% | 149 |
| qwen3.5:2b | 67% | 67% | 66% | 66% | 82% | 50% | 66% | 72% | 0% | 33% | 69 |
| qwen2.5:3b | 77% | 77% | 82% | 90% | 90% | 74% | 72% | 60% | 0% | 13% | 82 |
| llama3.2:3b | 14% | 14% | 18% | 16% | 30% | 16% | 10% | 0% | 4% | 1% | 80 |
| qwen3.5:latest | 89% | 89% | 90% | 96% | 90% | 86% | 86% | 86% | 0% | 18% | 11 |

Overall = unweighted mean over the categories. Native only = the same, counting calls printed as text as failures (what BFCL's function-calling mode would see). AST = mean over simple, multiple, parallel and parallel_multiple (irrelevance: correct = no call).

## By category

| Model | Category | N | Accuracy | Native | Calls native / text / none | Empty replies | Model errors | p50 s | p95 s | tok/s | Gen tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | simple | 50 | 68% | 68% | 43 / 0 / 7 | 4% | 0% | 0.4 | 0.6 | 151 | 42 |
| qwen2.5:1.5b | multiple | 50 | 78% | 78% | 46 / 0 / 4 | 2% | 0% | 0.4 | 0.6 | 150 | 41 |
| qwen2.5:1.5b | parallel | 50 | 42% | 42% | 46 / 0 / 4 | 2% | 0% | 0.7 | 1.4 | 149 | 94 |
| qwen2.5:1.5b | parallel_multiple | 50 | 50% | 50% | 49 / 0 / 1 | 0% | 0% | 0.9 | 1.4 | 146 | 99 |
| qwen2.5:1.5b | irrelevance | 50 | 70% | 70% | 12 / 3 / 35 | 0% | 0% | 0.9 | 2.7 | 148 | 142 |
| qwen3.5:2b | simple | 50 | 66% | 66% | 42 / 0 / 8 | 0% | 0% | 1.1 | 4.9 | 69 | 97 |
| qwen3.5:2b | multiple | 50 | 82% | 82% | 43 / 0 / 7 | 0% | 0% | 1.1 | 4.5 | 69 | 78 |
| qwen3.5:2b | parallel | 50 | 50% | 50% | 28 / 0 / 22 | 0% | 0% | 3.8 | 13.6 | 69 | 320 |
| qwen3.5:2b | parallel_multiple | 50 | 66% | 66% | 40 / 0 / 10 | 0% | 0% | 3.0 | 10.9 | 68 | 240 |
| qwen3.5:2b | irrelevance | 50 | 72% | 72% | 14 / 0 / 36 | 0% | 0% | 1.9 | 15.6 | 69 | 275 |
| qwen2.5:3b | simple | 50 | 90% | 90% | 50 / 0 / 0 | 0% | 0% | 0.7 | 1.0 | 83 | 40 |
| qwen2.5:3b | multiple | 50 | 90% | 90% | 49 / 0 / 1 | 2% | 0% | 0.7 | 0.9 | 83 | 36 |
| qwen2.5:3b | parallel | 50 | 74% | 74% | 48 / 0 / 2 | 4% | 0% | 1.5 | 4.2 | 81 | 127 |
| qwen2.5:3b | parallel_multiple | 50 | 72% | 72% | 50 / 0 / 0 | 0% | 0% | 1.8 | 2.8 | 81 | 117 |
| qwen2.5:3b | irrelevance | 50 | 60% | 60% | 20 / 0 / 30 | 2% | 0% | 1.3 | 6.1 | 82 | 152 |
| llama3.2:3b | simple | 50 | 16% | 16% | 48 / 1 / 1 | 0% | 0% | 0.6 | 1.5 | 80 | 42 |
| llama3.2:3b | multiple | 50 | 30% | 30% | 49 / 1 / 0 | 0% | 0% | 0.6 | 0.8 | 81 | 34 |
| llama3.2:3b | parallel | 50 | 16% | 16% | 47 / 2 / 1 | 0% | 0% | 1.3 | 2.1 | 80 | 90 |
| llama3.2:3b | parallel_multiple | 50 | 10% | 10% | 49 / 1 / 0 | 0% | 0% | 1.5 | 2.4 | 80 | 100 |
| llama3.2:3b | irrelevance | 50 | 0% | 0% | 44 / 6 / 0 | 0% | 0% | 0.5 | 0.8 | 82 | 34 |
| qwen3.5:latest | simple | 50 | 96% | 96% | 50 / 0 / 0 | 0% | 0% | 5.7 | 9.4 | 12 | 62 |
| qwen3.5:latest | multiple | 50 | 90% | 90% | 48 / 0 / 2 | 0% | 0% | 5.3 | 8.4 | 11 | 53 |
| qwen3.5:latest | parallel | 50 | 86% | 86% | 50 / 0 / 0 | 0% | 0% | 14.0 | 24.7 | 11 | 161 |
| qwen3.5:latest | parallel_multiple | 50 | 86% | 86% | 50 / 0 / 0 | 0% | 0% | 15.5 | 24.3 | 11 | 154 |
| qwen3.5:latest | irrelevance | 50 | 86% | 86% | 7 / 0 / 43 | 0% | 0% | 8.9 | 44.2 | 12 | 166 |

## Failure types

| Model | Category | Error type (BFCL checker) | Count |
|---|---|---|---|
| qwen2.5:1.5b | simple | `ast_decoder:decoder_failed` | 7 |
| qwen2.5:1.5b | simple | `simple_function_checker:missing_optional` | 3 |
| qwen2.5:1.5b | simple | `value_error:list/tuple` | 2 |
| qwen2.5:1.5b | simple | `simple_function_checker:missing_required` | 2 |
| qwen2.5:1.5b | simple | `type_error:simple` | 1 |
| qwen2.5:1.5b | simple | `value_error:string` | 1 |
| qwen2.5:1.5b | multiple | `ast_decoder:decoder_failed` | 4 |
| qwen2.5:1.5b | multiple | `simple_function_checker:missing_required` | 2 |
| qwen2.5:1.5b | multiple | `value_error:string` | 2 |
| qwen2.5:1.5b | multiple | `multiple_function_checker:wrong_count` | 1 |
| qwen2.5:1.5b | multiple | `simple_function_checker:missing_optional` | 1 |
| qwen2.5:1.5b | multiple | `value_error:others` | 1 |
| qwen2.5:1.5b | parallel | `parallel_function_checker_no_order:wrong_count` | 13 |
| qwen2.5:1.5b | parallel | `parallel_function_checker_no_order:cannot_find_match` | 12 |
| qwen2.5:1.5b | parallel | `ast_decoder:decoder_failed` | 4 |
| qwen2.5:1.5b | parallel_multiple | `parallel_function_checker_no_order:cannot_find_match` | 13 |
| qwen2.5:1.5b | parallel_multiple | `parallel_function_checker_no_order:wrong_count` | 11 |
| qwen2.5:1.5b | parallel_multiple | `ast_decoder:decoder_failed` | 1 |
| qwen2.5:1.5b | irrelevance | `irrelevance:function_called` | 15 |
| qwen3.5:2b | simple | `ast_decoder:decoder_failed` | 8 |
| qwen3.5:2b | simple | `value_error:others` | 5 |
| qwen3.5:2b | simple | `value_error:string` | 2 |
| qwen3.5:2b | simple | `simple_function_checker:missing_required` | 1 |
| qwen3.5:2b | simple | `simple_function_checker:missing_optional` | 1 |
| qwen3.5:2b | multiple | `ast_decoder:decoder_failed` | 7 |
| qwen3.5:2b | multiple | `value_error:string` | 2 |
| qwen3.5:2b | parallel | `ast_decoder:decoder_failed` | 22 |
| qwen3.5:2b | parallel | `parallel_function_checker_no_order:wrong_count` | 2 |
| qwen3.5:2b | parallel | `parallel_function_checker_no_order:cannot_find_match` | 1 |
| qwen3.5:2b | parallel_multiple | `ast_decoder:decoder_failed` | 10 |
| qwen3.5:2b | parallel_multiple | `parallel_function_checker_no_order:wrong_count` | 5 |
| qwen3.5:2b | parallel_multiple | `parallel_function_checker_no_order:cannot_find_match` | 2 |
| qwen3.5:2b | irrelevance | `irrelevance:function_called` | 14 |
| qwen2.5:3b | simple | `simple_function_checker:missing_optional` | 2 |
| qwen2.5:3b | simple | `value_error:others` | 1 |
| qwen2.5:3b | simple | `value_error:dict_value` | 1 |
| qwen2.5:3b | simple | `value_error:list/tuple` | 1 |
| qwen2.5:3b | multiple | `simple_function_checker:missing_optional` | 2 |
| qwen2.5:3b | multiple | `value_error:list/tuple` | 1 |
| qwen2.5:3b | multiple | `ast_decoder:decoder_failed` | 1 |
| qwen2.5:3b | multiple | `value_error:others` | 1 |
| qwen2.5:3b | parallel | `parallel_function_checker_no_order:cannot_find_match` | 7 |
| qwen2.5:3b | parallel | `parallel_function_checker_no_order:wrong_count` | 4 |
| qwen2.5:3b | parallel | `ast_decoder:decoder_failed` | 2 |
| qwen2.5:3b | parallel_multiple | `parallel_function_checker_no_order:cannot_find_match` | 9 |
| qwen2.5:3b | parallel_multiple | `parallel_function_checker_no_order:wrong_count` | 5 |
| qwen2.5:3b | irrelevance | `irrelevance:function_called` | 20 |
| llama3.2:3b | simple | `type_error:simple` | 35 |
| llama3.2:3b | simple | `value_error:string` | 3 |
| llama3.2:3b | simple | `ast_decoder:decoder_failed` | 1 |
| llama3.2:3b | simple | `simple_function_checker:wrong_func_name` | 1 |
| llama3.2:3b | simple | `simple_function_checker:unexpected_param` | 1 |
| llama3.2:3b | simple | `simple_function_checker:missing_optional` | 1 |
| llama3.2:3b | multiple | `type_error:simple` | 32 |
| llama3.2:3b | multiple | `value_error:string` | 1 |
| llama3.2:3b | multiple | `simple_function_checker:unexpected_param` | 1 |
| llama3.2:3b | multiple | `simple_function_checker:wrong_func_name` | 1 |
| llama3.2:3b | parallel | `parallel_function_checker_no_order:cannot_find_match` | 23 |
| llama3.2:3b | parallel | `parallel_function_checker_no_order:wrong_count` | 18 |
| llama3.2:3b | parallel | `ast_decoder:decoder_failed` | 1 |
| llama3.2:3b | parallel_multiple | `parallel_function_checker_no_order:cannot_find_match` | 40 |
| llama3.2:3b | parallel_multiple | `parallel_function_checker_no_order:wrong_count` | 5 |
| llama3.2:3b | irrelevance | `irrelevance:function_called` | 50 |
| qwen3.5:latest | simple | `value_error:others` | 1 |
| qwen3.5:latest | simple | `value_error:list/tuple` | 1 |
| qwen3.5:latest | multiple | `ast_decoder:decoder_failed` | 2 |
| qwen3.5:latest | multiple | `multiple_function_checker:wrong_count` | 1 |
| qwen3.5:latest | multiple | `value_error:others` | 1 |
| qwen3.5:latest | multiple | `value_error:string` | 1 |
| qwen3.5:latest | parallel | `parallel_function_checker_no_order:wrong_count` | 4 |
| qwen3.5:latest | parallel | `parallel_function_checker_no_order:cannot_find_match` | 3 |
| qwen3.5:latest | parallel_multiple | `parallel_function_checker_no_order:cannot_find_match` | 4 |
| qwen3.5:latest | parallel_multiple | `parallel_function_checker_no_order:wrong_count` | 3 |
| qwen3.5:latest | irrelevance | `irrelevance:function_called` | 7 |

## Notes

Differences from the official leaderboard: only five non-live Python single-turn categories (no Java/JavaScript, live, multi-turn or agentic ones); a seeded sample per category unless `--sample 0`; models are queried through Ollama's native tool calling (chat template of the model), not BFCL's per-model handlers or its prompting format; calls printed as JSON or Python text are accepted and counted separately; schema keys the Ollama Python client cannot carry (`default`, `optional`, `maximum`, nested object `properties`, ...) are moved into the parameter description; the overall score is the unweighted mean of the categories run here. Function names keep BFCL's `.` -> `_` rule; when a model calls the dotted name it read in the prompt, Ollama discards the call (unknown tool) and the reply is empty (*Empty replies*; BFCL would score the same item as a wrong function name). Numbers are comparable in spirit, not identical to leaderboard entries.

# BFCL: function calling

SearchBench's agent mode showed small local models failing at tool use itself: `qwen2.5:1.5b` never called a tool and `llama3.2:3b` searched but never fetched. The **Berkeley Function Calling Leaderboard** (BFCL) is the standard published benchmark for that one skill. Given a user request and one or more function definitions, does the model emit the right call(s), with the right argument names, types and values? And does it stay quiet when no function fits? `ollamasearch bench --suite bfcl` runs a subset of BFCL against local Ollama models and scores the calls with a port of BFCL's own AST checker.

## Source, license, citation

- Data: [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) on the Hugging Face Hub (v3 files), Apache-2.0.
- Code: [ShishirPatil/gorilla › berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), Apache-2.0. `src/ollamasearch/bench/bfcl_checker.py` ports the Python path of `bfcl_eval/eval_checker/ast_eval/ast_checker.py`, and the tool conversion in `bench/bfcl.py` mirrors `bfcl_eval/model_handler/utils.py::convert_to_tool` and `bfcl_eval/utils.py::_func_doc_language_specific_pre_processing`.
- Leaderboard: <https://gorilla.cs.berkeley.edu/leaderboard.html>
- Citation: Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, Joseph E. Gonzalez. *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models.* ICML 2025.

The files are downloaded on first use into `<data-dir>/bfcl/` (default `~/.cache/ollamasearch/datasets/bfcl/`) and never committed. Each run's `meta.json` records every file's URL, size and sha256. If a resumed run sees different files, it logs a warning.

## Categories

These are the five non-live, single-turn Python categories of BFCL v3 (JSON lines; `question` is a list of turns, and the first turn's messages are sent):

| Category | Items | Functions offered | Expected | Checked by |
|---|---|---|---|---|
| `simple` | 400 | 1 | exactly 1 call | `simple_function_checker` |
| `multiple` | 200 | 2–4 | 1 call to the right one | `multiple_function_checker` |
| `parallel` | 200 | 1 | 2–8 calls of it | `parallel_function_checker_no_order` |
| `parallel_multiple` | 200 | 2–4 | 2–5 calls of several | `parallel_function_checker_no_order` |
| `irrelevance` | 240 | 1 (unrelated) | **no** call | correct iff no call |

Files as downloaded on 2026-11-01 (sha256 prefix): `BFCL_v3_simple.json` fbc37b2ad252, `BFCL_v3_multiple.json` aef168155ebd, `BFCL_v3_parallel.json` fad375d68295, `BFCL_v3_parallel_multiple.json` 8863ea843323, `BFCL_v3_irrelevance.json` 975f51c51f68, plus `possible_answer/` for the first four (2911a2bc00df, 244e00ce9395, 3af54bf04e5b, df4d2c16238c).

## Protocol

1. **Tools.** Each item's function docs become Ollama/OpenAI tools exactly as BFCL builds them for function-calling models. The description gets " Note that the provided function is in Python 3 syntax." appended. Names with `.` get `_` (`math.factorial` → `math_factorial`) and are mapped back before checking. Types follow BFCL's `GORILLA_TO_OPENAPI`: `dict` → `object`, `float` → `number` (+ `format: float` and " This is a float type value."), `tuple` → `array`, `any` → `string`, and a missing type → `string`. Nested `properties` and `items` are converted recursively.
2. **Ollama fit.** The Ollama Python client (0.6) validates tools with a pydantic model that silently drops every parameter key except `type`, `items`, `description` and `enum`. In the v3 data that affects 374 `default` values, 49 `optional` flags, a few `maximum`/`required`, and 14 object parameters whose nested `properties` would vanish. BFCL's Gemini handler handles schema-limited APIs by moving those keys into the description ("Default is: …", "Optional: …", "Maximum value: …"). We do the same and append nested `properties` as JSON (`fit_ollama_schema`), so what the model sees is what we record.
3. **Call.** One `LLM.chat(messages, tools=...)` per item, at temperature 0 with the seed, `num_ctx`, `max_tokens` and `think` from the settings, and no system prompt (BFCL's FC mode adds none).
4. **Decode.** Native `tool_calls` are used if present. Small models often print the call instead, so the text is also parsed: JSON objects/arrays (`{"name": …, "arguments"|"parameters": …}`, `<tool_call>` tags, code fences, `{"tool_calls": [...]}`) and BFCL's prompting syntax `[f(a=1), g(b="x")]` (literal keyword arguments only). Each row records `call_source`: `native`, `text_json`, `text_python` or `none`.
5. **Check.** `bfcl_checker.check(category, functions, calls, ground_truth)` returns `(valid, error_type, detail)` with BFCL's error-type strings. It checks the function name, then required params present, no unexpected params, and exact Python types (int is accepted for float; bool is not int). Values must be in the possible-answer list, where `""` means the param may be omitted. Strings are normalised as BFCL does: lowercased, with spaces and `,./-_*^` stripped. Lists are ordered; dicts are compared key by key (one level, as in BFCL); lists of dicts are compared in order. Parallel calls are matched in any order, greedily, like BFCL. With no call at all you get `ast_decoder:decoder_failed`; irrelevance with a call gives `irrelevance:function_called`. The checker reproduces BFCL's quirks on purpose. For example, a `""` among list answers makes the item type check pass, so the value check reports the error instead.
6. **Sampling.** `--sample N` items per category (default 50; `0` = all). Items are drawn with `random.Random(f"bfcl:{seed}:{category}")` and kept in file order, so the same seed gives the same ids. The ids are listed in `meta.json["item_ids"]`.

As a sanity check, a gold call built from the first accepted value of every possible answer passes the checker for 998 of the 1,000 AST items. The two exceptions are dataset quirks that fail in BFCL too: `parallel_multiple_26` has a ground-truth param missing from the function doc, and `parallel_178` has an overlapping ground truth that greedy matching can trip on.

## Output

The usual layout: `meta.json` (`"suite": "bfcl"`, dataset URLs, license, citation, hashes, sample, seed, ids, settings), `runs.jsonl`, `summary.json` and `report.md`. Each row has the key `model|bfcl|<category>|<id>|0`, plus `suite`, `model`, `mode: "bfcl"`, `category`, `id`/`question_id`, `messages`, `score {correct, error_type, detail}`, `calls`, `call_source`, `empty_reply`, `content`, `thinking`, `error`, `latency_s` and `usage` (tokens and timings). Runs resume by skipping keys that are already present, and each model is unloaded after its items.

The summary has one group per model × category (`accuracy`, `native_accuracy`, `n`, `error_types`, `call_sources`, `text_call_rate`, `no_call_rate`, `empty_reply_rate`, `error_rate`, latency p50/p95, tokens/s, mean tokens), plus one `overall` group per model:

- `accuracy`: the **unweighted mean over categories**, as BFCL averages categories rather than items. `item_accuracy` is the per-item rate.
- `ast_accuracy`: the mean over `simple`, `multiple`, `parallel` and `parallel_multiple`, which is BFCL's "non-live AST" figure. `irrelevance_accuracy` is reported separately, as BFCL does with hallucination measurement.
- `native_accuracy`: the same, but calls printed as text count as failures. This is what BFCL's function-calling mode would see, so use it to compare against FC entries on the leaderboard.

## How to run

```bash
uv run ollamasearch bench --suite bfcl --models qwen2.5:1.5b,llama3.2:3b \
    --sample 50 --seed 42 --out bench/results/bfcl-$(date +%Y%m%d)
uv run ollamasearch bench --suite bfcl --categories simple,irrelevance --sample 0 --out ...
uv run ollamasearch bench --report-only --out bench/results/bfcl-...   # rebuild report
```

The full five categories at `--sample 0` are 1,240 short calls per model.

## Differences from the official leaderboard

- **Subset.** Only the five non-live Python single-turn categories. There are no Java/JavaScript, live (user-contributed), multi-turn, memory, web-search or format-sensitivity categories, so the numbers are not BFCL's "Overall".
- **Sampling.** The default is 50 per category. Use `--sample 0` to run all items.
- **Inference path.** Calls go through Ollama's native tool calling and the model's own chat template, not BFCL's per-model handlers (which use vLLM/SGLang plus a handler-specific template, or its prompting format with a system prompt). Ollama's server-side parser decides what counts as a tool call.
- **Text calls are accepted.** Calls printed as text count in `accuracy`; `native_accuracy` excludes them.
- **Schema folding.** Keys the Ollama client can't carry are moved into descriptions, as described above.
- **Dotted names.** BFCL's `.` → `_` rule is kept. When a model calls the *dotted* name it read in the user message (e.g. "use the function 'employee.fetch_data'"), Ollama discards the call because no tool has that name, and the reply comes back empty. These rows are flagged `empty_reply` (the *Empty replies* column) and score `ast_decoder:decoder_failed`. BFCL would score the same item as `wrong_func_name`, so accuracy is unaffected.
- **Temperature.** We use 0; BFCL uses 0.001.

## Results

Run on 2026-11-03: 50 items per category (250 per model), seed 42, temperature 0, `think=False`, Ollama 0.18.1,
RTX 4050 6 GB. Report: [`bench/results/bfcl/report.md`](../bench/results/bfcl/report.md).

| Model | Overall | AST (non-live) | simple | multiple | parallel | parallel_multiple | irrelevance | Text calls | No call |
|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 62% | 60% | 68% | 78% | 42% | 50% | 70% | 1% | 20% |
| `qwen3.5:2b` | 67% | 66% | 66% | 82% | 50% | 66% | 72% | 0% | 33% |
| `qwen2.5:3b` | 77% | 82% | 90% | 90% | 74% | 72% | 60% | 0% | 13% |
| `llama3.2:3b` | 14% | 18% | 16% | 30% | 16% | 10% | 0% | 4% | 1% |
| `qwen3.5:latest` | **89%** | **90%** | 96% | 90% | 86% | 86% | 86% | 0% | 18% |

`Native only` equals `Overall` for every model: text-printed calls never turned a failure into a pass.

- **`qwen2.5:3b` is the best small function caller** (82% AST), ahead of `qwen3.5:2b` (66%), which is the stronger
  search model overall. `qwen3.5:2b` often answers in prose instead of calling on `parallel` items (22 of
  50 with no call).
- **`llama3.2:3b` fails on types, not on picking the function.** Of its 214 failures, 67 are `type_error:simple` on
  `simple`/`multiple` (numbers sent as strings such as `"5"`, or wrapped as `{"type": "integer", "value": 60}`), 86 are
  parallel matching failures, and all 50 irrelevance items got a call anyway. It does emit native calls (47–49 of 50
  per category), so the harness isn't dropping them. BFCL's checker is strict about Python types, which is the protocol.
- **`qwen2.5:1.5b` can call functions** (62% overall, 43–49 native calls per AST category). Its 0% tool use in
  ollamasearch's agent mode comes from the agent system prompt, not from missing tool support (see
  [external-benchmarks.md](external-benchmarks.md#results)).
- 50 items per category gives about ±7 points of standard error per category and ±3 points on the overall mean, so
  the ranking `qwen3.5:latest` > `qwen2.5:3b` > {`qwen3.5:2b`, `qwen2.5:1.5b`} > `llama3.2:3b` is solid but the middle
  pair is a tie.

## Limitations

- A small sample (50 per category) has a ±7 pp standard error around 50% accuracy. Compare models within a run and use `--sample 0` for publishable numbers.
- AST checking says nothing about whether the call would *work*. It only says whether it matches one of the annotated answers. Those annotations miss some valid phrasings, and BFCL's own known false negatives carry over.
- Dict values are compared one level deep and parallel matching is greedy, both exactly as in BFCL.
- Thinking models (e.g. `qwen3.5:2b`) run with the `think` setting from the config (default off). The setting is recorded in `meta.json`, but results with and without thinking are not comparable.
- One generation per item with no retries. A transient Ollama error is recorded as `model_error`, counts as a failure, and is not retried on resume. To retry, delete those rows from `runs.jsonl`.

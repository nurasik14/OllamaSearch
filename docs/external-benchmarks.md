# External benchmarks

SearchBench is this project's own benchmark. To check that its conclusions hold on data built by other people, `ollamasearch bench --suite <task>` also runs **published QA benchmarks** through the *same* pipelines (`closed`, `rag`, `agent`, `multi`), with the same prompts, settings, output layout (`meta.json`, `runs.jsonl`, `summary.json`, `report.md`) and resume behaviour.

| Task | Measures | Items (default sample) | Docs per item | Language |
|---|---|---|---|---|
| `hotpotqa` | multi-hop retrieval + reading among look-alike distractors | 7,405 (100) | 10 Wikipedia paragraphs | en |
| `rgb-noise` | robustness to irrelevant passages (noise 0 / 0.4 / 0.8) | 300 questions × 3 (100 × 3) | 5 passages | en |
| `rgb-negative` | declining when no passage has the answer | 300 (100) | 5 irrelevant passages | en |
| `rgb-counterfactual` | noticing and correcting planted factual errors | 100 (100) | 1–5 counterfactual passages | en |
| `kazqad` | Kazakh reading comprehension with retrieval | 1,927 questions (100) | gold + 6 distractor passages | kk |
| `simpleqa` | short-form factuality, live web search | 4,326 (100) | none (SearXNG) | en |

## How items become a search problem

Tasks with documents get a **private corpus per item**. Each document is rendered as a small HTML page (`<title>` plus `<main>` with one `<p>` per line) and indexed with `CorpusBackend.from_html`. A single `Fetcher` serves the active item's pages through an `httpx.MockTransport`, so `rag`, `agent` and `multi` run the real search → fetch → extract → BM25-rank path with no network, and they can only see that item's documents. Retrieval is part of what's tested: RAG keeps only the top-5 BM25 hits and their best passages (≤ 6,000 chars), and the agent decides what to search and open. Corpora are LRU-cached by item id, so each one is built once per run (≈ 2 ms for 10 paragraphs), not once per model × mode.

`simpleqa` has no documents. It searches the live web through SearXNG with `RecordingSearch`, exactly like SearchBench's live track. Results are stored in `out/search_record.json`, so every model, and any resumed run, sees the same results for the same query.

`closed` mode ignores the documents in every task and gives the memory-only baseline.

## Running

```bash
uv sync --extra datasets          # pyarrow, needed for HotpotQA's Parquet file
uv run ollamasearch bench --suite hotpotqa --models qwen3.5:2b,llama3.2:3b \
    --modes closed,rag,agent,multi --sample 100 --seed 42 --out bench/results/hotpotqa
uv run ollamasearch bench --suite simpleqa --models qwen3.5:2b --modes closed,rag \
    --judge-model <grader-model>  # optional: official LLM grader on a local model
uv run ollamasearch bench --report-only --out bench/results/hotpotqa
```

- `--sample N` picks N items (questions for RGB) with a seeded permutation. `0` means all items, in file order. The sample is **prefix-stable**: `--sample 3` gives the first 3 items of `--sample 100`, so a run can be extended into the same `--out` and only the new items run.
- Datasets download on first use into `--data-dir` (default `~/.cache/ollamasearch/datasets/<task>/`) and are reused after that. `meta.json` records each file's URL, size and sha256, plus whether it matches the hash pinned in the adapter. A mismatch is logged but not fatal, because upstream files can be republished.
- `--judge-model` applies to `simpleqa`, `rgb-negative` and `rgb-counterfactual`. Grading runs as a separate pass **after** all answers, so only one model is in VRAM at a time. It rewrites the affected rows' scores (the string-based result is kept next to it) and is resumable. Rows record `score.grader` (`string` or `judge:<model>`), and so does `meta.json`. Other tasks ignore the option and log a warning.
- Rows have the SearchBench layout plus `suite`: `key` (`model|mode|id|repeat`), `model`, `mode`, `question_id`, `repeat`, `category`, `lang`, `question`, `answers`, `score`, `answer` (full trace), `timestamp`.

The report credits the dataset (authors, license, URL) and has an overall table (accuracy, latency p50/p95, tok/s, error rate), accuracy by category, the task's own metrics, and a "how this differs from the official protocol" note.

## Tasks

### `hotpotqa`: HotpotQA, distractor setting, dev split

- **Source**: <https://huggingface.co/datasets/hotpotqa/hotpot_qa>, file `distractor/validation-00000-of-00001.parquet` (27,452,575 bytes, sha256 `c20b638c…f7c6`, 7,405 questions: 5,918 bridge, 1,487 comparison). **License** CC BY-SA 4.0. **Citation**: Zhilin Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
- **Items**: one per question. The 10 context paragraphs (2 gold, 8 TF-IDF distractors) are the documents. Category = question type (`bridge` / `comparison`).
- **Metrics**:
  - `accuracy` (headline, = *correct*): the answer is normalised with the **official** `normalize_answer` (lowercase, strip punctuation, drop articles a/an/the, collapse whitespace), citation markers `[n]` are stripped first, and the normalised gold must appear as a whole-token sequence. For yes/no questions the **first** yes/no token of the answer must equal the gold, because "no" turns up inside sentences too easily.
  - `EM`, `F1`: exactly `exact_match_score` / `f1_score` of `hotpot_evaluate_v1.py`, including the rule that yes/no/noanswer mismatches score 0, computed on the citation-stripped answer.
  - **Gold paragraph recall**: the share of the gold (supporting-fact) paragraphs whose page reached the model (`answer.context_urls`: RAG context, agent search results and fetched pages). *Both gold paragraphs* is the share of questions where both did.
- **Differences from the official protocol**: officially a reader gets all 10 paragraphs and outputs a short span plus supporting sentences. Here retrieval is part of the test, and the pipelines answer in full sentences with citations, so EM is close to 0 by construction and F1 is diluted. That is why containment accuracy is the headline. Supporting-sentence prediction (SP EM/F1, joint metrics) is not scored; gold paragraph recall stands in for the retrieval half of it.

### RGB: `rgb-noise`, `rgb-negative`, `rgb-counterfactual`

- **Source**: <https://github.com/chen700564/RGB>, files `data/en_refine.json` (7,229,738 bytes, sha256 `1be6ec4b…4bcb`, 300 questions) and `data/en_fact.json` (241,320 bytes, sha256 `92f4b233…c9e5`, 100 questions). **License** CC BY-NC-SA 4.0 (non-commercial). **Citation**: Jiawei Chen, Hongyu Lin, Xianpei Han, Le Sun, *Benchmarking Large Language Models in Retrieval-Augmented Generation*, AAAI 2024.
- **Documents**: built exactly like the authors' `processdata` in `evalue.py`: `passage_num` 5, `neg_num = ceil(5 × noise_rate)`, the first N positive and first M negative passages (topped up from the other side when one runs short), and a shuffle under `random.seed(2333)` for every instance. A test compares our `process()` with a copy of their function across noise rates and passage counts. RGB passages have no titles, so pages are titled "Document k".
- **`rgb-noise`**: the same sampled questions at noise rates 0.0, 0.4 and 0.8 (categories `noise-0.0`, `noise-0.4`, `noise-0.8`; item id `<rgb id>@noise-<rate>`). *Correct* follows `evalue.py`: their `checkanswer` (each gold element must appear, case-insensitive substring; a nested list is a set of aliases of which any may appear), with an answer that contains a rejection counted as wrong (RGB's phrase `insufficient information`, or our pipelines' `I could not find this information.`). Extra metrics: rejection rate, and whether a positive passage reached the model.
- **`rgb-negative`** (noise 1.0, only irrelevant passages): *correct* = rejected. RGB's `evalue.py` looks for the exact phrase `insufficient information` that its own prompt asks for. Our prompts ask for a different not-found sentence, so a rejection is RGB's phrase, our sentence, or any phrase of the SearchBench abstention detector (en/ru/kk). We also report how often RGB's exact phrase appeared (it rarely will, since our prompt doesn't ask for it) and how often the model gave the gold answer anyway. **Needs an LLM judge officially**: `reject_evalue.py` asks ChatGPT whether the answer addresses the question. With `--judge-model` we run that prompt verbatim on the local model and apply their rule (`"not addressed"` in the reply → rejected).
- **`rgb-counterfactual`** (`en_fact.json`, `_fact` branch of `processdata` with noise 0 and correct_rate 0: up to 5 passages, all stating the planted wrong answer):
  - **Error detection rate (ED)**: the model says the documents are wrong. RGB's string check is `"factual errors" in prediction`, and `fact_evalue.py` asks ChatGPT. We accept that phrase plus common paraphrases ("the documents are incorrect", "contains an error", "contradict", ...) with a regex. With `--judge-model`, the `fact_evalue.py` prompt runs verbatim and their rule applies (`"has identified"` or `"Yes"` in the reply).
  - **Error correction rate (CR)**, as in `fact_evalue.py`'s `correct_rate`: among answers that detected the error, the share that also gives the true answer (`checkanswer`). A row is *correct* when it is detected **and** corrected.
  - Also reported: the true-answer rate regardless of detection (their `all_rate`), and the *misled* rate (the answer contains the planted fake answer and not the true one).
- **Differences from the official protocol**: RGB pastes all passages into its own prompt (`instruction.yaml`, or `instruction_fact.yaml` for counterfactuals, which warns that documents may contain errors). We keep our pipelines' prompts and let them search and rank the passages, so a passage can be missed by retrieval, and counterfactual detection is *spontaneous*. Our ED/CR are therefore not directly comparable to the paper's. RGB samples at temperature 0.7; we use our deterministic settings (temperature 0, seed 42). The Chinese and integration (`_int`) subsets are not included.

### `kazqad`: KazQAD reading comprehension, Kazakh test split

- **Source**: the authors' repository <https://github.com/IS2AI/KazQAD>, file `data/reading-comprehension/kazqad-reading-comprehension-v1.0-kk-test.jsonl` (13,213,212 bytes, sha256 `9bcf1990…9d36`). The Hugging Face copy (`issai/kazqad`, `kazqad/kazqad-reading-comprehension-v1.0-kk-test.jsonl.gz`) is **gated** (you have to log in and accept the terms), so the open GitHub copy of the same v1.0 release is downloaded instead. If you put the Hugging Face `.jsonl.gz` in `<data-dir>/kazqad/`, it is used instead. **License** CC BY-SA 4.0. **Citation**: Rustem Yeshpanov, Pavel Efimov, Leonid Boytsov, Ardak Shalkarbayuli, Pavel Braslavski, *KazQAD: Kazakh Open-Domain Question Answering Dataset*, LREC-COLING 2024.
- **Schema**: one row per (question, relevant passage): `id` (`bio0001bio#85927_11_3` = question `bio0001` + passage id), `title`, `context`, `question`, `answers: {text: [...], answer_start: [...]}`. The 2,713 rows cover 1,927 unique questions from the Kazakh national exam (UNT). Subject prefixes are the categories: `kzh` history of Kazakhstan 659, `bio` biology 379, `lit` literature 366, `geo` geography 346, `woh` world history 177.
- **Items**: rows are grouped per question. Its gold passage(s), with answers merged over passages, plus **6 distractor passages** sampled (seeded per question) from *other questions of the same subject* make the item's corpus. A distractor never contains any of the question's normalised answers. `lang = kk`.
- **Metrics**: *correct* = some gold answer appears as a whole-token sequence in the normalised answer. Normalisation is Kazakh-safe: NFKC, casefold, every Unicode punctuation/symbol character → space, whitespace collapsed, no English article removal. EM and token F1 (best over gold answers) are computed on the citation-stripped answer. **Stem match** is a lenient extra for an agglutinative language: each gold token matches an answer token on its first 6 letters, in order (`ұлтабарға` ~ `ұлтабар`). Also reported: whether a gold passage reached the model, and an abstention rate (any phrase from the multilingual detector; noisy on long answers).
- **Differences from the official protocol**: the official reading-comprehension setting gives an extractive reader the gold passage and scores the span with SQuAD-style EM/F1 (paper baseline EM 38.5, F1 54.2). We add retrieval over a small corpus, and the answers are generated sentences, so EM/F1 undercount. Accuracy and stem match are the comparable signals. The full open-domain setting (800k-passage Wikipedia corpus) is not used.

### `simpleqa`: SimpleQA

- **Source**: <https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv> (2,012,910 bytes, sha256 `feee3f7e…c032`, 4,326 questions; columns `metadata` (a Python dict literal with `topic`, `answer_type`, `urls`), `problem`, `answer`). Code and grader: <https://github.com/openai/simple-evals>. **License** MIT. **Citation**: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus, *Measuring short-form factuality in large language models*, 2024.
- **Items**: one per question. Category = `topic`. No documents: `rag`, `agent` and `multi` use live SearXNG (needs `SEARXNG_URL`, default `http://127.0.0.1:8080`), and `closed` is the official memory-only setting.
- **Grading**:
  - default **string grader**: CORRECT if the gold target, or a part the official grader would also accept (without a parenthetical, either side of a colon), appears as whole tokens in the normalised answer (SearchBench normalisation: thousands separators and decimal commas unified). Otherwise NOT_ATTEMPTED if the answer is empty or abstains (multilingual detector), else INCORRECT.
  - `--judge-model M`: the official `GRADER_TEMPLATE`, copied verbatim from `simpleqa_eval.py` (credited in the source; a test checks it byte for byte), runs on the local model M through `LLM.chat`. The grade is parsed as officially (`re.search(r"(A|B|C)")`, default NOT_ATTEMPTED).
- **Metrics** (all official): `is_correct`, `is_incorrect`, `is_not_attempted`, `is_given_attempted`, `accuracy_given_attempted`, F-score (harmonic mean of correct and correct-given-attempted). When a judge was used, the string grader's correct rate is reported alongside for comparison.
- **Differences from the official protocol**: officially the model answers from memory and a GPT-4-class model grades. A small local judge is much weaker: in our smoke run, `qwen3.5:2b` graded a clear "there is no record of…" answer as INCORRECT instead of NOT_ATTEMPTED. The string grader misses paraphrases, units and numbers written differently, so it under-counts CORRECT. Sampling uses our seeded permutation, not simple-evals' `random.Random(0).sample`. The live web changes, and several SimpleQA answers are on Wikipedia pages that SearXNG may or may not surface, so compare models within one run only.

## Results

Run on 2026-11-03/04 with resumable commands like the ones above: `qwen2.5:1.5b`, `qwen3.5:2b`,
`qwen2.5:3b` and `llama3.2:3b`, temperature 0, seed 42, `think=False`, `num_ctx` 8192, Ollama 0.18.1, string grader
(no `--judge-model`), RTX 4050 6 GB. Samples: HotpotQA, KazQAD and SimpleQA 60 questions; RGB 40 questions (× 3 noise
rates for `rgb-noise`). Every number below comes from the committed `bench/results/<task>/report.md`.

Accuracy (%), `closed / rag / agent / multi` (RGB has no closed-book run; for `rgb-negative` accuracy is the
rejection rate, for `rgb-counterfactual` it is detected-and-corrected):

| Model | HotpotQA | KazQAD | RGB noise | RGB negative | RGB counterfactual | SimpleQA |
|---|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 15 / 67 / 8 / 48 | 0 / 33 / 0 / 8 | – / 93 / 2 / 81 | – / 15 / 90 / 10 | – / 0 / 0 / 0 | 0 / 58 / 0 / 55 |
| `qwen3.5:2b` | 27 / 77 / 68 / 68 | 8 / 60 / 27 / 50 | – / 97 / 86 / 95 | – / 55 / 52 / 42 | – / 0 / 0 / 0 | 7 / 57 / 58 / 67 |
| `qwen2.5:3b` | 2 / 50 / 32 / 42 | 2 / 35 / 18 / 40 | – / 96 / 75 / 95 | – / 68 / 60 / 48 | – / 0 / 0 / 0 | 0 / 60 / 22 / 60 |
| `llama3.2:3b` | 8 / 62 / 48 / 42 | 5 / 52 / 8 / 30 | – / 97 / 81 / 83 | – / 75 / 45 / 57 | – / 0 / 0 / 0 | 0 / 57 / 42 / 40 |

Median latency per question: `rag` 0.4–1.5 s on the document tasks and 2.1–4.1 s on live SimpleQA; `agent` 0.1–4.0 s on documents
and up to 13.6 s on SimpleQA (the 1.5B model's 0.1 s is because it never searches); `multi` 3.6–12.9 s on documents and 8.6–27.3 s on SimpleQA.

Task metrics worth reading next to accuracy:

- **HotpotQA**: EM is 0–8% and F1 1–16% across all rows, as expected for full-sentence answers (see the protocol
  note). Gold paragraph recall is 91–98% for every rag and multi row (both gold paragraphs 82–95%), so a RAG accuracy
  of 50–77% is mostly a reading failure, not a retrieval failure. In agent mode recall depends on whether the model
  searches at all: 0% for `qwen2.5:1.5b`, 60% for `qwen2.5:3b`, 87–92% for the others.
- **KazQAD**: the gold passage reached the model in 98% of `llama3.2:3b` RAG and multi runs, yet RAG accuracy is 52%:
  reading Kazakh is the bottleneck. Stem match (lenient for agglutinative suffixes) is 55% for that row.
- **RGB noise**: RAG loses at most 15 points from noise 0 to 0.8 (the worst is `qwen2.5:1.5b`, 100 → 85); agent mode
  loses 20–46 points for the three models that use tools (`llama3.2:3b` 98 → 52).
- **RGB negative**: RGB's exact phrase `insufficient information` never appears (our prompt doesn't ask for it). The
  gold answer leaked into 0–12% of answers even though no passage contained it, so the models sometimes answer from
  memory instead of rejecting.
- **RGB counterfactual**: error detection 0–8% (3 of 480 answers), error correction 0%. Answers repeat the planted
  fake answer in 60–98% of rows for every model and mode, except `qwen2.5:1.5b` agent (0%), which refuses instead. RGB's
  prompt warns the model about factual errors and ours doesn't, so these are spontaneous-detection numbers.
- **SimpleQA**: closed-book, `qwen2.5:1.5b`, `qwen2.5:3b` and `llama3.2:3b` mostly decline (70–100% not attempted),
  while `qwen3.5:2b` answers everything and gets 93% wrong. With rag or multi, correct-given-attempted is 56–73% and the
  F-score 49–67%.

Two run notes:

- **`qwen2.5:1.5b` never calls a tool** in agent mode (0 calls in all 380 agent runs here). A direct probe of Ollama
  outside this code reproduces it: with the agent system prompt it answers the not-found sentence at once, and without
  it it describes a search in prose. Its agent rows measure that, not search skill.
- **Empty SearXNG results on SimpleQA.** `RecordingSearch` stores only non-empty results, so a query that came back
  empty for the first model was searched again for later models. 16 `qwen2.5:1.5b` rows (14 rag, 2 multi) had scored
  `search: no results` while later models got results for the same queries. Those rows were deleted and re-run against
  the recorded results (rag 47% → 58%, multi 53% → 55%; error rate 23% → 0%). A resumed run still treats such rows as
  finished, so check the report's error column after a live run.
- **One KazQAD question has no BM25 hits.** Search over its 7 passages returns nothing, so every model's rag and multi
  row for it ends in `search: no results` (the 2% error rate in those rows). It affects all models
  equally. Two agent rows (`qwen3.5:2b` on HotpotQA and RGB noise) are Ollama 500 errors from a malformed tool call in
  the model's output, and count as failures.

## Limitations

- **Not leaderboard numbers.** Every task is run through our retrieval pipelines and prompts, so scores measure "local model + ollamasearch" on that data, not the model alone under the official protocol. Use them to compare models and modes against each other.
- **Containment scoring** credits an answer that contains the gold among other candidates, or that contradicts itself (e.g. "no information… Scout Tufankjian is Armenian-American" counts for *Armenian*). EM/F1 are reported to show the strict side.
- **Heuristic detectors** (rejection, factual-error flags, abstention) are phrase lists. They are documented, unit-tested, and replaceable by the official LLM prompts with `--judge-model` where the original benchmark used a judge.
- **Licenses**: RGB is CC BY-NC-SA 4.0 (non-commercial); HotpotQA and KazQAD are CC BY-SA 4.0; SimpleQA is MIT. Datasets are downloaded at run time and never committed; reports credit the source.
- **Not included**: HotpotQA fullwiki, RGB Chinese and information-integration subsets, KazQAD open-domain retrieval over the full corpus.

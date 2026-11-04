"""Berkeley Function Calling Leaderboard (BFCL) subset: does the model call tools correctly?

SearchBench's agent mode showed that small models fail at tool use (never calling a tool, or
never fetching). BFCL isolates that skill with a published dataset and AST-based checking.

Protocol: each item's function docs are converted to OpenAI/Ollama tool schemas the way BFCL's
function-calling handlers do (``convert_to_tool`` + the Python-syntax hint), the first turn of
the question is sent with native tools at temperature 0, and the calls are scored with a port
of BFCL's AST checker (:mod:`.bfcl_checker`). Calls that a model prints as JSON (or Python
call syntax) instead of a native tool call are parsed too, and the path is recorded.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import json
import logging
import platform
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .. import __version__
from ..config import Settings
from ..fetch import make_ssl_context
from ..llm import LLM, ChatClient, ChatResult, Message, ToolCall
from .bfcl_checker import CheckResult, check, restore_names, tool_name
from .dataset import file_sha256
from .external import default_data_dir
from .report import _num, _pct, _table
from .runner import _tilde, read_rows
from .scoring import percentile

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, dict[str, Any]], None]

#: BFCL categories supported here (single-turn, Python, AST-checkable).
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
AST_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")

HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO}"
LEADERBOARD_URL = "https://gorilla.cs.berkeley.edu/leaderboard.html"
CODE_URL = "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard"
LICENSE = "Apache-2.0"
CITATION = (
    "Patil, Mao, Yan, Ji, Suresh, Stoica, Gonzalez: The Berkeley Function Calling Leaderboard "
    "(BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025"
)
DATASET_VERSION = "v3"
DEFAULT_SAMPLE = 50

#: BFCL's GORILLA_TO_OPENAPI type mapping (bfcl_eval/constants/type_mappings.py).
GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
}
PYTHON_HINT = " Note that the provided function is in Python 3 syntax."


@dataclass
class BfclConfig:
    models: Sequence[str]
    out_dir: Path
    categories: Sequence[str] = BFCL_CATEGORIES
    data_dir: Path = field(default_factory=default_data_dir)
    sample: int | None = None  # per category; None = default, 0 = all
    seed: int = 42
    settings: Settings = field(default_factory=Settings)


@dataclass(frozen=True)
class BfclItem:
    id: str
    category: str
    messages: list[Message]  # first turn of the question
    functions: list[dict[str, Any]]  # BFCL function docs (original names and types)
    ground_truth: list[dict[str, Any]] | None  # None for irrelevance


# --------------------------------------------------------------------------- data


def data_file(category: str) -> str:
    return f"BFCL_{DATASET_VERSION}_{category}.json"


def _files_for(category: str) -> list[str]:
    """Relative paths (under the dataset repo) needed for one category."""
    name = data_file(category)
    return [name] if category == "irrelevance" else [name, f"possible_answer/{name}"]


def _download(url: str, dest: Path, *, use_truststore: bool = True) -> None:
    log.info("downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=10.0),
        verify=make_ssl_context(use_truststore),
    ) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    tmp.replace(dest)


def ensure_files(
    data_dir: Path, categories: Sequence[str], *, use_truststore: bool = True
) -> dict[str, dict[str, Any]]:
    """Download missing files into ``data_dir/bfcl``; return {relpath: {url, sha256, bytes}}."""
    root = data_dir / "bfcl"
    info: dict[str, dict[str, Any]] = {}
    for category in categories:
        for rel in _files_for(category):
            path = root / rel
            url = f"{HF_BASE}/{rel}"
            if not path.is_file():
                _download(url, path, use_truststore=use_truststore)
            info[rel] = {"url": url, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    return info


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return rows


def load_category(data_dir: Path, category: str) -> list[BfclItem]:
    """Items of one category, in file order (files must already be in ``data_dir/bfcl``)."""
    if category not in BFCL_CATEGORIES:
        raise ValueError(f"unknown BFCL category {category!r}; choose from {BFCL_CATEGORIES}")
    root = data_dir / "bfcl"
    answers: dict[str, list[dict[str, Any]]] = {}
    if category != "irrelevance":
        for row in _read_jsonl(root / "possible_answer" / data_file(category)):
            answers[str(row["id"])] = row["ground_truth"]
    items = []
    for row in _read_jsonl(root / data_file(category)):
        item_id = str(row["id"])
        question = row["question"]
        # v3 stores a list of turns (each a list of messages); single-turn items have one.
        turn = question[0] if question and isinstance(question[0], list) else question
        if category != "irrelevance" and item_id not in answers:
            raise ValueError(f"{category}: no possible answer for {item_id}")
        items.append(
            BfclItem(
                id=item_id,
                category=category,
                messages=[dict(m) for m in turn],
                functions=list(row["function"]),
                ground_truth=answers.get(item_id),
            )
        )
    return items


def sample_items(items: Sequence[BfclItem], sample: int | None, seed: int) -> list[BfclItem]:
    """Deterministic per-category sample, kept in file order. ``0`` = all, ``None`` = default."""
    n = DEFAULT_SAMPLE if sample is None else sample
    if n <= 0 or n >= len(items):
        return list(items)
    category = items[0].category if items else ""
    rng = random.Random(f"bfcl:{seed}:{category}")
    picked = set(rng.sample(range(len(items)), n))
    return [it for i, it in enumerate(items) if i in picked]


# --------------------------------------------------------------------------- tools


def _cast_to_openai_type(properties: dict[str, Any]) -> dict[str, Any]:
    """BFCL's ``_cast_to_openai_type`` with the GORILLA_TO_OPENAPI mapping (in place)."""
    for key, value in properties.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = GORILLA_TO_OPENAPI.get(var_type, "string")

        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast_to_openai_type(value["properties"])
            elif "items" in value:
                items = value["items"]
                items["type"] = GORILLA_TO_OPENAPI.get(items.get("type", "string"), "string")
                if items["type"] == "array" and "items" in items:
                    inner = items["items"]
                    inner["type"] = GORILLA_TO_OPENAPI.get(inner.get("type", "string"), "string")
                elif items["type"] == "object" and "properties" in items:
                    items["properties"] = _cast_to_openai_type(items["properties"])
        properties[key] = value
    return properties


def convert_to_tools(functions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """BFCL function docs -> Ollama/OpenAI ``{"type": "function", "function": ...}`` tools.

    Mirrors BFCL's ``_func_doc_language_specific_pre_processing`` (Python hint appended to the
    description) followed by ``convert_to_tool`` for OpenAI-style models (``.`` -> ``_`` in
    names, ``dict`` -> ``object``, ``float`` -> ``number``, ``tuple`` -> ``array``,
    ``any`` -> ``string``, recursively for nested properties and items).
    """
    tools = []
    funcs: list[dict[str, Any]] = copy.deepcopy([dict(f) for f in functions])
    for func in funcs:
        func["description"] = func.get("description", "") + PYTHON_HINT
        func["name"] = tool_name(str(func["name"]))
        params = func.setdefault("parameters", {})
        params["type"] = "object"
        params["properties"] = _cast_to_openai_type(params.get("properties", {}))
        tools.append({"type": "function", "function": func})
    return tools


# Keys the Ollama Python client's ``Tool`` model silently drops from a parameter, folded into
# the description with the wording BFCL uses for schema-limited APIs (its Gemini handler).
_FOLDED_KEYS = (
    ("default", "Default is: {}."),
    ("optional", "Optional: {}."),
    ("required", "Required: {}."),
    ("maximum", "Maximum value: {}."),
    ("minItems", "Minimum number of items: {}."),
    ("maxItems", "Maximum number of items: {}."),
    ("additionalProperties", "Additional properties: {}."),
)


def fit_ollama_schema(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep what the Ollama client would drop visible to the model.

    ``ollama.Tool`` (0.6) keeps only ``type``/``items``/``description``/``enum`` per parameter.
    Like BFCL's Gemini path, other keys move into the description; ``format`` is kept only for
    strings (floats already say "This is a float type value."). Nested ``properties`` of an
    object parameter are appended as JSON, since the client would discard them entirely.
    """
    out: list[dict[str, Any]] = copy.deepcopy([dict(t) for t in tools])
    for tool in out:
        params = tool["function"]["parameters"]
        for key in [k for k in params if k not in ("type", "properties", "required")]:
            del params[key]
        for prop in params["properties"].values():
            desc = str(prop.get("description", ""))
            for key, text in _FOLDED_KEYS:
                if key in prop:
                    desc += " " + text.format(prop.pop(key))
            if "format" in prop:
                fmt = prop.pop("format")
                if prop.get("type") == "string":
                    desc += f" Format: {fmt}."
            if "properties" in prop:
                desc += " Properties: " + json.dumps(prop.pop("properties"), ensure_ascii=False)
            prop["description"] = desc.strip()
    return out


def item_tools(item: BfclItem) -> list[dict[str, Any]]:
    """The tools sent to Ollama for one item."""
    return fit_ollama_schema(convert_to_tools(item.functions))


# --------------------------------------------------------------------------- text calls

TEXT_SOURCES = ("text_json", "text_python")
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json|python|tool_code)?\s*(.*?)\s*```", re.DOTALL)


def _as_call(data: Any) -> ToolCall | None:
    """``{"name": ..., "arguments"|"parameters": {...}}`` (optionally under ``"function"``)."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("function"), dict) and "name" not in data:
        data = data["function"]
    name = data.get("name")
    if not isinstance(name, str) or not ("arguments" in data or "parameters" in data):
        return None
    args = data.get("arguments", data.get("parameters"))
    if isinstance(args, str):
        with contextlib.suppress(ValueError):
            args = json.loads(args)
    return ToolCall(name=name, arguments=args)


def _json_values(text: str) -> list[Any]:
    """Every top-level JSON object/array embedded in ``text``."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            values.append(value)
            i = end
        else:
            i += 1
    return values


def _json_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for value in _json_values(text):
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            value = value["tool_calls"]
        for candidate in value if isinstance(value, list) else [value]:
            call = _as_call(candidate)
            if call is not None:
                calls.append(call)
    return calls


def _python_calls(text: str) -> list[ToolCall]:
    """BFCL's prompting format: ``[f(a=1), g(b="x")]`` (literal keyword arguments only)."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return []
    body = tree.body
    nodes = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls: list[ToolCall] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            return []
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except ValueError:
            return []
        calls.append(ToolCall(name=ast.unparse(node.func), arguments=args))
    return calls


def parse_text_calls(content: str) -> tuple[list[ToolCall], str]:
    """Recover calls printed as text. Returns (calls, source) with source ``"text_json"``,
    ``"text_python"`` or ``""`` when nothing call-shaped was found."""
    text = content.strip()
    if not text:
        return [], ""
    blobs = _TOOL_CALL_TAG.findall(text) or _FENCE.findall(text) or [text]
    calls = [c for blob in blobs for c in _json_calls(blob)]
    if calls:
        return calls, "text_json"
    calls = [c for blob in blobs for c in _python_calls(blob)]
    if calls:
        return calls, "text_python"
    return [], ""


def extract_calls(result: ChatResult) -> tuple[list[ToolCall], str]:
    """Native tool calls if any, else calls parsed from the text. Source ``"none"`` if neither."""
    if result.tool_calls:
        return list(result.tool_calls), "native"
    calls, source = parse_text_calls(result.content)
    return (calls, source) if calls else ([], "none")


def to_bfcl_calls(calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    """Our calls -> BFCL's decoded shape ``[{name: arguments}]``."""
    return [{c.name: c.arguments} for c in calls]


def score_item(item: BfclItem, calls: Sequence[ToolCall]) -> CheckResult:
    decoded = restore_names(to_bfcl_calls(calls), item.functions)
    return check(item.category, item.functions, decoded, item.ground_truth)


# --------------------------------------------------------------------------- runner


def row_key(model: str, category: str, item_id: str, repeat: int = 0) -> str:
    return f"{model}|bfcl|{category}|{item_id}|{repeat}"


def _meta(
    cfg: BfclConfig,
    files: Mapping[str, Mapping[str, Any]],
    item_ids: Mapping[str, Sequence[str]],
    server_version: str | None,
) -> dict[str, Any]:
    settings = asdict(cfg.settings)
    settings.pop("ollama_api_key", None)
    settings["cache_dir"] = _tilde(settings["cache_dir"]) if settings["cache_dir"] else None
    return {
        "suite": "bfcl",
        "title": f"Berkeley Function Calling Leaderboard ({DATASET_VERSION}, non-live, Python)",
        "models": list(cfg.models),
        "modes": ["bfcl"],
        "categories": list(cfg.categories),
        "dataset": {
            "source_url": SOURCE_URL,
            "leaderboard_url": LEADERBOARD_URL,
            "code_url": CODE_URL,
            "license": LICENSE,
            "citation": CITATION,
            "version": DATASET_VERSION,
            "data_dir": _tilde(cfg.data_dir / "bfcl"),
            "files": {k: dict(v) for k, v in files.items()},
        },
        "sample": DEFAULT_SAMPLE if cfg.sample is None else cfg.sample,
        "seed": cfg.seed,
        "item_ids": {k: list(v) for k, v in item_ids.items()},
        "temperature": 0.0,
        "ollamasearch_version": __version__,
        "ollama_server_version": server_version,
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": settings,
    }


def _check_hashes(meta_path: Path, files: Mapping[str, Mapping[str, Any]]) -> None:
    """Warn when a resumed run sees different dataset files than it started with."""
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))["dataset"]["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    for rel, info in files.items():
        if rel in old and old[rel].get("sha256") != info["sha256"]:
            log.warning("%s changed since this run started (sha256 differs); results mix", rel)


def _run_one(llm: LLM, item: BfclItem, tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    llm.reset_usage()
    start = time.perf_counter()
    error = None
    result = ChatResult(content="")
    try:
        result = llm.chat(item.messages, tools=tools)
    except Exception as exc:  # a failed call is a scored failure, like SearchBench errors
        log.warning("%s %s: %s", llm.model, item.id, exc)
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - start
    usage = llm.reset_usage()
    calls, source = extract_calls(result)
    verdict = CheckResult(False, "model_error", error) if error else score_item(item, calls)
    out: dict[str, Any] = {
        "score": {
            "correct": verdict.valid,
            "error_type": verdict.error_type,
            "detail": verdict.detail,
        },
        "calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
        "call_source": source,
        "content": result.content,
        "thinking": result.thinking,
        "error": error,
        "latency_s": round(latency, 3),
        "usage": usage.to_dict(),
    }
    out["empty_reply"] = is_empty_reply(out)
    if out["empty_reply"]:
        note = (
            f"Empty reply after {usage.eval_tokens} generated tokens "
            "(Ollama discards tool calls to unknown names)."
        )
        out["score"]["detail"] = " ".join(filter(None, [verdict.detail, note]))
    return out


def is_empty_reply(row: Mapping[str, Any]) -> bool:
    """Tokens were generated but neither text nor a call came back.

    Ollama drops a parsed tool call whose name is not a registered tool (typically the dotted
    name from the prompt instead of BFCL's underscored one), leaving an empty reply.
    """
    return (
        not row.get("error")
        and not row.get("calls")
        and not str(row.get("content") or "").strip()
        and row.get("usage", {}).get("eval_tokens", 0) > 0
    )


def run_bfcl(
    cfg: BfclConfig,
    client: ChatClient,
    *,
    progress: ProgressFn | None = None,
    server_version: str | None = None,
) -> dict[str, Any]:
    """Run (or resume) BFCL categories for each model; returns the summary dict."""
    for category in cfg.categories:
        if category not in BFCL_CATEGORIES:
            raise ValueError(f"unknown BFCL category {category!r}; choose from {BFCL_CATEGORIES}")
    s = cfg.settings
    files = ensure_files(cfg.data_dir, cfg.categories, use_truststore=s.use_truststore)
    selected = {
        c: sample_items(load_category(cfg.data_dir, c), cfg.sample, cfg.seed)
        for c in cfg.categories
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = cfg.out_dir / "runs.jsonl"
    meta_path = cfg.out_dir / "meta.json"
    if meta_path.exists():
        _check_hashes(meta_path, files)
    else:
        item_ids = {c: [it.id for it in items] for c, items in selected.items()}
        meta_path.write_text(
            json.dumps(_meta(cfg, files, item_ids, server_version), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    done = {r["key"] for r in read_rows(runs_path)}
    total = len(cfg.models) * sum(len(v) for v in selected.values())
    count = 0

    with runs_path.open("a", encoding="utf-8") as out:
        for model in cfg.models:
            llm = LLM(
                client,
                model,
                think=s.think,
                temperature=0.0,
                seed=s.seed,
                num_ctx=s.num_ctx,
                max_tokens=s.max_tokens,
            )
            ran_any = False
            for category, items in selected.items():
                for item in items:
                    count += 1
                    key = row_key(model, category, item.id)
                    if key in done:
                        continue
                    ran_any = True
                    row = {
                        "key": key,
                        "suite": "bfcl",
                        "model": model,
                        "mode": "bfcl",
                        "category": category,
                        "id": item.id,
                        "question_id": item.id,
                        "repeat": 0,
                        "messages": item.messages,
                        **_run_one(llm, item, item_tools(item)),
                        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    }
                    out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    out.flush()
                    if progress is not None:
                        progress(count, total, row)
            if ran_any:
                llm.unload()  # free VRAM before the next model (6 GB cards)

    return finalize_bfcl(cfg.out_dir)


# --------------------------------------------------------------------------- summary


def _stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(r["score"]["correct"]) for r in rows)
    latencies = [float(r.get("latency_s", 0.0)) for r in rows]
    usages = [r.get("usage", {}) for r in rows]
    eval_tokens = sum(u.get("eval_tokens", 0) for u in usages)
    eval_s = sum(u.get("eval_s", 0.0) for u in usages)
    errors = Counter(r["score"]["error_type"] for r in rows if not r["score"]["correct"])
    sources = Counter(r.get("call_source", "none") for r in rows)
    text_calls = sum(sources.get(k, 0) for k in TEXT_SOURCES)
    # BFCL's function-calling mode only sees native calls: a call printed as text fails there.
    native_correct = sum(
        bool(r["score"]["correct"]) and r.get("call_source") not in TEXT_SOURCES for r in rows
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "native_accuracy": native_correct / n if n else None,
        "error_types": dict(errors.most_common()),
        "call_sources": dict(sources.most_common()),
        "text_call_rate": text_calls / n if n else None,
        "no_call_rate": sources.get("none", 0) / n if n else None,
        "empty_reply_rate": sum(is_empty_reply(r) for r in rows) / n if n else None,
        "error_rate": sum(r.get("error") is not None for r in rows) / n if n else None,
        "latency_mean_s": sum(latencies) / n if n else None,
        "latency_p50_s": percentile(latencies, 50) if n else None,
        "latency_p95_s": percentile(latencies, 95) if n else None,
        "tokens_per_s": eval_tokens / eval_s if eval_s > 0 else None,
        "mean_prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages) / n if n else None,
        "mean_eval_tokens": eval_tokens / n if n else None,
    }


def summarize_bfcl(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Groups per model x category, plus one ``"overall"`` group per model.

    ``overall.accuracy`` is the unweighted mean of the category accuracies (BFCL averages
    categories, not items); ``ast_accuracy`` is the same mean over the AST categories only
    (BFCL's "non-live AST" summary, irrelevance being reported separately).
    """
    by: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    models: list[str] = []
    for row in rows:
        if row.get("suite", "bfcl") != "bfcl":
            continue
        by[(row["model"], row["category"])].append(row)
        if row["model"] not in models:
            models.append(row["model"])
    groups: list[dict[str, Any]] = []
    for model in models:
        cats = [c for c in BFCL_CATEGORIES if (model, c) in by]
        cats += sorted(c for (m, c) in by if m == model and c not in BFCL_CATEGORIES)
        per_cat = []
        for category in cats:
            g = {
                "model": model,
                "mode": "bfcl",
                "category": category,
                **_stats(by[(model, category)]),
            }
            per_cat.append(g)
        accs = {g["category"]: g["accuracy"] for g in per_cat if g["accuracy"] is not None}
        native = [g["native_accuracy"] for g in per_cat if g["native_accuracy"] is not None]
        ast_accs = [a for c, a in accs.items() if c in AST_CATEGORIES]
        stats = _stats([r for c in cats for r in by[(model, c)]])
        overall = {
            "model": model,
            "mode": "bfcl",
            "category": "overall",
            **stats,
            "accuracy": sum(accs.values()) / len(accs) if accs else None,
            "native_accuracy": sum(native) / len(native) if native else None,
            "item_accuracy": stats["accuracy"],
            "ast_accuracy": sum(ast_accs) / len(ast_accs) if ast_accs else None,
            "irrelevance_accuracy": accs.get("irrelevance"),
            "by_category": accs,
        }
        groups += [overall, *per_cat]
    return {"groups": groups}


def finalize_bfcl(out_dir: Path) -> dict[str, Any]:
    """(Re)build summary.json and report.md from runs.jsonl."""
    rows = read_rows(out_dir / "runs.jsonl")
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    summary = {"meta": meta, **summarize_bfcl(rows)}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_bfcl_report(summary), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- report

DIFFERENCES = (
    "Differences from the official leaderboard: only five non-live Python single-turn "
    "categories (no Java/JavaScript, live, multi-turn or agentic ones); a seeded sample per "
    "category unless `--sample 0`; models are queried through Ollama's native tool calling "
    "(chat template of the model), not BFCL's per-model handlers or its prompting format; "
    "calls printed as JSON or Python text are accepted and counted separately; schema keys the "
    "Ollama Python client cannot carry (`default`, `optional`, `maximum`, nested object "
    "`properties`, ...) are moved into the parameter description; the overall score is "
    "the unweighted mean of the categories run here. Function names keep BFCL's `.` -> `_` "
    "rule; when a model calls the dotted name it read in the prompt, Ollama discards the call "
    "(unknown tool) and the reply is empty (*Empty replies*; BFCL would score the same item as "
    "a wrong function name). Numbers are comparable in spirit, not "
    "identical to leaderboard entries."
)


def render_bfcl_report(summary: Mapping[str, Any]) -> str:
    meta = summary.get("meta", {})
    groups: list[Mapping[str, Any]] = summary.get("groups", [])
    ds = meta.get("dataset", {})
    parts = [f"# BFCL report: {meta.get('title', 'Berkeley Function Calling Leaderboard')}", ""]
    parts += [
        f"Data and checker: Berkeley Function Calling Leaderboard, {CITATION}. "
        f"License {LICENSE}. Data: {SOURCE_URL} · Leaderboard: {LEADERBOARD_URL} · "
        f"Code: {CODE_URL}",
        "",
    ]
    if meta:
        files = ds.get("files", {})
        hashes = ", ".join(
            f"`{rel}` ({str(v.get('sha256', ''))[:12]})" for rel, v in sorted(files.items())
        )
        parts += [
            f"- Models: {', '.join(meta.get('models', []))}",
            f"- Categories: {', '.join(meta.get('categories', []))}",
            f"- Sample: {meta.get('sample') or 'all'} per category, seed {meta.get('seed')}",
            f"- Files: {hashes}" if hashes else "- Files: –",
            f"- Temperature 0, think={meta.get('settings', {}).get('think')}, "
            f"num_ctx={meta.get('settings', {}).get('num_ctx')}, "
            f"max_tokens={meta.get('settings', {}).get('max_tokens')}",
            f"- ollamasearch {meta.get('ollamasearch_version')}, "
            f"Ollama {meta.get('ollama_server_version') or '?'}, started {meta.get('started_at')}",
            "",
        ]
    if not groups:
        return "\n".join([*parts, "No runs yet.", ""])

    overall = [g for g in groups if g["category"] == "overall"]
    per_cat = [g for g in groups if g["category"] != "overall"]
    cats = [c for c in BFCL_CATEGORIES if any(g["category"] == c for g in per_cat)]

    parts += ["## Overall", ""]
    parts.append(
        _table(
            [
                "Model",
                "Overall",
                "Native only",
                "AST (non-live)",
                *cats,
                "Text calls",
                "No call",
                "tok/s",
            ],
            [
                [
                    g["model"],
                    _pct(g["accuracy"]),
                    _pct(g["native_accuracy"]),
                    _pct(g["ast_accuracy"]),
                    *(_pct(g["by_category"].get(c)) for c in cats),
                    _pct(g["text_call_rate"]),
                    _pct(g["no_call_rate"]),
                    _num(g["tokens_per_s"], 0),
                ]
                for g in overall
            ],
        )
    )
    parts += [
        "",
        "Overall = unweighted mean over the categories. Native only = the same, counting calls "
        "printed as text as failures (what BFCL's function-calling mode would see). AST = mean "
        "over simple, multiple, parallel and parallel_multiple (irrelevance: correct = no call).",
        "",
        "## By category",
        "",
    ]
    parts.append(
        _table(
            [
                "Model",
                "Category",
                "N",
                "Accuracy",
                "Native",
                "Calls native / text / none",
                "Empty replies",
                "Model errors",
                "p50 s",
                "p95 s",
                "tok/s",
                "Gen tok",
            ],
            [
                [
                    g["model"],
                    g["category"],
                    str(g["n"]),
                    _pct(g["accuracy"]),
                    _pct(g["native_accuracy"]),
                    "{} / {} / {}".format(
                        g["call_sources"].get("native", 0),
                        g["call_sources"].get("text_json", 0)
                        + g["call_sources"].get("text_python", 0),
                        g["call_sources"].get("none", 0),
                    ),
                    _pct(g.get("empty_reply_rate")),
                    _pct(g["error_rate"]),
                    _num(g["latency_p50_s"]),
                    _num(g["latency_p95_s"]),
                    _num(g["tokens_per_s"], 0),
                    _num(g["mean_eval_tokens"], 0),
                ]
                for g in per_cat
            ],
        )
    )
    parts += ["", "## Failure types", ""]
    failure_rows = [
        [g["model"], g["category"], f"`{etype}`", str(count)]
        for g in per_cat
        for etype, count in g["error_types"].items()
    ]
    if failure_rows:
        parts.append(
            _table(["Model", "Category", "Error type (BFCL checker)", "Count"], failure_rows)
        )
    else:
        parts.append("No failures.")
    parts += ["", "## Notes", "", DIFFERENCES, ""]
    return "\n".join(parts)

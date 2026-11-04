"""Answering pipelines: closed-book baseline, single-shot RAG, and a tool-calling agent."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings
from .llm import LLM, ChatResult, Message, ToolCall
from .prompts import (
    AGENT_FINAL_NUDGE,
    AGENT_SYSTEM,
    CLOSED_BOOK_SYSTEM,
    SYSTEM_PROMPT,
    ContextSource,
    rag_user_prompt,
    search_results_text,
)
from .rank import Document, select_passages
from .search import PageFetcher, SearchBackend, SearchError

log = logging.getLogger(__name__)

MODES = ("closed", "rag", "agent", "multi")
TokenCallback = Callable[[str], None]

_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
_URL_RE = re.compile(r"https?://[^\s\)\]\>\"'<,]+")


@dataclass
class Source:
    number: int
    url: str
    title: str


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: Any
    status: str  # "ok" | "error" (tool ran but failed) | "invalid" (bad name/arguments)
    detail: str = ""
    elapsed: float = 0.0
    result_chars: int = 0
    from_text: bool = False  # parsed from message text instead of native tool_calls


@dataclass
class Answer:
    question: str
    text: str
    mode: str
    model: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    context_urls: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    step_limit_reached: bool = False
    thinking: str = ""
    error: str | None = None

    @property
    def cited_urls(self) -> list[str]:
        by_number = {s.number: s.url for s in self.sources}
        return [by_number[n] for n in self.cited if n in by_number]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cited_urls"] = self.cited_urls
        return d


def extract_citations(text: str, sources: Sequence[Source]) -> list[int]:
    """Source numbers cited as [n] / [n, m], plus sources whose URL appears verbatim."""
    valid = {s.number for s in sources}
    found: list[int] = []
    for group in _CITE_RE.findall(text):
        for part in re.split(r"[,;]", group):
            n = int(part.strip())
            if n in valid and n not in found:
                found.append(n)
    by_url = {s.url.rstrip("/"): s.number for s in sources}
    for url in _URL_RE.findall(text):
        n_url = by_url.get(url.rstrip("/.").rstrip("/"))
        if n_url is not None and n_url not in found:
            found.append(n_url)
    return found


class Pipeline:
    mode = "base"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        on_token: TokenCallback | None = None,
    ) -> Answer:
        raise NotImplementedError

    def _generate(self, messages: list[Message], on_token: TokenCallback | None) -> ChatResult:
        if on_token is not None:
            return self.llm.stream(messages, on_token)
        return self.llm.chat(messages)

    def _answer(self, question: str, **kwargs: Any) -> Answer:
        return Answer(question=question, mode=self.mode, model=self.llm.model, **kwargs)


class ClosedBookPipeline(Pipeline):
    """No retrieval: measures what the model knows (or hallucinates) on its own."""

    mode = "closed"

    def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        on_token: TokenCallback | None = None,
    ) -> Answer:
        self.llm.reset_usage()
        start = time.perf_counter()
        messages: list[Message] = [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            *history,
            {"role": "user", "content": question},
        ]
        error = None
        try:
            result = self._generate(messages, on_token)
        except Exception as exc:
            log.warning("model call failed: %s", exc)
            result, error = ChatResult(content=""), f"llm: {exc}"
        total = time.perf_counter() - start
        return self._answer(
            question,
            text=result.content.strip(),
            thinking=result.thinking,
            timings={"llm_s": total, "total_s": total},
            tokens=self.llm.reset_usage().to_dict(),
            steps=1,
            error=error,
        )


class RagPipeline(Pipeline):
    """search -> concurrent fetch -> BM25 passage selection -> one answer call."""

    mode = "rag"

    def __init__(
        self,
        llm: LLM,
        backend: SearchBackend,
        fetcher: PageFetcher,
        *,
        top_k: int = 5,
        context_chars: int = 6000,
        source_chars: int = 1500,
        passage_chars: int = 500,
    ) -> None:
        super().__init__(llm)
        self.backend = backend
        self.fetcher = fetcher
        self.top_k = top_k
        self.context_chars = context_chars
        self.source_chars = source_chars
        self.passage_chars = passage_chars

    def build_context(
        self, question: str, timings: dict[str, float]
    ) -> tuple[list[ContextSource], str | None]:
        t = time.perf_counter()
        error = None
        try:
            results = self.backend.search(question, self.top_k)
        except SearchError as exc:
            log.warning("search failed: %s", exc)
            results, error = [], f"search: {exc}"
        if not results and error is None:
            # An empty result list is almost always the search engine failing (rate-limited
            # upstream engines), not a real "nothing exists"; surface it so benchmarks can
            # tell infrastructure failures from model abstentions.
            error = "search: no results"
        timings["search_s"] = time.perf_counter() - t

        t = time.perf_counter()
        pages = self.fetcher.fetch_many([r.url for r in results])
        timings["fetch_s"] = time.perf_counter() - t

        t = time.perf_counter()
        docs = []
        for i, (res, page) in enumerate(zip(results, pages, strict=True)):
            text = page.text if page.ok and page.text else res.snippet
            if text:
                docs.append(Document(i, page.title or res.title, text))
        passages = select_passages(
            question,
            docs,
            budget_chars=self.context_chars,
            per_source_chars=self.source_chars,
            passage_chars=self.passage_chars,
        )
        grouped: dict[int, list[str]] = {}
        for p in passages:
            grouped.setdefault(p.source_id, []).append(p.text)
        sources = []
        for number, source_id in enumerate(sorted(grouped), start=1):
            res, page = results[source_id], pages[source_id]
            sources.append(
                ContextSource(number, page.title or res.title, res.url, grouped[source_id])
            )
        timings["rank_s"] = time.perf_counter() - t
        return sources, error

    def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        on_token: TokenCallback | None = None,
    ) -> Answer:
        self.llm.reset_usage()
        start = time.perf_counter()
        timings: dict[str, float] = {}
        context, error = self.build_context(question, timings)

        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": rag_user_prompt(question, context)},
        ]
        t = time.perf_counter()
        try:
            result = self._generate(messages, on_token)
        except Exception as exc:
            log.warning("model call failed: %s", exc)
            result, error = ChatResult(content=""), f"llm: {exc}"
        timings["llm_s"] = time.perf_counter() - t
        timings["total_s"] = time.perf_counter() - start

        sources = [Source(c.number, c.url, c.title) for c in context]
        text = result.content.strip()
        return self._answer(
            question,
            text=text,
            thinking=result.thinking,
            sources=sources,
            cited=extract_citations(text, sources),
            context_urls=[s.url for s in sources],
            timings=timings,
            tokens=self.llm.reset_usage().to_dict(),
            steps=1,
            error=error,
        )


def _tool(name: str, description: str, param: str, param_description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {param: {"type": "string", "description": param_description}},
                "required": [param],
            },
        },
    }


AGENT_TOOLS = [
    _tool(
        "web_search",
        "Search the web. Returns numbered results with title, URL and snippet.",
        "query",
        "Search query, e.g. the key entity names from the question.",
    ),
    _tool(
        "web_fetch",
        "Open a web page and return its passages most relevant to the question.",
        "url",
        "Full URL of the page, taken from a web_search result.",
    ),
]
_ARG_ALIASES = {
    "query": ("query", "q", "search", "search_query", "text", "keywords"),
    "url": ("url", "link", "href", "page", "uri"),
}
_TOOL_PARAM = {"web_search": "query", "web_fetch": "url"}
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_text_tool_calls(content: str) -> list[ToolCall]:
    """Recover tool calls that small models print as JSON text instead of native calls."""
    text = content.strip()
    blobs = _TOOL_CALL_TAG.findall(text)
    if not blobs:
        stripped = _FENCE.sub("", text).strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            blobs = [stripped]
    calls = []
    for blob in blobs:
        try:
            data = json.loads(blob)
        except ValueError:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            continue
        if data["name"] not in _TOOL_PARAM:
            continue
        args = data.get("arguments", data.get("parameters", {}))
        calls.append(ToolCall(name=data["name"], arguments=args))
    return calls


class SourceRegistry:
    """Stable source numbering across all tool calls of one agent run."""

    def __init__(self) -> None:
        self.sources: list[Source] = []
        self._by_url: dict[str, Source] = {}
        self.context_urls: list[str] = []

    def add(self, url: str, title: str) -> int:
        key = url.rstrip("/")
        if key not in self._by_url:
            src = Source(len(self.sources) + 1, url, title)
            self.sources.append(src)
            self._by_url[key] = src
        elif title and not self._by_url[key].title:
            self._by_url[key].title = title
        if url not in self.context_urls:
            self.context_urls.append(url)
        return self._by_url[key].number

    def number(self, url: str) -> int | None:
        src = self._by_url.get(url.rstrip("/"))
        return src.number if src else None


class AgentPipeline(Pipeline):
    """Native tool-calling loop with web_search and web_fetch."""

    mode = "agent"

    def __init__(
        self,
        llm: LLM,
        backend: SearchBackend,
        fetcher: PageFetcher,
        *,
        max_steps: int = 4,
        results_per_search: int = 5,
        fetch_chars: int = 2500,
        passage_chars: int = 500,
    ) -> None:
        super().__init__(llm)
        self.backend = backend
        self.fetcher = fetcher
        self.max_steps = max(1, max_steps)
        self.results_per_search = results_per_search
        self.fetch_chars = fetch_chars
        self.passage_chars = passage_chars

    @staticmethod
    def _argument(call: ToolCall, param: str) -> str | None:
        args = call.arguments
        if isinstance(args, str):
            return args.strip() or None  # bare string argument: be lenient
        if not isinstance(args, dict):
            return None
        for key in _ARG_ALIASES[param]:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _web_search(self, query: str, registry: SourceRegistry) -> tuple[str, str, str]:
        try:
            results = self.backend.search(query, self.results_per_search)
        except SearchError as exc:
            return f"Search failed: {exc}", "error", str(exc)
        rows = [(registry.add(r.url, r.title), r.title, r.url, r.snippet) for r in results]
        if not rows:
            return search_results_text(rows), "error", "no results"
        return search_results_text(rows), "ok", f"{len(rows)} results"

    def _web_fetch(self, url: str, question: str, registry: SourceRegistry) -> tuple[str, str, str]:
        page = self.fetcher.fetch_many([url])[0]
        if not page.ok:
            return f"Could not fetch {url}: {page.error}", "error", page.error or ""
        number = registry.add(url, page.title)
        passages = select_passages(
            question,
            [Document(number, page.title, page.text)],
            budget_chars=self.fetch_chars,
            per_source_chars=self.fetch_chars,
            passage_chars=self.passage_chars,
        )
        body = "\n...\n".join(p.text for p in passages) or page.text[: self.fetch_chars]
        return f"[{number}] {page.title}\nURL: {url}\n{body}", "ok", f"{len(page.text)} chars"

    def _execute(
        self,
        call: ToolCall,
        step: int,
        question: str,
        registry: SourceRegistry,
        from_text: bool,
        seen: set[tuple[str, str]],
    ) -> tuple[ToolCallRecord, str]:
        start = time.perf_counter()
        param = _TOOL_PARAM.get(call.name)
        if param is None:
            output = (
                f"Error: unknown tool '{call.name}'. Available tools: web_search(query), "
                "web_fetch(url)."
            )
            status, detail = "invalid", "unknown tool"
        elif (value := self._argument(call, param)) is None:
            output = f"Error: {call.name} requires a non-empty string argument '{param}'."
            status, detail = "invalid", f"bad arguments: {call.arguments!r}"[:200]
        elif (call.name, value.casefold()) in seen:
            output = (
                f"You already called {call.name} with this {param}; its result is above. "
                "Use a different query or URL, or answer now."
            )
            status, detail = "ok", "duplicate call, not re-run"
        elif call.name == "web_search":
            seen.add((call.name, value.casefold()))
            output, status, detail = self._web_search(value, registry)
        else:
            seen.add((call.name, value.casefold()))
            output, status, detail = self._web_fetch(value, question, registry)
        record = ToolCallRecord(
            step=step,
            name=call.name,
            arguments=call.arguments,
            status=status,
            detail=detail,
            elapsed=time.perf_counter() - start,
            result_chars=len(output),
            from_text=from_text,
        )
        return record, output

    def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        on_token: TokenCallback | None = None,
    ) -> Answer:
        self.llm.reset_usage()
        start = time.perf_counter()
        registry = SourceRegistry()
        records: list[ToolCallRecord] = []
        messages: list[Message] = [
            {"role": "system", "content": AGENT_SYSTEM},
            *history,
            {"role": "user", "content": question},
        ]
        llm_s = 0.0
        final: ChatResult | None = None
        thinking: list[str] = []
        error = None
        steps = 0
        step_limit = False
        seen: set[tuple[str, str]] = set()
        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                t = time.perf_counter()
                result = self.llm.chat(messages, tools=AGENT_TOOLS)
                llm_s += time.perf_counter() - t
                if result.thinking:
                    thinking.append(result.thinking)
                calls, from_text = result.tool_calls, False
                if not calls:
                    calls, from_text = parse_text_tool_calls(result.content), True
                if not calls:
                    final = result
                    break
                assistant = ChatResult(
                    content="" if from_text else result.content, tool_calls=calls
                )
                messages.append(assistant.as_message())
                for call in calls:
                    record, output = self._execute(call, step, question, registry, from_text, seen)
                    records.append(record)
                    messages.append({"role": "tool", "tool_name": call.name, "content": output})
            if final is None:  # step limit: force an answer without tools
                step_limit = True
                t = time.perf_counter()
                messages.append({"role": "user", "content": AGENT_FINAL_NUDGE})
                final = self._generate(messages, on_token)
                llm_s += time.perf_counter() - t
                if final.thinking:
                    thinking.append(final.thinking)
            elif on_token is not None and final.content:
                on_token(final.content)
        except Exception as exc:
            log.warning("agent model call failed: %s", exc)
            error = f"llm: {exc}"

        text = final.content.strip() if final is not None else ""
        sources = registry.sources
        total = time.perf_counter() - start
        tool_s = sum(r.elapsed for r in records)
        return self._answer(
            question,
            text=text,
            thinking="\n".join(thinking),
            sources=list(sources),
            cited=extract_citations(text, sources),
            context_urls=list(registry.context_urls),
            tool_calls=records,
            timings={"llm_s": llm_s, "tools_s": tool_s, "total_s": total},
            tokens=self.llm.reset_usage().to_dict(),
            steps=steps,
            step_limit_reached=step_limit,
            error=error,
        )


def build_pipeline(
    mode: str,
    llm: LLM,
    backend: SearchBackend | None,
    fetcher: PageFetcher | None,
    settings: Settings,
) -> Pipeline:
    if mode == "closed":
        return ClosedBookPipeline(llm)
    if backend is None or fetcher is None:
        raise ValueError(f"mode {mode!r} needs a search backend and a fetcher")
    if mode == "rag":
        return RagPipeline(
            llm,
            backend,
            fetcher,
            top_k=settings.top_k,
            context_chars=settings.context_char_budget,
            source_chars=settings.source_char_budget,
            passage_chars=settings.passage_chars,
        )
    if mode == "agent":
        return AgentPipeline(
            llm,
            backend,
            fetcher,
            max_steps=settings.agent_max_steps,
            results_per_search=settings.top_k,
            fetch_chars=settings.source_char_budget + 1000,
            passage_chars=settings.passage_chars,
        )
    if mode == "multi":
        from .multiagent import ResearchPipeline  # local import: multiagent imports this module

        return ResearchPipeline(llm, backend, fetcher, settings, verify=settings.multi_verify)
    raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")


_Registry = SourceRegistry  # backwards-compatible private alias

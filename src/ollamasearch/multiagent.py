"""Multi-agent research pipeline: planner -> searchers/readers -> synthesizer -> verifier.

Built for small local models: every role is one LLM call with a narrow job and (except the
synthesizer) a JSON schema for its output, so a 2B model never has to drive a tool loop.

    plan -> round 1 (search + fetch concurrently, rank, read) -> optional re-plan -> round 2
         -> synthesize -> verify -> (one follow-up round -> re-synthesize -> verify)

Source numbers are global across the run, so citations in findings, drafts and the final
answer all map to the same URLs. See docs/multi-agent.md.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .fetch import FetchResult
from .llm import LLM, ChatResult, Message, Usage
from .pipelines import (
    Answer,
    Pipeline,
    Source,
    SourceRegistry,
    TokenCallback,
    ToolCallRecord,
    extract_citations,
)
from .prompts import NOT_FOUND, ContextSource, format_sources
from .prompts_multi import (
    PLAN_SCHEMA,
    PLANNER_SYSTEM,
    READ_SCHEMA,
    READER_SYSTEM,
    REPLAN_SCHEMA,
    REPLANNER_SYSTEM,
    SYNTH_SYSTEM,
    VERIFIER_SYSTEM,
    VERIFY_SCHEMA,
    FindingView,
    planner_prompt,
    question_language,
    reader_prompt,
    replanner_prompt,
    synth_prompt,
    verifier_prompt,
)
from .rank import Document, select_passages, tokenize
from .search import PageFetcher, SearchBackend, SearchError, SearchResult

log = logging.getLogger(__name__)

ROLES = ("plan", "read", "synthesize", "verify")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_NULLS = {"", "null", "none", "n/a", "unknown"}


@dataclass(frozen=True)
class SubQuestion:
    question: str
    query: str
    needs_previous: bool = False


@dataclass
class Finding:
    sub_question: str
    query: str
    found: bool
    finding: str
    source_numbers: list[int]
    date_or_version: str | None
    round: int

    def view(self) -> FindingView:
        return FindingView(
            self.sub_question, self.finding, self.source_numbers, self.date_or_version
        )


@dataclass
class Verdict:
    supported: bool
    problems: list[str] = field(default_factory=list)
    follow_up_query: str | None = None


@dataclass
class _Searched:
    results: list[SearchResult] = field(default_factory=list)
    records: list[ToolCallRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """Mutable state of one ``run()``."""

    question: str
    registry: SourceRegistry = field(default_factory=SourceRegistry)
    passages: dict[int, list[str]] = field(default_factory=dict)  # source number -> passages
    pages: dict[str, FetchResult] = field(default_factory=dict)  # fetched this run, by URL
    searched: list[str] = field(default_factory=list)  # queries, in order
    records: list[ToolCallRecord] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    llm_calls: int = 0
    search_results: int = 0

    def add_time(self, key: str, seconds: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + seconds

    def error(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a model's JSON reply, tolerating code fences and chatter around the object."""
    s = _FENCE.sub("", text.strip()).strip()
    for candidate in (s, m.group(0) if (m := _OBJECT.search(s)) else None):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _optional_text(value: Any) -> str | None:
    s = _text(value)
    return None if s.casefold() in _NULLS else s


def _abstains(text: str) -> bool:
    return NOT_FOUND.casefold().rstrip(".") in text.casefold()


def usable_findings(findings: Sequence[Finding]) -> list[Finding]:
    """What the synthesizer works from: the found findings, or else the partial ones.

    Small readers sometimes write the answer and still say ``found: false``; partial findings
    give the synthesizer a chance, and the verifier decides whether such a draft survives.
    """
    found = [f for f in findings if f.found]
    return found or [f for f in findings if f.finding and f.source_numbers]


def parse_plan(data: dict[str, Any] | None, question: str, limit: int = 3) -> list[SubQuestion]:
    """Planner JSON -> sub-questions. Anything unusable falls back to the question itself."""
    fallback = [SubQuestion(question, question)]
    if data is None:
        return fallback
    raw = data.get("sub_questions")
    subs: list[SubQuestion] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        q = _text(item.get("question"))
        query = _text(item.get("query")) or q
        if not q and not query:
            continue
        key = (q or query).casefold()
        if key in seen:
            continue
        seen.add(key)
        subs.append(SubQuestion(q or query, query, bool(item.get("needs_previous"))))
    if not subs:
        return fallback
    if data.get("simple") is True or len(subs) == 1:
        # A simple question is researched as asked; keep the planner's search query.
        return [SubQuestion(question, subs[0].query)]
    subs = subs[: max(1, limit)]
    if subs[0].needs_previous:  # the first hop can't depend on anything
        subs[0] = SubQuestion(subs[0].question, subs[0].query, False)
    return subs


class ResearchPipeline(Pipeline):
    """Same interface as the other pipelines: ``run(question, history=, on_token=) -> Answer``.

    Budgets come from ``settings``: ``top_k`` results per search, ``context_char_budget`` /
    ``source_char_budget`` / ``passage_chars`` for each reader's passages (and the verifier's
    evidence), and ``agent_max_steps`` as the total number of search rounds.
    """

    mode = "multi"

    def __init__(
        self,
        llm: LLM,
        backend: SearchBackend,
        fetcher: PageFetcher,
        settings: Settings,
        *,
        role_llms: Mapping[str, LLM] | None = None,
        max_sub_questions: int = 3,
        verify: bool = True,
        follow_up_rounds: int = 1,
    ) -> None:
        super().__init__(llm)
        self.backend = backend
        self.fetcher = fetcher
        self.settings = settings
        unknown = set(role_llms or {}) - set(ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}; choose from {', '.join(ROLES)}")
        self.role_llms = dict(role_llms or {})
        self.max_sub_questions = max(1, max_sub_questions)
        self.verify = verify
        self.follow_up_rounds = max(0, follow_up_rounds)
        self.max_rounds = max(1, settings.agent_max_steps)

    # -- plumbing ---------------------------------------------------------------------------

    def _llm(self, role: str) -> LLM:
        return self.role_llms.get(role, self.llm)

    def _llms(self) -> list[LLM]:
        out: list[LLM] = [self.llm]
        for llm in self.role_llms.values():
            if all(llm is not o for o in out):
                out.append(llm)
        return out

    def _usage(self) -> dict[str, float]:
        total = Usage()
        for llm in self._llms():
            u = llm.reset_usage()
            total.calls += u.calls
            total.prompt_tokens += u.prompt_tokens
            total.eval_tokens += u.eval_tokens
            total.prompt_eval_s += u.prompt_eval_s
            total.eval_s += u.eval_s
            total.load_s += u.load_s
            total.total_s += u.total_s
            total.truncated += u.truncated
        return total.to_dict()

    def _chat(
        self,
        run: _Run,
        role: str,
        messages: list[Message],
        *,
        schema: Mapping[str, Any] | None = None,
        on_token: TokenCallback | None = None,
    ) -> ChatResult:
        """One role call; exceptions propagate after the call is counted and timed."""
        llm = self._llm(role)
        run.llm_calls += 1
        t = time.perf_counter()
        try:
            if on_token is not None:
                result = llm.stream(messages, on_token)
            else:
                result = llm.chat(messages, format=schema)
        finally:
            elapsed = time.perf_counter() - t
            run.add_time(f"{role}_s", elapsed)
            run.add_time("llm_s", elapsed)
        if result.thinking:
            run.thinking.append(result.thinking)
        return result

    def _json_call(
        self,
        run: _Run,
        role: str,
        step: int,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, ToolCallRecord]:
        start = time.perf_counter()
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        data: dict[str, Any] | None = None
        try:
            result = self._chat(run, role, messages, schema=schema)
        except Exception as exc:
            log.warning("%s call failed: %s", role, exc)
            run.error(f"llm: {exc}")
            status, detail, chars = "error", f"llm: {exc}"[:200], 0
        else:
            data = parse_json_object(result.content)
            chars = len(result.content)
            status, detail = ("ok", "") if data is not None else ("invalid", "reply was not JSON")
            if data is None:
                log.info("%s returned invalid JSON: %.200r", role, result.content)
        record = ToolCallRecord(
            step=step,
            name=role,
            arguments=arguments,
            status=status,
            detail=detail,
            elapsed=time.perf_counter() - start,
            result_chars=chars,
        )
        run.records.append(record)
        return data, record

    # -- roles ------------------------------------------------------------------------------

    def _plan(self, run: _Run, history: Sequence[Message]) -> list[SubQuestion]:
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {str(m.get('content', ''))[:300]}" for m in history[-6:]
        )
        data, record = self._json_call(
            run,
            "plan",
            0,
            PLANNER_SYSTEM,
            planner_prompt(run.question, history_text),
            PLAN_SCHEMA,
            {"question": run.question},
        )
        subs = parse_plan(data, run.question, self.max_sub_questions)
        if record.status == "ok":
            kind = "simple" if len(subs) == 1 else f"{len(subs)} sub-questions"
            record.detail = f"{kind}: " + " | ".join(
                s.query + (" (needs previous)" if s.needs_previous else "") for s in subs
            )
        else:
            record.detail += "; fallback: one sub-question = the question"
        return subs

    def _replan(
        self,
        run: _Run,
        step: int,
        findings: Sequence[Finding],
        pending: Sequence[SubQuestion],
    ) -> list[SubQuestion]:
        lines = [f.view().line() for f in findings if f.finding]
        open_questions = [s.question for s in pending]
        open_questions += [f.sub_question for f in findings if not f.found]
        data, record = self._json_call(
            run,
            "plan",
            step,
            REPLANNER_SYSTEM,
            replanner_prompt(run.question, lines, open_questions, run.searched),
            REPLAN_SCHEMA,
            {"question": run.question, "round": step},
        )
        if data is None:  # keep the original dependent sub-questions as they were planned
            record.detail += "; fallback: pending sub-questions as planned"
            return [SubQuestion(s.question, s.query) for s in pending]
        done = {q.casefold() for q in run.searched}
        subs: list[SubQuestion] = []
        raw = data.get("sub_questions")
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            q = _text(item.get("question"))
            query = _text(item.get("query")) or q
            if query and query.casefold() not in done:
                done.add(query.casefold())
                subs.append(SubQuestion(q or query, query))
        subs = subs[:2]
        record.detail = " | ".join(s.query for s in subs) if subs else "nothing more to search"
        return subs

    def _search(self, queries: Sequence[str], fallback: str, step: int) -> _Searched:
        """Search every query and merge the results (deduplicated by URL); if that finds
        nothing, search ``fallback``. Runs in a worker thread."""
        out = _Searched()
        pending = list(dict.fromkeys(q for q in queries if q))
        tried: set[str] = set()
        while pending:
            q = pending.pop(0)
            tried.add(q.casefold())
            start = time.perf_counter()
            results: list[SearchResult] = []
            try:
                results = self.backend.search(q, self.settings.top_k)
                status = "ok" if results else "error"
                detail = f"{len(results)} results" if results else "no results"
            except SearchError as exc:
                log.warning("search failed: %s", exc)
                status, detail = "error", str(exc)[:200]
                out.errors.append(f"search: {exc}")
            out.records.append(
                ToolCallRecord(
                    step=step,
                    name="search",
                    arguments={"query": q},
                    status=status,
                    detail=detail,
                    elapsed=time.perf_counter() - start,
                    result_chars=sum(len(r.snippet) for r in results),
                )
            )
            known = {r.url for r in out.results}
            out.results += [r for r in results if r.url not in known]
            if not pending and not out.results and fallback and fallback.casefold() not in tried:
                pending.append(fallback)
        return out

    def _retrieve(
        self, run: _Run, subs: Sequence[SubQuestion], step: int, anchor: str | None
    ) -> list[list[ContextSource]]:
        """Search all sub-questions concurrently, fetch the union of new URLs once, rank.

        ``anchor`` (the user's own question, in round 1) is searched alongside the first
        sub-question, so a bad planner query never leaves the readers with less than RAG gets.
        """
        jobs = [[s.query] for s in subs]
        if anchor:
            jobs[0].append(anchor)
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, len(subs))) as pool:
            searched = list(
                pool.map(
                    lambda job: self._search(job[0], job[1].question, step),
                    zip(jobs, subs, strict=True),
                )
            )
        run.add_time("search_s", time.perf_counter() - t)
        all_results: list[list[SearchResult]] = []
        for out in searched:  # merged in sub-question order: deterministic numbering and trace
            run.records.extend(out.records)
            for e in out.errors:
                run.error(e)
            run.searched.extend(r.arguments["query"] for r in out.records)
            run.search_results += len(out.results)
            all_results.append(out.results)

        t = time.perf_counter()
        new_urls = list(
            dict.fromkeys(r.url for rs in all_results for r in rs if r.url not in run.pages)
        )
        for page in self.fetcher.fetch_many(new_urls):
            run.pages[page.url] = page
            run.records.append(
                ToolCallRecord(
                    step=step,
                    name="fetch",
                    arguments={"url": page.url},
                    status="ok" if page.ok else "error",
                    detail=f"{len(page.text)} chars" if page.ok else (page.error or "")[:200],
                    elapsed=page.elapsed,
                    result_chars=len(page.text),
                )
            )
        run.add_time("fetch_s", time.perf_counter() - t)

        t = time.perf_counter()
        contexts = [
            self._rank(run, sub, results) for sub, results in zip(subs, all_results, strict=True)
        ]
        run.add_time("rank_s", time.perf_counter() - t)
        return contexts

    def _rank(
        self, run: _Run, sub: SubQuestion, results: Sequence[SearchResult]
    ) -> list[ContextSource]:
        docs: list[Document] = []
        for i, res in enumerate(results):
            page = run.pages.get(res.url)
            ok = page is not None and page.ok and bool(page.text)
            text = page.text if ok and page is not None else res.snippet
            if text:
                title = (page.title if page is not None else "") or res.title
                docs.append(Document(i, title, text))
        s = self.settings
        query = sub.question if sub.query == sub.question else f"{sub.question} {sub.query}"
        passages = select_passages(
            query,
            docs,
            budget_chars=s.context_char_budget,
            per_source_chars=s.source_char_budget,
            passage_chars=s.passage_chars,
        )
        grouped: dict[int, list[str]] = {}
        for p in passages:
            grouped.setdefault(p.source_id, []).append(p.text)
        sources: list[ContextSource] = []
        for i in sorted(grouped):
            res = results[i]
            page = run.pages.get(res.url)
            title = (page.title if page is not None else "") or res.title
            number = run.registry.add(res.url, title)
            known = run.passages.setdefault(number, [])
            known.extend(p for p in grouped[i] if p not in known)
            sources.append(ContextSource(number, title, res.url, grouped[i]))
        return sources

    def _read(
        self, run: _Run, sub: SubQuestion, sources: list[ContextSource], step: int
    ) -> Finding:
        empty = Finding(sub.question, sub.query, False, "", [], None, step)
        if not sources:
            run.records.append(
                ToolCallRecord(step, "read", {"sub_question": sub.question}, "error", "no passages")
            )
            return empty
        text = format_sources(sources)
        data, record = self._json_call(
            run,
            "read",
            step,
            READER_SYSTEM,
            reader_prompt(sub.question, text),
            READ_SCHEMA,
            {"sub_question": sub.question},
        )
        if data is None:
            return empty
        shown = {s.number for s in sources}
        numbers: list[int] = []
        raw = data.get("source_numbers")
        for n in raw if isinstance(raw, list) else []:
            if isinstance(n, int) and not isinstance(n, bool) and n in shown and n not in numbers:
                numbers.append(n)
        finding = _text(data.get("finding"))
        found = data.get("found") is True and bool(finding)
        if finding and not numbers:
            numbers = self._best_sources(finding, sources)
        result = Finding(
            sub.question,
            sub.query,
            found,
            finding,
            numbers,
            _optional_text(data.get("date_or_version")),
            step,
        )
        cites = "".join(f"[{n}]" for n in numbers)
        record.detail = f"{'found' if found else 'not found'}: {finding[:160]} {cites}".strip()
        return result

    @staticmethod
    def _best_sources(finding: str, sources: Sequence[ContextSource]) -> list[int]:
        """The reader forgot its citations: pick the source sharing most terms with the finding."""
        terms = set(tokenize(finding))
        best, best_overlap = None, 0
        for s in sources:
            overlap = len(terms & set(tokenize(" ".join(s.passages))))
            if overlap > best_overlap:
                best, best_overlap = s.number, overlap
        return [best] if best is not None else []

    def _round(
        self, run: _Run, subs: Sequence[SubQuestion], step: int, anchor: str | None = None
    ) -> list[Finding]:
        contexts = self._retrieve(run, subs, step, anchor)
        return [self._read(run, sub, ctx, step) for sub, ctx in zip(subs, contexts, strict=True)]

    def _synthesize(
        self,
        run: _Run,
        findings: Sequence[Finding],
        history: Sequence[Message],
        step: int,
        on_token: TokenCallback | None,
    ) -> str:
        found = usable_findings(findings)
        start = time.perf_counter()
        if not found:
            run.records.append(
                ToolCallRecord(
                    step, "synthesize", {"findings": 0}, "ok", "nothing found: not-found answer"
                )
            )
            if on_token is not None:
                on_token(NOT_FOUND)
            return NOT_FOUND
        partial = not any(f.found for f in found)
        messages: list[Message] = [
            {"role": "system", "content": SYNTH_SYSTEM},
            *history,
            {
                "role": "user",
                "content": synth_prompt(
                    run.question, [f.view() for f in found], question_language(run.question)
                ),
            },
        ]
        try:
            text = self._chat(run, "synthesize", messages, on_token=on_token).content.strip()
            status, detail = "ok", ""
        except Exception as exc:
            log.warning("synthesize call failed: %s", exc)
            run.error(f"llm: {exc}")
            # Fall back to the findings themselves; they are already cited sentences.
            text = NOT_FOUND if partial else " ".join(f.view().line() for f in found)
            status, detail = "error", f"llm: {exc}"[:200]
            if on_token is not None:
                on_token(text)
        if not text:
            text, status, detail = NOT_FOUND, "invalid", "empty reply"
        elif not _abstains(text) and not extract_citations(text, run.registry.sources):
            # Every finding is sourced, so an uncited answer can still point at its evidence.
            numbers = list(dict.fromkeys(n for f in found for n in f.source_numbers))[:3]
            if numbers:
                suffix = " " + "".join(f"[{n}]" for n in numbers)
                text += suffix
                detail = "citations added from findings"
                if on_token is not None:
                    on_token(suffix)
        run.records.append(
            ToolCallRecord(
                step,
                "synthesize",
                {"findings": len(found)},
                status,
                detail or text[:160],
                time.perf_counter() - start,
                len(text),
            )
        )
        return text

    def _evidence(self, run: _Run, draft: str, findings: Sequence[Finding]) -> str:
        # Cited sources first, then the findings' own sources in case the draft mis-cited.
        cited = extract_citations(draft, run.registry.sources)
        used = [n for f in usable_findings(findings) for n in f.source_numbers]
        numbers = list(dict.fromkeys([*cited, *used]))
        by_number = {s.number: s for s in run.registry.sources}
        budget = self.settings.context_char_budget
        sources: list[ContextSource] = []
        for n in numbers:
            src = by_number.get(n)
            if src is None:
                continue
            kept: list[str] = []
            for p in run.passages.get(n, []):
                if len(p) > budget:
                    break
                kept.append(p)
                budget -= len(p)
            if kept:
                sources.append(ContextSource(n, src.title, src.url, kept))
        return format_sources(sources) if sources else "(no passages)"

    def _verify(self, run: _Run, draft: str, findings: Sequence[Finding], step: int) -> Verdict:
        data, record = self._json_call(
            run,
            "verify",
            step,
            VERIFIER_SYSTEM,
            verifier_prompt(run.question, draft, self._evidence(run, draft, findings)),
            VERIFY_SCHEMA,
            {"draft": draft[:300]},
        )
        if data is None:  # no usable verdict: don't punish the draft for the checker failing
            record.detail += "; treated as supported"
            return Verdict(True)
        raw = data.get("problems")
        problems = [p for p in (_text(x) for x in (raw if isinstance(raw, list) else [])) if p]
        verdict = Verdict(
            supported=data.get("supported") is not False,
            problems=problems,
            follow_up_query=_optional_text(data.get("follow_up_query")),
        )
        if verdict.supported:
            record.detail = "supported"
        else:
            record.detail = "unsupported: " + ("; ".join(problems) or "no reason given")[:200]
            if verdict.follow_up_query:
                record.detail += f" -> follow-up: {verdict.follow_up_query}"
        return verdict

    # -- orchestration ----------------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        on_token: TokenCallback | None = None,
    ) -> Answer:
        self._usage()  # reset counters on every role model
        start = time.perf_counter()
        run = _Run(question)

        plan = self._plan(run, history)
        first = [s for s in plan if not s.needs_previous]
        pending = [s for s in plan if s.needs_previous]
        findings = self._round(run, first, 1, anchor=question)
        rounds = 1

        # Multi-hop / gap filling: re-plan with the findings, then one more round.
        if rounds < self.max_rounds and (pending or not all(f.found for f in findings)):
            subs = self._replan(run, rounds + 1, findings, pending)
            if subs:
                rounds += 1
                findings += self._round(run, subs, rounds)

        # Without a verifier the first draft is final and can be streamed as it is generated.
        stream = on_token if not self.verify else None
        draft = self._synthesize(run, findings, history, rounds, stream)
        drafts: list[tuple[str, Verdict]] = []
        if self.verify and not _abstains(draft):
            verdict = self._verify(run, draft, findings, rounds)
            drafts.append((draft, verdict))
            follow_ups = 0
            confirmed_missing = False
            while (
                not verdict.supported
                and verdict.follow_up_query
                and follow_ups < self.follow_up_rounds
                and rounds < self.max_rounds
                and verdict.follow_up_query.casefold() not in {q.casefold() for q in run.searched}
            ):
                follow_ups += 1
                rounds += 1
                sub = SubQuestion(question, verdict.follow_up_query)
                new = [f for f in self._round(run, [sub], rounds) if f.found]
                if not new:
                    # The verifier rejected the draft and a targeted search found nothing to
                    # back it: the original finding was a misreading. Re-synthesizing from the
                    # same findings would only repeat the draft.
                    confirmed_missing = True
                    break
                findings += new
                draft = self._synthesize(run, findings, history, rounds, None)
                verdict = self._verify(run, draft, findings, rounds)
                drafts.append((draft, verdict))
            draft = self._choose(drafts, findings, confirmed_missing)
            if draft == NOT_FOUND:
                last_verify = [r for r in run.records if r.name == "verify"][-1]
                last_verify.detail += " => final answer: not found"
        if self.verify and on_token is not None:
            on_token(draft)

        if run.search_results == 0 and not any(e.startswith("search") for e in run.errors):
            run.error("search: no results")
        run.timings["total_s"] = time.perf_counter() - start
        sources: list[Source] = list(run.registry.sources)
        return self._answer(
            question,
            text=draft,
            thinking="\n".join(run.thinking),
            sources=sources,
            cited=extract_citations(draft, sources),
            context_urls=list(run.registry.context_urls),
            tool_calls=run.records,
            timings=run.timings,
            tokens=self._usage(),
            steps=run.llm_calls,
            error="; ".join(run.errors[:3]) or None,
        )

    @staticmethod
    def _choose(
        drafts: Sequence[tuple[str, Verdict]],
        findings: Sequence[Finding],
        confirmed_missing: bool = False,
    ) -> str:
        """Pick the final answer once verification is over.

        The latest verified draft wins. With none verified, the answer is NOT_FOUND when no
        finding supports one: nothing was found, or only partial findings were used, or the
        follow-up round confirmed the verifier's rejection. Otherwise a single rejection is not
        trusted over the reader, and the draft with the fewest problems is kept.
        """
        supported = [d for d, v in drafts if v.supported]
        if supported:
            return supported[-1]
        if not drafts or confirmed_missing or not any(f.found for f in findings):
            return NOT_FOUND
        fewest = min(len(v.problems) for _, v in drafts)
        return [d for d, v in drafts if len(v.problems) == fewest][-1]

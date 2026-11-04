"""Multi-agent research pipeline (mode "multi"): planner, searchers, synthesizer, verifier."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from ollamasearch.config import Settings
from ollamasearch.fetch import Fetcher, FetchResult
from ollamasearch.llm import LLM
from ollamasearch.multiagent import (
    ResearchPipeline,
    SubQuestion,
    parse_json_object,
    parse_plan,
)
from ollamasearch.prompts import NOT_FOUND
from ollamasearch.prompts_multi import (
    PLAN_SCHEMA,
    PLANNER_SYSTEM,
    READ_SCHEMA,
    READER_SYSTEM,
    REPLANNER_SYSTEM,
    SYNTH_SYSTEM,
    VERIFIER_SYSTEM,
    VERIFY_SCHEMA,
    question_language,
)
from ollamasearch.search import CorpusBackend, SearchError, SearchResult

from .conftest import FakeOllama

Q_POP = "What is the population of Zarnitsa-on-Ural?"
Q_HOP = "In what year was the company that built the Orlek Bridge founded?"

ROLE_BY_SYSTEM = {
    PLANNER_SYSTEM: "plan",
    REPLANNER_SYSTEM: "replan",
    READER_SYSTEM: "read",
    SYNTH_SYSTEM: "synthesize",
    VERIFIER_SYSTEM: "verify",
}
NOT_READ = {"finding": "", "source_numbers": [], "date_or_version": None, "found": False}
DEFAULTS: dict[str, Any] = {
    "plan": "not json",  # -> fallback plan: the question itself
    "replan": {"sub_questions": []},
    "read": NOT_READ,
    "synthesize": "Default answer.",
    "verify": {"problems": [], "supported": True, "follow_up_query": None},
}


class Roles:
    """FakeOllama responder that dispatches on the role's system prompt.

    Each role has its own queue of replies (dict -> JSON content, str, Exception, or a callable
    receiving the user prompt); an empty queue falls back to ``DEFAULTS``.
    """

    def __init__(self, **queues: list[Any]) -> None:
        self.queues = {role: list(items) for role, items in queues.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        role = ROLE_BY_SYSTEM[kwargs["messages"][0]["content"]]
        self.calls.append((role, kwargs))
        queue = self.queues.get(role, [])
        item = queue.pop(0) if queue else DEFAULTS[role]
        if callable(item) and not isinstance(item, Exception):
            item = item(kwargs["messages"][-1]["content"])
        if isinstance(item, Exception):
            return item
        return json.dumps(item) if isinstance(item, dict) else item

    def roles(self) -> list[str]:
        return [role for role, _ in self.calls]

    def prompts(self, role: str) -> list[str]:
        return [str(kw["messages"][-1]["content"]) for r, kw in self.calls if r == role]

    def kwargs(self, role: str) -> list[dict[str, Any]]:
        return [kw for r, kw in self.calls if r == role]


def number_of(prompt: str, name: str) -> int:
    """Source number that ``http://corpus.local/<name>`` has in a reader/verifier prompt."""
    m = re.search(rf"\[(\d+)\] [^\n]*\nURL: http://corpus\.local/{re.escape(name)}\n", prompt)
    assert m, f"{name} not in prompt:\n{prompt}"
    return int(m.group(1))


def found(finding: str, name: str, date: str | None = None) -> Callable[[str], dict[str, Any]]:
    """Reader reply that cites ``name`` by whatever number it has in the prompt."""

    def reply(prompt: str) -> dict[str, Any]:
        n = number_of(prompt, name)
        return {"finding": finding, "source_numbers": [n], "date_or_version": date, "found": True}

    return reply


def plan(*subs: tuple[str, str, bool], simple: bool = False) -> dict[str, Any]:
    return {
        "sub_questions": [{"question": q, "query": s, "needs_previous": d} for q, s, d in subs],
        "simple": simple,
    }


def cite(text: str, finding: str = "") -> Callable[[str], str]:
    """Synthesizer reply citing (as ``{n}``) the source number shown after ``finding``."""

    def reply(prompt: str) -> str:
        n = re.search(re.escape(finding) + r"[^\n]*?\[(\d+)\]", prompt)
        assert n, prompt
        return text.format(n=n.group(1))

    return reply


def unsupported(problem: str, follow_up: str | None) -> dict[str, Any]:
    return {"problems": [problem], "supported": False, "follow_up_query": follow_up}


@pytest.fixture
def make(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher, settings: Settings
) -> Callable[..., tuple[ResearchPipeline, Roles]]:
    def build(
        roles: Roles | None = None,
        *,
        backend: Any = None,
        fetcher: Any = None,
        settings_: Settings | None = None,
        **kwargs: Any,
    ) -> tuple[ResearchPipeline, Roles]:
        roles = roles or Roles()
        fake.default = roles  # type: ignore[assignment]
        pipeline = ResearchPipeline(
            llm,
            backend or corpus,
            fetcher or corpus_fetcher,
            settings_ or settings,
            **kwargs,
        )
        return pipeline, roles

    return build


Make = Callable[..., tuple[ResearchPipeline, Roles]]


# -- parsing -------------------------------------------------------------------------------------


def test_parse_json_object_tolerates_fences_and_chatter() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("no json here") is None
    assert parse_json_object("") is None


def test_parse_plan_decomposed() -> None:
    data = plan(
        ("Which company built the Orlek Bridge?", "Orlek Bridge builder", False),
        ("When was that company founded?", "company founded", True),
    )
    subs = parse_plan(data, Q_HOP)
    assert subs == [
        SubQuestion("Which company built the Orlek Bridge?", "Orlek Bridge builder", False),
        SubQuestion("When was that company founded?", "company founded", True),
    ]


def test_parse_plan_simple_researches_the_question_itself() -> None:
    data = plan(("Zarnitsa population?", "Zarnitsa population", False), simple=True)
    assert parse_plan(data, Q_POP) == [SubQuestion(Q_POP, "Zarnitsa population")]
    # one sub-question without the flag is treated the same way
    data = plan(("Who built it?", "Orlek Bridge", False))
    assert parse_plan(data, Q_HOP) == [SubQuestion(Q_HOP, "Orlek Bridge")]
    # "simple" wins even if the model listed more sub-questions
    data = plan(("a?", "a", False), ("b?", "b", False), simple=True)
    assert parse_plan(data, Q_POP) == [SubQuestion(Q_POP, "a")]


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"sub_questions": []},
        {"sub_questions": "Zarnitsa"},
        {"sub_questions": [{"question": "", "query": ""}, 3, None]},
    ],
)
def test_parse_plan_invalid_or_empty_falls_back_to_the_question(data: Any) -> None:
    assert parse_plan(data, Q_POP) == [SubQuestion(Q_POP, Q_POP)]


def test_parse_plan_limits_dedupes_and_anchors_first_hop() -> None:
    data = plan(
        ("a?", "a", True),
        ("A?", "a again", False),  # duplicate question (case-insensitive)
        ("b?", "b", False),
        ("c?", "c", False),
        ("d?", "d", False),
    )
    subs = parse_plan(data, Q_POP, limit=3)
    assert [s.question for s in subs] == ["a?", "b?", "c?"]
    assert subs[0].needs_previous is False  # the first hop cannot depend on anything
    # a missing query falls back to the sub-question text
    assert parse_plan({"sub_questions": [{"question": "x?"}, {"query": "y"}]}, Q_POP) == [
        SubQuestion("x?", "x?"),
        SubQuestion("y", "y"),
    ]


def test_question_language() -> None:
    assert question_language("Какова глубина озера?") == "Russian"
    assert question_language("Көкжайсаң ауылы қай жылы құрылған?") == "Kazakh"
    assert question_language(Q_POP) == "the language of the question"
    assert question_language("???") == "the language of the question"


# -- the happy path --------------------------------------------------------------------------


def test_simple_question_end_to_end(make: Make, fake: FakeOllama) -> None:
    roles = Roles(
        plan=[plan((Q_POP, "Zarnitsa-on-Ural population", False), simple=True)],
        read=[found("Zarnitsa-on-Ural has 7,420 inhabitants (2023 census).", "zarnitsa.html")],
        synthesize=[cite("The population is 7,420 [{n}].")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)

    assert answer.mode == "multi" and answer.model == "qwen3.5:2b"
    assert answer.text.startswith("The population is 7,420 [")
    assert roles.roles() == ["plan", "read", "synthesize", "verify"]
    assert answer.steps == 4 <= 6  # a simple question stays cheap
    assert answer.cited_urls == ["http://corpus.local/zarnitsa.html"]
    assert "http://corpus.local/zarnitsa.html" in answer.context_urls
    assert [s.url for s in answer.sources] == answer.context_urls
    assert answer.error is None

    names = [c.name for c in answer.tool_calls]
    assert names[0] == "plan" and names[-3:] == ["read", "synthesize", "verify"]
    # round 1 searches the planner's query and the user's question, then fetches the union
    searches = [c.arguments["query"] for c in answer.tool_calls if c.name == "search"]
    assert searches == ["Zarnitsa-on-Ural population", Q_POP]
    fetches = [c.arguments["url"] for c in answer.tool_calls if c.name == "fetch"]
    assert len(fetches) == len(set(fetches)) > 0
    assert all(c.status == "ok" for c in answer.tool_calls)
    assert {c.step for c in answer.tool_calls if c.name in ("search", "fetch", "read")} == {1}

    # structured output: every JSON role passes its schema, the synthesizer writes text
    assert roles.kwargs("plan")[0]["format"] == PLAN_SCHEMA
    assert roles.kwargs("read")[0]["format"] == READ_SCHEMA
    assert roles.kwargs("verify")[0]["format"] == VERIFY_SCHEMA
    assert "format" not in roles.kwargs("synthesize")[0]

    for key in ("plan_s", "search_s", "fetch_s", "rank_s", "read_s", "synthesize_s"):
        assert key in answer.timings
    assert answer.timings["verify_s"] >= 0 and answer.timings["total_s"] >= answer.timings["llm_s"]
    assert answer.tokens["calls"] == 4 and answer.tokens["eval_tokens"] == 40
    assert len(fake.calls) == 4


def test_reader_sees_ranked_passages_and_verifier_sees_cited_evidence(make: Make) -> None:
    roles = Roles(
        read=[found("7,420 people.", "zarnitsa.html")],
        synthesize=[cite("7,420 [{n}].")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)
    reader = roles.prompts("read")[0]
    assert "Sub-question: " + Q_POP in reader
    assert "According to the 2023 census the town has a population of 7,420" in reader
    assert "cookies" not in reader.casefold()  # boilerplate never reaches the reader
    verifier = roles.prompts("verify")[0]
    n = answer.cited[0]
    assert f"[{n}] Zarnitsa-on-Ural" in verifier and "7,420" in verifier
    assert "Draft answer: " + answer.text in verifier


# -- searchers -------------------------------------------------------------------------------


class SlowCorpus:
    """Corpus backend that sleeps per search and records the peak number of searches in flight."""

    name = "slow-corpus"

    def __init__(self, inner: CorpusBackend, delay: float = 0.05) -> None:
        self.inner = inner
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.queries.append(query)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return self.inner.search(query, k)


def test_concurrent_searchers_share_global_source_numbers(
    make: Make, corpus: CorpusBackend
) -> None:
    backend = SlowCorpus(corpus)
    roles = Roles(
        plan=[
            plan(
                ("What is the population of Zarnitsa-on-Ural?", "Zarnitsa-on-Ural", False),
                ("How deep is lake Tengizkol?", "Тенгизколь глубина", False),
            )
        ],
        read=[
            found("7,420 people.", "zarnitsa.html"),
            found("Tengizkol is 48 m deep.", "tengizkol.html"),
        ],
        synthesize=[lambda p: "7,420 people; 48 m deep " + "".join(re.findall(r"\[\d+\]", p))],
    )
    pipeline, roles = make(roles, backend=backend)
    answer = pipeline.run("Population of Zarnitsa-on-Ural and depth of Tengizkol?")

    assert backend.peak == 2  # the two sub-questions searched at the same time
    first, second = roles.prompts("read")
    numbers = [s.number for s in answer.sources]
    assert numbers == list(range(1, len(numbers) + 1))
    by_url = {s.url: s.number for s in answer.sources}
    # a URL keeps its number in every reader prompt; numbers never clash across readers
    for prompt in (first, second):
        for n, url in re.findall(r"\[(\d+)\] [^\n]*\nURL: (\S+)", prompt):
            assert by_url[url] == int(n)
    assert number_of(second, "tengizkol.html") == by_url["http://corpus.local/tengizkol.html"]
    assert set(answer.cited_urls) == {
        "http://corpus.local/zarnitsa.html",
        "http://corpus.local/tengizkol.html",
    }
    # fetches are not repeated for URLs both sub-questions found
    fetched = [c.arguments["url"] for c in answer.tool_calls if c.name == "fetch"]
    assert len(fetched) == len(set(fetched))


def test_search_falls_back_to_sub_question_when_query_finds_nothing(
    make: Make, corpus: CorpusBackend
) -> None:
    backend = SlowCorpus(corpus, delay=0)
    roles = Roles(
        plan=[
            plan(
                ("Какова глубина озера Тенгизколь?", "Тенгизколь", False),
                ("What is the population of Zarnitsa-on-Ural?", "qqqq zzzz", False),
            )
        ]
    )
    pipeline, _ = make(roles, backend=backend)
    answer = pipeline.run("Zarnitsa and Tengizkol?")
    searches = [(c.arguments["query"], c.status) for c in answer.tool_calls if c.name == "search"]
    assert searches == [
        ("Тенгизколь", "ok"),
        ("Zarnitsa and Tengizkol?", "ok"),  # the user's question rides along in round 1
        ("qqqq zzzz", "error"),
        ("What is the population of Zarnitsa-on-Ural?", "ok"),  # fallback
    ]


# -- reader ----------------------------------------------------------------------------------


def test_reader_not_found_gives_not_found_without_synthesis(make: Make) -> None:
    pipeline, roles = make(Roles())
    answer = pipeline.run("What is the phone number of the Zarnitsa-on-Ural mayor's office?")
    assert answer.text == NOT_FOUND
    # nothing found -> one re-plan (which gives up), no synthesizer or verifier call
    assert roles.roles() == ["plan", "read", "replan"]
    assert answer.steps == 3
    synth = [c for c in answer.tool_calls if c.name == "synthesize"]
    assert synth[0].detail.startswith("nothing found") and answer.cited == []
    assert answer.context_urls  # passages did reach the reader


def test_reader_invalid_json_and_bad_source_numbers(make: Make) -> None:
    roles = Roles(
        read=[
            "I think it's 7420",  # not JSON
        ],
        replan=[{"sub_questions": [{"question": Q_POP, "query": "Zarnitsa census"}]}],
    )
    roles.queues["read"].append(
        {"finding": "7,420 people live there.", "source_numbers": [99, True], "found": True}
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)
    read_calls = [c for c in answer.tool_calls if c.name == "read"]
    assert read_calls[0].status == "invalid"
    # invented / boolean source numbers are dropped and replaced by the best-matching source
    assert read_calls[1].status == "ok" and "found: 7,420" in read_calls[1].detail
    zarnitsa = next(s.number for s in answer.sources if s.url.endswith("zarnitsa.html"))
    assert read_calls[1].detail.endswith(f"[{zarnitsa}]")


def test_partial_finding_is_synthesized_but_needs_the_verifier(make: Make) -> None:
    partial = {"finding": "Zarnitsa has 7,420 people.", "source_numbers": [1], "found": False}
    # verifier accepts -> the partial finding becomes the answer
    pipeline, roles = make(Roles(read=[partial], synthesize=["7,420 [1]."]))
    assert pipeline.run(Q_POP).text == "7,420 [1]."
    # verifier rejects and gives no follow-up -> only partial findings: not found
    roles = Roles(
        read=[partial],
        synthesize=["7,420 [1]."],
        verify=[unsupported("not stated", None)],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)
    assert answer.text == NOT_FOUND
    assert roles.roles() == ["plan", "read", "replan", "synthesize", "verify"]
    assert answer.tool_calls[-1].detail.endswith("=> final answer: not found")


# -- synthesizer -----------------------------------------------------------------------------


def test_synthesizer_cites_finding_sources_and_answers_in_question_language(make: Make) -> None:
    q = "Какова максимальная глубина озера Тенгизколь?"
    roles = Roles(
        read=[found("Максимальная глубина озера Тенгизколь — 48 метров.", "tengizkol.html")],
        synthesize=[cite("48 метров [{n}].")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(q)
    prompt = roles.prompts("synthesize")[0]
    n = number_of(roles.prompts("read")[0], "tengizkol.html")
    assert f"- Максимальная глубина озера Тенгизколь — 48 метров. [{n}]" in prompt
    assert "Answer in Russian" in prompt and f"e.g. [{n}]" in prompt
    assert answer.cited_urls == ["http://corpus.local/tengizkol.html"]


def test_uncited_answer_gets_citations_from_its_findings(make: Make) -> None:
    roles = Roles(read=[found("7,420 people.", "zarnitsa.html")], synthesize=["About 7,420."])
    pipeline, _ = make(roles)
    answer = pipeline.run(Q_POP)
    assert re.fullmatch(r"About 7,420\. \[\d+\]", answer.text)
    assert answer.cited_urls == ["http://corpus.local/zarnitsa.html"]


def test_history_reaches_planner_and_synthesizer(make: Make) -> None:
    history = [
        {"role": "user", "content": "Tell me about Zarnitsa-on-Ural."},
        {"role": "assistant", "content": "It is a river town."},
    ]
    roles = Roles(read=[found("7,420 people.", "zarnitsa.html")])
    pipeline, roles = make(roles)
    pipeline.run("What is its population?", history=history)
    assert "user: Tell me about Zarnitsa-on-Ural." in roles.prompts("plan")[0]
    assert roles.kwargs("synthesize")[0]["messages"][1:3] == history


# -- verifier --------------------------------------------------------------------------------


def test_verifier_triggers_exactly_one_follow_up_round(make: Make) -> None:
    roles = Roles(
        read=[
            found("The Orlek Bridge was built by Tamarisk Engineering.", "orlek-bridge.html"),
            found("Tamarisk Engineering was founded in 1987.", "tamarisk.html"),
        ],
        synthesize=["It was built by Tamarisk Engineering [1].", "Founded in 1987 [2]."],
        verify=[
            unsupported("founding year missing", "Tamarisk Engineering founded"),
            unsupported("still unsure", "Tamarisk Engineering history"),
        ],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_HOP)
    first_pass = ["plan", "read", "synthesize", "verify"]
    assert roles.roles() == [*first_pass, "read", "synthesize", "verify"]  # one follow-up round
    searches = [c for c in answer.tool_calls if c.name == "search"]
    assert searches[-1].arguments == {"query": "Tamarisk Engineering founded"}
    assert searches[-1].step == 2
    assert "Tamarisk Engineering history" not in [c.arguments["query"] for c in searches]
    # neither draft verified, both have one problem: the latest (better informed) one is kept
    assert answer.text == "Founded in 1987 [2]."
    assert answer.steps == 7


def test_follow_up_respects_the_round_budget(make: Make, settings: Settings) -> None:
    roles = Roles(
        read=[found("Built by Tamarisk Engineering.", "orlek-bridge.html")],
        synthesize=["Tamarisk Engineering [1]."],
        verify=[unsupported("year missing", "Tamarisk Engineering founded")],
    )
    pipeline, roles = make(roles, settings_=settings.with_(agent_max_steps=1))
    answer = pipeline.run(Q_HOP)
    assert roles.roles() == ["plan", "read", "synthesize", "verify"]  # no round 2 at all
    assert answer.text == "Tamarisk Engineering [1]."  # a lone rejection doesn't erase it
    # follow-ups can also be switched off entirely
    roles = Roles(
        read=[found("Built by Tamarisk Engineering.", "orlek-bridge.html")],
        verify=[unsupported("year missing", "Tamarisk Engineering founded")],
    )
    pipeline, roles = make(roles, follow_up_rounds=0)
    pipeline.run(Q_HOP)
    assert roles.roles() == ["plan", "read", "synthesize", "verify"]


def test_follow_up_that_finds_nothing_confirms_not_found(make: Make) -> None:
    roles = Roles(
        read=[found("The mayor's office phone is 1987.", "zarnitsa.html"), NOT_READ],
        synthesize=["The phone number is 1987 [1]."],
        verify=[unsupported("no phone number in sources", "Zarnitsa mayor office phone")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run("What is the phone number of the Zarnitsa-on-Ural mayor's office?")
    assert answer.text == NOT_FOUND
    assert roles.roles() == ["plan", "read", "synthesize", "verify", "read"]
    verify = [c for c in answer.tool_calls if c.name == "verify"][-1]
    assert "follow-up: Zarnitsa mayor office phone" in verify.detail
    assert verify.detail.endswith("=> final answer: not found")


def test_repeated_follow_up_query_is_not_searched_again(make: Make) -> None:
    roles = Roles(
        plan=[plan((Q_POP, "Zarnitsa population", False), simple=True)],
        read=[found("7,420.", "zarnitsa.html")],
        verify=[unsupported("hmm", "zarnitsa POPULATION")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)
    assert roles.roles() == ["plan", "read", "synthesize", "verify"]
    assert answer.text.startswith("Default answer.")


# -- conflicting / multi-hop -----------------------------------------------------------------

CENSUS_OLD = """<html><head><title>Velmira census 2019</title></head><body><main>
<p>The 2019 census counted 18,460 residents in the town of Velmira.</p></main></body></html>"""
CENSUS_NEW = """<html><head><title>Velmira census 2024</title></head><body><main>
<p>The 2024 census counted 19,835 residents in the town of Velmira, up from 2019.</p>
</main></body></html>"""


def test_conflicting_findings_carry_dates_and_prefer_the_newest(make: Make) -> None:
    corpus = CorpusBackend.from_html({"old.html": CENSUS_OLD, "new.html": CENSUS_NEW})
    q = "What is the population of Velmira according to the latest census?"
    roles = Roles(
        plan=[
            plan(
                ("Velmira population 2019 census?", "Velmira census 2019", False),
                ("Velmira population 2024 census?", "Velmira census 2024", False),
            )
        ],
        read=[
            found("Velmira had 18,460 residents.", "old.html", "2019 census"),
            found("Velmira has 19,835 residents.", "new.html", "2024 census"),
        ],
        synthesize=[cite("19,835 residents (2024 census) [{n}].", "19,835 residents.")],
    )
    with Fetcher(transport=corpus.transport()) as fetcher:
        pipeline, roles = make(roles, backend=corpus, fetcher=fetcher)
        answer = pipeline.run(q)
    prompt = roles.prompts("synthesize")[0]
    assert "(as of: 2019 census)" in prompt and "(as of: 2024 census)" in prompt
    system = roles.kwargs("synthesize")[0]["messages"][0]["content"]
    assert "most recent" in system and "outdated" in system
    assert "most recent" in roles.kwargs("read")[0]["messages"][0]["content"]
    assert answer.cited_urls == ["http://corpus.local/new.html"]
    assert sorted(answer.context_urls) == [
        "http://corpus.local/new.html",
        "http://corpus.local/old.html",
    ]


def test_multi_hop_dependent_sub_question_runs_in_round_two(make: Make) -> None:
    roles = Roles(
        plan=[
            plan(
                ("Which company built the Orlek Bridge?", "Orlek Bridge built by", False),
                ("When was the company that built the Orlek Bridge founded?", "founded", True),
            )
        ],
        read=[
            found("The Orlek Bridge was built by Tamarisk Engineering.", "orlek-bridge.html"),
            found("Tamarisk Engineering was founded in 1987.", "tamarisk.html"),
        ],
        replan=[
            {
                "sub_questions": [
                    {
                        "question": "When was Tamarisk Engineering founded?",
                        "query": "Tamarisk Engineering founded",
                    }
                ]
            }
        ],
        synthesize=[lambda p: "Tamarisk Engineering was founded in 1987 " + p[-60:]],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_HOP)

    assert roles.roles() == ["plan", "read", "replan", "read", "synthesize", "verify"]
    assert answer.steps == 6
    # round 1 only researched the independent hop; the dependent one waited for its finding
    assert roles.prompts("read")[0].endswith(
        "Sub-question: Which company built the Orlek Bridge?\n\nYour JSON:"
    )
    replan = roles.prompts("replan")[0]
    assert "The Orlek Bridge was built by Tamarisk Engineering." in replan
    assert "- When was the company that built the Orlek Bridge founded?" in replan
    assert "- Orlek Bridge built by" in replan  # already-searched queries are listed
    round2 = [c for c in answer.tool_calls if c.step == 2]
    # synthesize/verify carry the number of the last round
    names = [c.name for c in round2 if c.name != "fetch"]
    assert names == ["plan", "search", "read", "synthesize", "verify"]
    assert round2[1].arguments == {"query": "Tamarisk Engineering founded"}
    assert "Sub-question: When was Tamarisk Engineering founded?" in roles.prompts("read")[1]
    synth = roles.prompts("synthesize")[0]
    assert "built by Tamarisk Engineering" in synth and "founded in 1987" in synth
    assert "http://corpus.local/tamarisk.html" in answer.context_urls


def test_replan_failure_runs_pending_sub_questions_as_planned(make: Make) -> None:
    roles = Roles(
        plan=[
            plan(
                ("Which company built the Orlek Bridge?", "Orlek Bridge", False),
                ("When was it founded?", "Tamarisk Engineering founded", True),
            )
        ],
        read=[found("Built by Tamarisk Engineering.", "orlek-bridge.html")],
        replan=["garbage"],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_HOP)
    replan = next(c for c in answer.tool_calls if c.name == "plan" and c.step == 2)
    assert replan.status == "invalid" and "fallback" in replan.detail
    searches = [c.arguments["query"] for c in answer.tool_calls if c.name == "search"]
    assert searches[-1] == "Tamarisk Engineering founded"


# -- failures --------------------------------------------------------------------------------


class BrokenSearch:
    name = "broken"

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        raise SearchError("engine down")


class BrokenFetcher:
    def fetch(self, url: str) -> FetchResult:
        return FetchResult(url=url, error="HTTP 500")

    def fetch_many(self, urls: Sequence[str]) -> list[FetchResult]:
        return [self.fetch(u) for u in urls]


def test_search_errors_are_recorded_and_still_answer(make: Make) -> None:
    pipeline, roles = make(Roles(), backend=BrokenSearch())
    answer = pipeline.run(Q_POP)
    assert answer.text == NOT_FOUND
    assert answer.error is not None and answer.error.startswith("search: engine down")
    searches = [c for c in answer.tool_calls if c.name == "search"]
    assert searches and all(c.status == "error" for c in searches)
    reads = [c for c in answer.tool_calls if c.name == "read"]
    assert reads[0].status == "error" and reads[0].detail == "no passages"
    assert "read" not in roles.roles()  # nothing to read: no reader call


def test_empty_search_is_reported_as_error(make: Make) -> None:
    corpus = CorpusBackend.from_html({"a.html": "<html><body><p>unrelated</p></body></html>"})
    pipeline, _ = make(Roles(), backend=corpus)
    answer = pipeline.run(Q_POP)
    assert answer.error == "search: no results" and answer.text == NOT_FOUND


def test_fetch_errors_fall_back_to_snippets(make: Make) -> None:
    roles = Roles(read=[found("7,420 people.", "zarnitsa.html")])
    pipeline, roles = make(roles, fetcher=BrokenFetcher())
    answer = pipeline.run(Q_POP)
    fetches = [c for c in answer.tool_calls if c.name == "fetch"]
    assert fetches and all(c.status == "error" and c.detail == "HTTP 500" for c in fetches)
    assert "URL: http://corpus.local/zarnitsa.html" in roles.prompts("read")[0]
    assert answer.cited_urls == ["http://corpus.local/zarnitsa.html"]  # still sourced
    assert answer.error is None  # failed fetches are normal on the web, not run errors


def test_llm_errors_in_every_role_are_recorded_and_still_answer(make: Make) -> None:
    roles = Roles(
        plan=[RuntimeError("planner down")],
        read=[found("Zarnitsa-on-Ural has 7,420 people.", "zarnitsa.html")],
        synthesize=[RuntimeError("synth down")],
        verify=[RuntimeError("verify down")],
    )
    pipeline, roles = make(roles)
    answer = pipeline.run(Q_POP)
    # the findings themselves become the answer, still cited
    assert answer.text.startswith("Zarnitsa-on-Ural has 7,420 people. [")
    assert answer.cited_urls == ["http://corpus.local/zarnitsa.html"]
    status = {c.name: c.status for c in answer.tool_calls if c.name != "fetch"}
    assert status["plan"] == status["synthesize"] == status["verify"] == "error"
    assert answer.error is not None and "planner down" in answer.error
    assert answer.steps == 4


def test_reader_llm_error_yields_not_found(make: Make) -> None:
    pipeline, _ = make(Roles(read=[RuntimeError("boom")]))
    answer = pipeline.run(Q_POP)
    assert answer.text == NOT_FOUND and answer.error == "llm: boom"
    assert next(c for c in answer.tool_calls if c.name == "read").status == "error"


def test_default_fake_replies_still_give_an_answer(
    fake: FakeOllama, llm: LLM, corpus: CorpusBackend, corpus_fetcher: Fetcher
) -> None:
    # "Default answer." for every call: invalid JSON everywhere, still a non-empty answer
    answer = ResearchPipeline(llm, corpus, corpus_fetcher, Settings()).run(Q_POP)
    assert answer.text == NOT_FOUND
    assert answer.steps == 3 and len(fake.calls) == 3  # plan, read, re-plan
    assert answer.context_urls and answer.error is None


# -- streaming / roles -----------------------------------------------------------------------


def test_streams_only_the_final_answer(make: Make, fake: FakeOllama) -> None:
    roles = Roles(
        read=[found("7,420 people.", "zarnitsa.html")],
        synthesize=["A draft [1].", "Final 7,420 [1]."],
        verify=[unsupported("unclear", "Zarnitsa census 2023")],
    )
    roles.queues["read"].append(found("Census: 7,420 people.", "zarnitsa.html"))
    pipeline, _ = make(roles)
    tokens: list[str] = []
    answer = pipeline.run(Q_POP, on_token=tokens.append)
    assert "".join(tokens) == answer.text == "Final 7,420 [1]."
    assert not any(kw.get("stream") for kw in fake.calls)  # drafts are never streamed


def test_streams_tokens_live_without_verifier(make: Make, fake: FakeOllama) -> None:
    roles = Roles(
        read=[found("7,420 people.", "zarnitsa.html")],
        synthesize=["The town has 7,420 people."],
    )
    pipeline, roles = make(roles, verify=False)
    tokens: list[str] = []
    answer = pipeline.run(Q_POP, on_token=tokens.append)
    assert len(tokens) > 2 and "".join(tokens) == answer.text
    assert re.fullmatch(r"The town has 7,420 people\. \[\d+\]", answer.text)
    assert fake.calls[-1].get("stream") is True
    assert "verify" not in roles.roles()


def test_streams_not_found(make: Make) -> None:
    pipeline, _ = make(Roles())
    tokens: list[str] = []
    answer = pipeline.run(Q_POP, on_token=tokens.append)
    assert tokens == [NOT_FOUND] == [answer.text]


def test_per_role_models(make: Make, fake: FakeOllama) -> None:
    checker_fake = FakeOllama(default=json.dumps(DEFAULTS["verify"]))
    checker = LLM(checker_fake, "checker:7b")
    roles = Roles(read=[found("7,420 people.", "zarnitsa.html")])
    pipeline, roles = make(roles, role_llms={"verify": checker})
    answer = pipeline.run(Q_POP)
    assert "verify" not in roles.roles() and len(checker_fake.calls) == 1
    assert checker_fake.calls[0]["model"] == "checker:7b"
    assert answer.tokens["calls"] == 4  # usage summed over both models
    assert checker.usage.calls == 0  # and reset afterwards
    with pytest.raises(ValueError, match="unknown roles"):
        make(Roles(), role_llms={"judge": checker})

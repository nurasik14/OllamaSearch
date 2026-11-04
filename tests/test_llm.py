from __future__ import annotations

from typing import Any

import pytest

from ollamasearch.llm import LLM, ChatResult, ToolCall, Usage

from .conftest import FakeOllama, tool


def test_chat_records_usage_and_options(fake: FakeOllama) -> None:
    fake.script = ["Hello"]
    llm = LLM(fake, "m", temperature=0.0, seed=7, num_ctx=4096, max_tokens=256, think=False)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result.content == "Hello"
    call = fake.calls[0]
    assert call["options"] == {"temperature": 0.0, "seed": 7, "num_ctx": 4096, "num_predict": 256}
    assert call["think"] is False
    usage = llm.reset_usage()
    assert usage.calls == 1 and usage.eval_tokens == 10 and usage.prompt_tokens == 100
    assert usage.tokens_per_s == pytest.approx(50.0)
    assert llm.usage.calls == 0


def test_generation_is_capped_by_default(fake: FakeOllama) -> None:
    LLM(fake, "m").chat([])
    assert fake.calls[0]["options"]["num_predict"] == 1024
    LLM(fake, "m", max_tokens=None).chat([])
    assert "num_predict" not in fake.calls[1]["options"]


def test_think_none_omits_flag(fake: FakeOllama) -> None:
    LLM(fake, "m", think=None).chat([])
    assert "think" not in fake.calls[0]


def test_think_retry_when_model_does_not_support_it(fake: FakeOllama) -> None:
    def reject(kwargs: dict[str, Any]) -> str:
        raise RuntimeError('"llama3.2:3b" does not support thinking')

    fake.script = [reject, "ok"]
    llm = LLM(fake, "m", think=True)
    assert llm.chat([{"role": "user", "content": "x"}]).content == "ok"
    assert "think" not in fake.calls[1] and llm.think is None


def test_other_errors_propagate(fake: FakeOllama) -> None:
    fake.script = [ConnectionError("down")]
    with pytest.raises(ConnectionError):
        LLM(fake, "m").chat([])


def test_tool_calls_parsed(fake: FakeOllama) -> None:
    fake.script = [{"tool_calls": [tool("web_search", {"query": "x"})]}]
    result = LLM(fake, "m").chat([], tools=[{"type": "function"}])
    assert result.tool_calls == [ToolCall("web_search", {"query": "x"})]
    assert fake.calls[0]["tools"] == [{"type": "function"}]


def test_string_tool_arguments_are_json_decoded(fake: FakeOllama) -> None:
    fake.script = [
        {
            "tool_calls": [
                tool("web_fetch", '{"url": "https://e.com"}'),
                tool("web_fetch", "{broken"),
            ]
        }
    ]
    calls = LLM(fake, "m").chat([]).tool_calls
    assert calls[0].arguments == {"url": "https://e.com"}
    assert calls[1].arguments == "{broken"


def test_stream_calls_back_and_counts_final_chunk(fake: FakeOllama) -> None:
    fake.script = ["one two three"]
    tokens: list[str] = []
    llm = LLM(fake, "m")
    result = llm.stream([], tokens.append)
    assert "".join(tokens) == "one two three" == result.content
    assert llm.usage.calls == 1


def test_unload_uses_keep_alive_zero(fake: FakeOllama) -> None:
    LLM(fake, "qwen3.5:2b").unload()
    assert fake.unloads == ["qwen3.5:2b"]


def test_chat_result_as_message() -> None:
    msg = ChatResult("", tool_calls=[ToolCall("web_search", {"query": "q"}), ToolCall("x", "bad")])
    assert msg.as_message()["tool_calls"][1]["function"]["arguments"] == {}
    assert ChatResult("hi").as_message() == {"role": "assistant", "content": "hi"}


def test_usage_from_dict_response() -> None:
    u = Usage()
    u.add_response({"eval_count": 4, "eval_duration": 2_000_000_000, "total_duration": None})
    assert u.eval_tokens == 4 and u.tokens_per_s == 2.0
    assert Usage().tokens_per_s == 0.0
    assert u.to_dict()["tokens_per_s"] == 2.0
    u.add_response({"done_reason": "length"})
    assert u.truncated == 1


def test_structured_output_format_is_forwarded(fake: FakeOllama) -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    fake.script = ['{"q": "x"}', "{}"]
    llm = LLM(fake, "m")
    assert llm.chat([], format=schema).content == '{"q": "x"}'
    assert fake.calls[0]["format"] == schema
    llm.chat([], format="json")
    assert fake.calls[1]["format"] == "json"
    llm.chat([])
    assert "format" not in fake.calls[2]

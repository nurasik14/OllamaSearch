"""Thin wrapper over the Ollama client that records token counts and timings."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

NS = 1e-9
Message = dict[str, Any]


class ChatClient(Protocol):
    """The subset of :class:`ollama.Client` used here (tests pass a fake)."""

    def chat(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class Usage:
    """Token and timing counters, summed over every model call of one answer."""

    calls: int = 0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    load_s: float = 0.0
    total_s: float = 0.0
    truncated: int = 0  # responses cut off by the max_tokens cap (done_reason == "length")

    def add_response(self, resp: Any) -> None:
        def num(name: str) -> int:
            value = _get(resp, name)
            return int(value) if isinstance(value, (int, float)) else 0

        self.calls += 1
        self.prompt_tokens += num("prompt_eval_count")
        self.eval_tokens += num("eval_count")
        self.prompt_eval_s += num("prompt_eval_duration") * NS
        self.eval_s += num("eval_duration") * NS
        self.load_s += num("load_duration") * NS
        self.total_s += num("total_duration") * NS
        if _get(resp, "done_reason") == "length":
            self.truncated += 1

    @property
    def tokens_per_s(self) -> float:
        return self.eval_tokens / self.eval_s if self.eval_s > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = asdict(self)
        d["tokens_per_s"] = round(self.tokens_per_s, 2)
        return d


@dataclass
class ToolCall:
    name: str
    arguments: Any  # usually a dict; may be a JSON string or junk from small models


@dataclass
class ChatResult:
    content: str
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> Message:
        msg: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": c.name,
                        "arguments": c.arguments if isinstance(c.arguments, dict) else {},
                    }
                }
                for c in self.tool_calls
            ]
        return msg


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in raw or []:
        fn = _get(item, "function")
        name = _get(fn, "name") if fn is not None else None
        args = _get(fn, "arguments") if fn is not None else None
        if isinstance(args, Mapping):
            args = dict(args)
        elif isinstance(args, str):
            with contextlib.suppress(ValueError):
                args = json.loads(args)
        calls.append(ToolCall(name=str(name or ""), arguments=args))
    return calls


class LLM:
    """One model + generation options. Accumulates :class:`Usage` until :meth:`reset_usage`."""

    def __init__(
        self,
        client: ChatClient,
        model: str,
        *,
        think: bool | None = False,
        temperature: float = 0.0,
        seed: int | None = 42,
        num_ctx: int | None = 8192,
        max_tokens: int | None = 1024,
        keep_alive: str | float | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.think = think
        self.keep_alive = keep_alive
        self.options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            self.options["seed"] = seed
        if num_ctx:
            self.options["num_ctx"] = num_ctx
        if max_tokens:  # never let a degenerate model generate forever
            self.options["num_predict"] = max_tokens
        self.usage = Usage()

    def reset_usage(self) -> Usage:
        usage, self.usage = self.usage, Usage()
        return usage

    def _kwargs(self, messages: Sequence[Message], **extra: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "options": self.options,
            **extra,
        }
        if self.think is not None:
            kwargs["think"] = self.think
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        return kwargs

    def _call(self, **kwargs: Any) -> Any:
        try:
            return self.client.chat(**kwargs)
        except Exception as exc:
            # Non-thinking models reject think=True; retry once without the flag.
            if "think" in kwargs and "thinking" in str(exc).lower():
                log.info("%s does not support think=%s; retrying without", self.model, self.think)
                self.think = None
                kwargs.pop("think")
                return self.client.chat(**kwargs)
            raise

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] | None = None,
        format: str | Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """One non-streaming call. ``format`` is passed to Ollama: ``"json"`` or a JSON Schema
        dict for structured output (used by the multi-agent roles and the BFCL runner)."""
        extra: dict[str, Any] = {"tools": list(tools)} if tools else {}
        if format is not None:
            extra["format"] = format if isinstance(format, str) else dict(format)
        resp = self._call(**self._kwargs(messages, **extra))
        self.usage.add_response(resp)
        msg = _get(resp, "message")
        return ChatResult(
            content=str(_get(msg, "content") or ""),
            thinking=str(_get(msg, "thinking") or ""),
            tool_calls=_parse_tool_calls(_get(msg, "tool_calls")),
        )

    def stream(
        self, messages: Sequence[Message], on_token: Callable[[str], None] | None = None
    ) -> ChatResult:
        """Stream the answer, calling ``on_token`` for each content chunk."""
        chunks: Iterator[Any] = self._call(**self._kwargs(messages, stream=True))
        content: list[str] = []
        thinking: list[str] = []
        for chunk in chunks:
            msg = _get(chunk, "message")
            if (t := _get(msg, "thinking")) is not None and t:
                thinking.append(str(t))
            if (c := _get(msg, "content")) is not None and c:
                content.append(str(c))
                if on_token is not None:
                    on_token(str(c))
            if _get(chunk, "done"):
                self.usage.add_response(chunk)
        return ChatResult(content="".join(content), thinking="".join(thinking))

    def unload(self) -> None:
        """Ask Ollama to free the model's memory now (keep_alive=0)."""
        try:
            self.client.chat(model=self.model, messages=[], keep_alive=0)
        except Exception:
            log.debug("unload of %s failed", self.model, exc_info=True)

"""Runtime settings, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

DEFAULT_MODEL = "qwen3.5:2b"


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _think(value: str) -> bool | None:
    """Parse OLLAMASEARCH_THINK: ``false`` (default), ``true`` or ``auto`` (model decides)."""
    v = value.strip().lower()
    if v in {"auto", "none", ""}:
        return None
    return _bool(v)


def _default_cache_dir(env: Mapping[str, str]) -> Path:
    base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ollamasearch"


@dataclass(frozen=True)
class Settings:
    """All tunables in one place. Build with :meth:`from_env` or construct directly in code."""

    ollama_host: str | None = None  # None -> ollama client default (OLLAMA_HOST or localhost)
    model: str = DEFAULT_MODEL
    searxng_url: str = "http://127.0.0.1:8080"
    ollama_api_key: str | None = field(default=None, repr=False)
    backend: str = "searxng"

    # search / fetch
    top_k: int = 5
    search_timeout: float = 15.0
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    max_page_bytes: int = 3_000_000
    fetch_concurrency: int = 8
    per_host_concurrency: int = 2
    use_truststore: bool = True
    cache_dir: Path | None = None  # None disables the on-disk page cache

    # context building
    source_char_budget: int = 1500  # max chars of passages kept per source
    context_char_budget: int = 6000  # max chars of passages across all sources
    passage_chars: int = 500

    # model
    think: bool | None = False
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    # Hard cap on generated tokens per call. Without it Ollama generates until the context
    # is full and then keeps going (context shift), so a degenerate small model never stops.
    max_tokens: int = 1024
    agent_max_steps: int = 4
    llm_timeout: float = 300.0  # max seconds waiting on Ollama (per response / stream chunk)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, dotenv_path: str | Path | None = ".env"
    ) -> Settings:
        """Read settings. Real environment variables win over values from ``dotenv_path``."""
        values: dict[str, str] = {}
        if dotenv_path is not None and Path(dotenv_path).is_file():
            values.update({k: v for k, v in dotenv_values(dotenv_path).items() if v is not None})
        values.update(os.environ if env is None else env)

        kwargs: dict[str, Any] = {}

        def put(key: str, name: str, conv: Any = str) -> None:
            raw = values.get(key)
            if raw is not None and raw.strip() != "":
                kwargs[name] = conv(raw.strip())

        put("OLLAMA_HOST", "ollama_host")
        put("OLLAMASEARCH_MODEL", "model")
        put("SEARXNG_URL", "searxng_url", lambda s: s.rstrip("/"))
        put("OLLAMA_API_KEY", "ollama_api_key")
        put("OLLAMASEARCH_BACKEND", "backend")
        put("OLLAMASEARCH_TOP_K", "top_k", int)
        put("OLLAMASEARCH_SEARCH_TIMEOUT", "search_timeout", float)
        put("OLLAMASEARCH_CONNECT_TIMEOUT", "connect_timeout", float)
        put("OLLAMASEARCH_READ_TIMEOUT", "read_timeout", float)
        put("OLLAMASEARCH_MAX_PAGE_BYTES", "max_page_bytes", int)
        put("OLLAMASEARCH_FETCH_CONCURRENCY", "fetch_concurrency", int)
        put("OLLAMASEARCH_TRUSTSTORE", "use_truststore", _bool)
        put("OLLAMASEARCH_SOURCE_CHARS", "source_char_budget", int)
        put("OLLAMASEARCH_CONTEXT_CHARS", "context_char_budget", int)
        put("OLLAMASEARCH_THINK", "think", _think)
        put("OLLAMASEARCH_TEMPERATURE", "temperature", float)
        put("OLLAMASEARCH_SEED", "seed", int)
        put("OLLAMASEARCH_NUM_CTX", "num_ctx", int)
        put("OLLAMASEARCH_MAX_STEPS", "agent_max_steps", int)
        put("OLLAMASEARCH_LLM_TIMEOUT", "llm_timeout", float)
        put("OLLAMASEARCH_MAX_TOKENS", "max_tokens", int)

        cache = values.get("OLLAMASEARCH_CACHE_DIR")
        if cache is None:
            kwargs["cache_dir"] = _default_cache_dir(values)
        elif cache.strip().lower() not in {"", "none", "off", "0"}:
            kwargs["cache_dir"] = Path(cache).expanduser()

        return cls(**kwargs)

    def with_(self, **changes: Any) -> Settings:
        """Return a copy with some fields replaced (ignores ``None`` values)."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})

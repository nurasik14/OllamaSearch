from __future__ import annotations

from pathlib import Path

from ollamasearch.config import DEFAULT_MODEL, Settings


def test_defaults_without_env(tmp_path: Path) -> None:
    s = Settings.from_env(env={"XDG_CACHE_HOME": str(tmp_path)}, dotenv_path=None)
    assert s.model == DEFAULT_MODEL == "qwen3.5:2b"
    assert s.searxng_url == "http://127.0.0.1:8080"
    assert s.ollama_api_key is None and s.think is False
    assert s.cache_dir == tmp_path / "ollamasearch"


def test_env_overrides() -> None:
    s = Settings.from_env(
        env={
            "OLLAMA_HOST": "http://gpu-box:11434",
            "OLLAMASEARCH_MODEL": "llama3.2:3b",
            "SEARXNG_URL": "http://searx:8888/",
            "OLLAMA_API_KEY": "secret",
            "OLLAMASEARCH_TOP_K": "3",
            "OLLAMASEARCH_THINK": "auto",
            "OLLAMASEARCH_TRUSTSTORE": "no",
            "OLLAMASEARCH_CACHE_DIR": "off",
            "OLLAMASEARCH_CONTEXT_CHARS": "9000",
        },
        dotenv_path=None,
    )
    assert s.ollama_host == "http://gpu-box:11434" and s.model == "llama3.2:3b"
    assert s.searxng_url == "http://searx:8888"
    assert s.top_k == 3 and s.think is None and s.use_truststore is False
    assert s.cache_dir is None and s.context_char_budget == 9000
    assert "secret" not in repr(s)


def test_dotenv_file_and_env_precedence(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OLLAMASEARCH_MODEL=from-file\nOLLAMASEARCH_TOP_K=7\nOLLAMASEARCH_THINK=true\n"
    )
    s = Settings.from_env(env={"OLLAMASEARCH_MODEL": "from-env"}, dotenv_path=dotenv)
    assert s.model == "from-env" and s.top_k == 7 and s.think is True


def test_missing_dotenv_and_blank_values_ignored(tmp_path: Path) -> None:
    s = Settings.from_env(env={"OLLAMASEARCH_MODEL": "  "}, dotenv_path=tmp_path / "missing")
    assert s.model == DEFAULT_MODEL


def test_with_ignores_none() -> None:
    s = Settings().with_(model="x", think=None)
    assert s.model == "x" and s.think is False


def test_multi_verify_setting(tmp_path: Path) -> None:
    assert Settings().multi_verify is True
    env = {"OLLAMASEARCH_MULTI_VERIFY": "false", "XDG_CACHE_HOME": str(tmp_path)}
    assert Settings.from_env(env=env, dotenv_path=None).multi_verify is False

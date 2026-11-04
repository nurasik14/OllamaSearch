"""Download, cache and sample helpers shared by the dataset adapters."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx

from ...dataset import file_sha256

log = logging.getLogger(__name__)

T = TypeVar("T")

USER_AGENT = "ollamasearch-bench (+https://github.com/nurasik14/OllamaSearch)"


class DatasetUnavailableError(RuntimeError):
    """The dataset file could not be downloaded or read."""


@dataclass(frozen=True)
class DatasetFile:
    """One downloaded dataset file, as recorded in meta.json."""

    url: str
    path: Path
    sha256: str
    bytes: int
    expected_sha256: str | None = None

    @property
    def verified(self) -> bool | None:
        """``True``/``False`` against the hash pinned in the adapter; ``None`` if not pinned."""
        return None if self.expected_sha256 is None else self.sha256 == self.expected_sha256


def _new_client(timeout: float) -> httpx.Client:
    """HTTP client for dataset downloads (tests replace this to forbid network access)."""
    return httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout)


def fetch_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> DatasetFile:
    """Download ``url`` to ``dest`` once (atomic rename), then reuse the cached copy.

    A hash different from the pinned one is logged, not fatal: upstream may republish a
    file, and the hash actually used is recorded in meta.json either way.
    """
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        log.info("downloading %s -> %s", url, dest)
        own = client is None
        http = client or _new_client(timeout)
        try:
            with http.stream("GET", url) as resp, tmp.open("wb") as out:
                if resp.status_code != 200:
                    raise DatasetUnavailableError(f"GET {url}: HTTP {resp.status_code}")
                for chunk in resp.iter_bytes():
                    out.write(chunk)
            tmp.replace(dest)
        except httpx.HTTPError as exc:
            raise DatasetUnavailableError(f"cannot download {url}: {exc}") from exc
        finally:
            tmp.unlink(missing_ok=True)
            if own:
                http.close()
    sha = file_sha256(dest)
    if expected_sha256 is not None and sha != expected_sha256:
        log.warning(
            "%s: sha256 %s differs from the pinned %s (upstream changed?)",
            dest,
            sha[:12],
            expected_sha256[:12],
        )
    return DatasetFile(url, dest, sha, dest.stat().st_size, expected_sha256)


def sample_indices(n_total: int, sample: int | None, default: int, seed: int) -> list[int]:
    """Deterministic, prefix-stable sample: a seeded permutation cut to ``sample`` items.

    ``sample=None`` -> ``default``; ``0`` -> all, in file order. Prefix-stable means
    ``--sample 3`` gives the first 3 items of ``--sample 100``, so a run can be extended
    later and resumed without redoing work.
    """
    n = default if sample is None else sample
    order = list(range(n_total))
    if n <= 0:
        return order
    random.Random(seed).shuffle(order)
    return order[:n]


def pick(items: Sequence[T], indices: Sequence[int]) -> list[T]:
    return [items[i] for i in indices]

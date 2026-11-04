"""Dataset adapters. Each module registers itself with :func:`..register` on import.

- ``hotpotqa``: HotpotQA distractor dev (per-question corpus of 10 paragraphs)
- ``rgb-noise``, ``rgb-negative``, ``rgb-counterfactual``: RGB English (5 passages each)
- ``kazqad``: KazQAD Kazakh reading comprehension test (gold + distractor passages)
- ``simpleqa``: SimpleQA (live web through SearXNG)
"""

from __future__ import annotations

from . import hotpotqa, kazqad, rgb, simpleqa

__all__ = ["hotpotqa", "kazqad", "rgb", "simpleqa"]

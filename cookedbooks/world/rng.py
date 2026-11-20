"""Seeded random streams. Every stream is derived with sha256 from (seed, attempt, label).

Never `hash()`, never the global `random` functions, no clock, no OS entropy.
"""

from __future__ import annotations

import hashlib
import random

_DOMAIN = "cookedbooks-world-v1"


def digest_hex(*parts: object) -> str:
    """The sha256 hex digest of the parts joined with '|' (parts rendered with str)."""
    text = "|".join(str(part) for part in (_DOMAIN, *parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream(seed: int, attempt: int, label: str) -> random.Random:
    """An independent `random.Random` for one purpose (`label`) of one generation attempt."""
    return random.Random(int(digest_hex(seed, attempt, label), 16))


def chance(rng: random.Random, percent: int) -> bool:
    """True with probability percent/100, using integers only."""
    return rng.randrange(100) < percent


def weighted_index(rng: random.Random, weights: list[int]) -> int:
    """Pick an index with probability proportional to integer `weights`."""
    point = rng.randrange(sum(weights))
    for index, weight in enumerate(weights):
        point -= weight
        if point < 0:
            return index
    raise AssertionError("unreachable")

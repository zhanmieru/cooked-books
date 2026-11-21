"""Seeded random streams. Every stream is derived with sha256 from (seed, attempt, label).

Never `hash()`, never the global `random` functions, no clock, no OS entropy.
"""

from __future__ import annotations

import hashlib
import json
import random

_DOMAIN = "cookedbooks-world-v1"


def digest_hex(*parts: object) -> str:
    """The sha256 hex digest of the parts joined with '|' (parts rendered with str)."""
    text = "|".join(str(part) for part in (_DOMAIN, *parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream(seed: int, attempt: int, label: str) -> random.Random:
    """An independent `random.Random` for one purpose (`label`) of one generation attempt."""
    return random.Random(int(digest_hex(seed, attempt, label), 16))


def config_salt(generator_config: dict) -> str:
    """A short digest of the generator config (`WorldConfig.generator_config`), mixed into every world
    stream label.

    Without it, two configs run with the same seed draw the same numbers wherever
    their draws line up (the same advances share for the first partial
    allocation of every recipe at seed 0, say), so a measurement over many
    configs at the same seeds has far fewer independent samples than episodes.
    Configs that mean the same thing share a generator config, so they share the books.
    """
    return digest_hex("config", json.dumps(generator_config, sort_keys=True, separators=(",", ":")))[:16]


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

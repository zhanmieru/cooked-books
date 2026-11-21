"""Verified generation: the generator's retry loop under the independent verifier (DESIGN.md section 7).

    result = generate_verified(seed, team_config, output_dir)

Attempt 0 is generated into `output_dir` and checked by
`cookedbooks.verifier.verify_episode`. If the verifier rejects it, the files the
attempt wrote are removed and attempt 1 is generated (the generator derives
its internal seed from `sha256(seed, attempt)`), and so on. The first accepted
attempt stays in `output_dir`; its `provenance.json` records the attempt
number. After `max_attempts` rejections a `RuntimeError` lists the verifier's
problem codes of every attempt. The generator and the verifier are both
deterministic, so the same `(seed, team_config)` always ends on the same
attempt and the same files.

This module lives outside `cookedbooks.world` and `cookedbooks.verifier` on
purpose: the generator must not import the verifier, and the verifier must
not import the generator. Only this module sees both.

`ValueError` (a bad config) and `NotImplementedError` (no injector registered
for a requested error) come from the generator before any file is written;
they pass straight through, without retries. So does any other exception
from the generator or the verifier: a retry is only for a verifier rejection.

`STATS` counts, for the whole process, the episodes generated here, the
attempts the verifier rejected and their problem codes. The verification
report (`python -m cookedbooks.verify`) prints it.
"""

from __future__ import annotations

import copy
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cookedbooks.verifier import verify_episode

DEFAULT_MAX_ATTEMPTS = 20  # DESIGN.md section 7: at most 20 attempts
EPISODE_FILES = (
    "manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json",
    "truth.json", "provenance.json",
)


@dataclass(frozen=True)
class RejectedAttempt:
    """One attempt the verifier rejected, with its sorted problem codes."""

    attempt: int
    codes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    """The accepted attempt and every attempt rejected before it."""

    output_dir: str
    attempt: int
    rejected: tuple[RejectedAttempt, ...] = ()


@dataclass
class GenerationStats:
    """Process-wide counters of verified generation."""

    episodes: int = 0          # episodes accepted
    failed_episodes: int = 0   # (seed, config) pairs that ran out of attempts
    attempts: int = 0          # attempts generated and verified
    rejected: int = 0          # attempts the verifier rejected
    codes: Counter = field(default_factory=Counter)  # problem code -> rejected attempts carrying it

    def record_attempt(self, codes: tuple[str, ...] | None) -> None:
        """Count one verified attempt; `codes` is None when it was accepted."""
        self.attempts += 1
        if codes is not None:
            self.rejected += 1
            self.codes.update(codes)

    def reset(self) -> None:
        self.episodes = self.failed_episodes = self.attempts = self.rejected = 0
        self.codes = Counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes, "failed_episodes": self.failed_episodes, "attempts": self.attempts,
            "rejected": self.rejected, "codes": dict(sorted(self.codes.items())),
        }

    def summary(self) -> str:
        """One line for the verification report."""
        line = (f"generator attempts rejected by the verifier: {self.rejected} of {self.attempts} "
                f"({self.episodes} episodes accepted")
        if self.failed_episodes:
            line += f", {self.failed_episodes} ran out of attempts"
        line += ")"
        if self.codes:
            line += "; codes: " + ", ".join(f"{code} x{n}" for code, n in sorted(self.codes.items()))
        return line


STATS = GenerationStats()


def generate_verified(seed: int, team_config: Any, output_dir: str | os.PathLike, *,
                      max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> GenerationResult:
    """Generate attempts 0, 1, ... into `output_dir` until the verifier accepts one.

    Raises RuntimeError naming each attempt's problem codes after `max_attempts`
    rejections; `output_dir` then holds none of the rejected files.
    """
    # Looked up at call time so that a test can substitute the generator module.
    from cookedbooks.world.generate import generate_episode

    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive int")
    folder = Path(output_dir)
    existing = _entries(folder) - set(EPISODE_FILES)  # never removed: not ours
    rejected: list[RejectedAttempt] = []
    for attempt in range(max_attempts):
        generate_episode(seed, copy.deepcopy(team_config), output_dir, attempt=attempt)
        report = verify_episode(output_dir)
        if report.ok:
            STATS.record_attempt(None)
            STATS.episodes += 1
            return GenerationResult(str(output_dir), attempt, tuple(rejected))
        codes = tuple(sorted(report.codes())) or ("<no code>",)
        STATS.record_attempt(codes)
        rejected.append(RejectedAttempt(attempt, codes))
        _clear(folder, keep=existing)
    STATS.failed_episodes += 1
    listing = "; ".join(f"attempt {r.attempt}: {', '.join(r.codes)}" for r in rejected)
    raise RuntimeError(f"the verifier rejected all {max_attempts} generation attempts for seed {seed} ({listing})")


def _entries(folder: Path) -> set[str]:
    return set(os.listdir(folder)) if folder.is_dir() else set()


def _clear(folder: Path, *, keep: set[str]) -> None:
    """Remove what a rejected attempt wrote: every entry of `folder` not in `keep`."""
    for name in sorted(_entries(folder) - keep):
        path = folder / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

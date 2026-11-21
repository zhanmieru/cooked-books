"""Baseline diagnostics: precision, recall, F1, return, steps and time per policy.

Each available policy runs once per open episode of a split, in `closed`
mode, across the process boundary; the metrics come from
`judge_adapter.score_episode` on the judge side. These are diagnostics of
scripted baselines on the open episodes, not a claim about training.

    python -m cookedbooks.runner.diagnostics [--split dev|train|both]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cookedbooks.runner.episodes import EPISODES_DIR, episodes
from cookedbooks.runner.run import run_episode
from cookedbooks.runner.trajectories import POLICY_SEED, policies

MODE = "closed"


@dataclass(frozen=True)
class PolicyStats:
    policy: str
    episodes: int
    precision: float
    recall: float
    f1: float
    episode_return: float
    inspection_steps: float
    wall_time_s: float
    failures: int


def _number(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def measure(split: str, root: Path = EPISODES_DIR, names: list[str] | None = None) -> list[PolicyStats]:
    """Run every policy on every episode of `split` and average per policy."""
    out = []
    entries = episodes(split, root)
    for policy in names if names is not None else policies():
        runs = [run_episode(policy, e.folder(root), mode=MODE, policy_seed=POLICY_SEED) for e in entries]
        n = len(runs) or 1

        def mean(values: list[float]) -> float:
            return sum(values) / n

        out.append(PolicyStats(
            policy=policy, episodes=len(runs),
            precision=mean([_number(r.metrics.get("precision"), 0.0) for r in runs]),
            recall=mean([_number(r.metrics.get("recall"), 0.0) for r in runs]),
            f1=mean([_number(r.metrics.get("f1"), 0.0) for r in runs]),
            episode_return=mean([r.episode_return for r in runs]),
            inspection_steps=mean([float(r.inspection_steps) for r in runs]),
            wall_time_s=mean([r.wall_time_s for r in runs]),
            failures=sum(1 for r in runs if r.failure is not None),
        ))
    return out


def table(stats: list[PolicyStats]) -> list[str]:
    """Markdown table lines."""
    lines = ["| Policy | Precision | Recall | F1 | Return | Steps | Time per episode, s | Episodes |",
             "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        episodes_cell = str(s.episodes) + (f" ({s.failures} failed)" if s.failures else "")
        lines.append(f"| {s.policy} | {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {s.episode_return:+.3f} "
                     f"| {s.inspection_steps:.1f} | {s.wall_time_s:.3f} | {episodes_cell} |")
    return lines


def section(splits: tuple[str, ...] = ("dev", "train"), root: Path = EPISODES_DIR) -> str:
    """The report section: one table per split."""
    lines = [
        "## Baseline diagnostics",
        "",
        "Scripted baselines on the open episodes in `closed` mode, each policy in its own restricted process,",
        "metrics from `judge_adapter.score_episode` on the judge side. These numbers are diagnostics of the",
        "baselines, not a claim about training. Time is wall-clock from reset to the end of the episode",
        "and varies between machines. Precision and recall are averaged over episodes; on clean books an",
        "empty answer has P = R = 0 and F1 = 1 by the contract, so even a perfect policy shows mean P and R",
        "below 1 when the split has clean episodes.",
        "",
    ]
    for split in splits:
        stats = measure(split, root)
        lines += [f"{split.capitalize()} episodes ({len(episodes(split, root))}):", ""] + table(stats) + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.runner.diagnostics",
                                     description="Print the baseline diagnostics table.")
    parser.add_argument("--split", choices=("dev", "train", "both"), default="both")
    args = parser.parse_args(argv)
    splits = ("dev", "train") if args.split == "both" else (args.split,)
    print(section(splits), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

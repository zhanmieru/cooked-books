"""`trajectories.jsonl`: every baseline policy, in train mode, on the ten train episodes.

One JSON object per step with exactly the fields
`episode_id, policy, split, step, action, observation, reward, terminated, truncated`.
Step 0 of each run is the reset (`action: null`, reward 0). No wall-clock
time, no paths, no seeds. The policy list is data: `POLICIES` names the
baselines in order and a name whose module does not exist is skipped.

    python -m cookedbooks.runner.trajectories            # rewrite the file
    python -m cookedbooks.runner.trajectories --check    # a fresh build must equal the committed file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from cookedbooks.runner.episodes import EPISODES_DIR, REPO_ROOT, episodes
from cookedbooks.runner.remote import available_policies
from cookedbooks.runner.run import run_episode

TRAJECTORIES_PATH = REPO_ROOT / "trajectories.jsonl"
POLICIES = ("random_explorer", "simple_recon", "flag_all", "oracle", "evidence_solver")
POLICY_SEED = 0
SPLIT = "train"
MODE = "train"
FIELDS = ("episode_id", "policy", "split", "step", "action", "observation", "reward", "terminated", "truncated")


def policies() -> list[str]:
    """The names in POLICIES whose module exists, in POLICIES order."""
    present = set(available_policies())
    return [name for name in POLICIES if name in present]


def records(root: Path = EPISODES_DIR) -> Iterator[dict]:
    """Every trajectory record, policy by policy, episode by episode (index order)."""
    for policy in policies():
        for entry in episodes(SPLIT, root):
            run = run_episode(policy, entry.folder(root), mode=MODE, policy_seed=POLICY_SEED)
            if run.failure is not None:
                raise RuntimeError(f"{policy} failed on {entry.name}: {run.failure}")
            for record in run.records:
                line = {"episode_id": run.episode_id, "policy": policy, "split": SPLIT, **record}
                yield {name: line[name] for name in FIELDS}


def render(root: Path = EPISODES_DIR) -> str:
    """The whole file as text."""
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                   for r in records(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.runner.trajectories",
                                     description="Build trajectories.jsonl from the baselines on the train episodes.")
    parser.add_argument("--check", action="store_true", help="compare a fresh build with the committed file")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        same = TRAJECTORIES_PATH.is_file() and TRAJECTORIES_PATH.read_text(encoding="utf-8") == text
        print("trajectories: OK" if same else "trajectories: differs from a fresh build")
        return 0 if same else 1
    TRAJECTORIES_PATH.write_text(text, encoding="utf-8")
    size = TRAJECTORIES_PATH.stat().st_size
    print(f"trajectories: {text.count(chr(10))} lines, {size} bytes, policies {', '.join(policies())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

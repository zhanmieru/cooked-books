"""The open episodes: ten train and ten dev folders generated from `episodes/index.json`.

Each index entry gives `split`, `name`, `seed`, a short `note` and the exact
`team_config`; the folder `episodes/<split>/<name>/` holds the eight files
`judge_adapter.generate_episode(seed, team_config, folder)` writes. These are
the open episodes, so `truth.json` and `provenance.json` are included.

    python -m cookedbooks.runner.episodes            # regenerate every folder
    python -m cookedbooks.runner.episodes --check    # fresh generation must equal the committed bytes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODES_DIR = REPO_ROOT / "episodes"
INDEX_PATH = EPISODES_DIR / "index.json"
SPLITS = ("train", "dev")
EPISODE_FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json",
                 "truth.json", "provenance.json")


@dataclass(frozen=True)
class OpenEpisode:
    split: str
    name: str
    seed: int
    note: str
    team_config: dict

    def folder(self, root: Path = EPISODES_DIR) -> Path:
        return root / self.split / self.name


def load_index(path: Path = INDEX_PATH) -> list[OpenEpisode]:
    """The index entries in file order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [OpenEpisode(e["split"], e["name"], int(e["seed"]), e.get("note", ""), e["team_config"])
            for e in data["episodes"]]


def episodes(split: str | None = None, root: Path = EPISODES_DIR) -> list[OpenEpisode]:
    """Index entries of one split (or all), in index order."""
    return [e for e in load_index(root / "index.json") if split is None or e.split == split]


def _adapter():
    try:
        import judge_adapter
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT))
        import judge_adapter
    return judge_adapter


def generate(entries: list[OpenEpisode], root: Path) -> None:
    """Generate every entry into `root/<split>/<name>/` (existing folders are replaced)."""
    adapter = _adapter()
    for entry in entries:
        folder = entry.folder(root)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        adapter.generate_episode(entry.seed, json.loads(json.dumps(entry.team_config)), str(folder))


def check(root: Path = EPISODES_DIR) -> list[str]:
    """Differences between the committed folders and a fresh generation; empty means identical."""
    entries = load_index(root / "index.json")
    problems: list[str] = []
    expected_dirs = {(e.split, e.name) for e in entries}
    for split in SPLITS:
        split_dir = root / split
        present = {(split, p.name) for p in split_dir.iterdir() if p.is_dir()} if split_dir.is_dir() else set()
        for split_name in sorted(present - expected_dirs):
            problems.append(f"{split_name[0]}/{split_name[1]}: folder not in the index")
    with tempfile.TemporaryDirectory(prefix="cookedbooks-episodes-check-") as tmp:
        fresh = Path(tmp)
        generate(entries, fresh)
        for entry in entries:
            committed, new = entry.folder(root), entry.folder(fresh)
            label = f"{entry.split}/{entry.name}"
            if not committed.is_dir():
                problems.append(f"{label}: missing")
                continue
            names = sorted(p.name for p in committed.iterdir())
            if names != sorted(EPISODE_FILES):
                problems.append(f"{label}: files {names}")
            for name in EPISODE_FILES:
                a, b = committed / name, new / name
                if not a.is_file() or a.read_bytes() != b.read_bytes():
                    problems.append(f"{label}/{name}: differs from a fresh generation")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.runner.episodes",
                                     description="Regenerate or check the open train/dev episodes.")
    parser.add_argument("--check", action="store_true", help="compare the committed folders with a fresh generation")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print(problem)
        print("episodes: OK" if not problems else f"episodes: {len(problems)} problem(s)")
        return 0 if not problems else 1
    entries = load_index()
    for split in SPLITS:
        if (EPISODES_DIR / split).is_dir():
            shutil.rmtree(EPISODES_DIR / split)
    generate(entries, EPISODES_DIR)
    print(f"episodes: generated {len(entries)} folders under {EPISODES_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

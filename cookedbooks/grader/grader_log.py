"""The grader's journal (EVALUATION_CONTRACT.md, section 6, «Журнал оценщика»).

One JSON line per `score_episode` call, appended to the file named by the
environment variable `COOKEDBOOKS_GRADER_LOG`, or to
`<tempfile.gettempdir()>/cookedbooks/grader_log.jsonl` when it is unset or
empty. Nothing is ever written inside the episode folder: a configured path
that falls inside it is replaced by the default. Lines carry no wall-clock
time. Writing is best effort: any failure is swallowed, scoring never breaks.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

LOG_ENV_VAR = "COOKEDBOOKS_GRADER_LOG"


def default_log_path() -> Path:
    return Path(tempfile.gettempdir()) / "cookedbooks" / "grader_log.jsonl"


def _is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except (ValueError, OSError):
        return False
    return True


def log_path(episode_dir: str | os.PathLike | None = None) -> Path:
    """Where the next line goes: the env var path, or the default, never inside `episode_dir`."""
    configured = os.environ.get(LOG_ENV_VAR)
    path = Path(configured) if configured else default_log_path()
    if episode_dir is not None and _is_inside(path, Path(episode_dir)):
        path = default_log_path()
    return path


def write_entry(entry: dict, episode_dir: str | os.PathLike | None = None) -> bool:
    """Append `entry` as one JSON line. Returns False (and never raises) if that failed."""
    try:
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        path = log_path(episode_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - a broken log must never break scoring
        return False
    return True

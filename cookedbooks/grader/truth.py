"""Read `truth.json` for the grader (EVALUATION_CONTRACT.md, section 2).

The jury hand-writes and relabels truth files, and a crash here would fail a
whole block of their tests, so this reader never rejects a single issue:

- both the wrapper `{"issues": [...]}` and a bare list of issues are accepted;
- per issue the grader uses `issue_id`, `type`, `affected_ids`,
  `expected_amount_minor`, `currency` and `evidence_sets`; unknown keys are
  ignored;
- an integer `issue_id` is used as `str(issue_id)`; a missing, null or other
  non-string `issue_id` becomes `"#<file position>"` (both with a warning);
- an issue with a missing or malformed field, or one that is not an object at
  all, is kept with `matchable: False`: it counts as a false negative and no
  finding can match it; its warnings name the field;
- `evidence_sets` missing, null or `[]` is not malformed, but such an issue
  can never be matched either (condition 5 needs a set to cover).

Every returned issue has the keys `issue_id` (str), `type`, `affected_ids`,
`expected_amount_minor`, `currency`, `evidence_sets`, `matchable` (bool) and
`warnings` (list of str). The five data fields are only guaranteed to have
their contract types when `matchable` is True. Nothing is stripped, case
folded or converted apart from the `issue_id` rule above.

ValueError is raised only when the file is not an episode truth at all:
missing, unreadable, not UTF-8, not JSON, or neither a list nor an object
with an `issues` list. There is no check that an issue "makes accounting
sense" and no cross-check against the public files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

TRUTH_FILE = "truth.json"
DATA_FIELDS = ("type", "affected_ids", "expected_amount_minor", "currency")


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def _is_list_of_str(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _issue_id(raw: dict, position: int) -> tuple[str, list[str]]:
    value = raw.get("issue_id")
    if isinstance(value, str):
        return value, []
    if _is_int(value):
        return str(value), [f"issue at position {position}: integer issue_id {value} used as {str(value)!r}"]
    synthetic = f"#{position}"
    what = "missing or null" if value is None else "not a string"
    return synthetic, [f"issue at position {position}: issue_id is {what}; using {synthetic!r}"]


def _field_problems(raw: dict) -> list[str]:
    problems = [f"{field} is missing" for field in DATA_FIELDS if field not in raw]
    for field in ("type", "currency"):
        if field in raw and not isinstance(raw[field], str):
            problems.append(f"{field} is not a string")
    if "affected_ids" in raw and not _is_list_of_str(raw["affected_ids"]):
        problems.append("affected_ids is not a list of strings")
    if "expected_amount_minor" in raw and not _is_int(raw["expected_amount_minor"]):
        problems.append("expected_amount_minor is not an integer")
    evidence_sets = raw.get("evidence_sets")
    if evidence_sets is not None and not (
        isinstance(evidence_sets, list) and all(_is_list_of_str(s) for s in evidence_sets)
    ):
        problems.append("evidence_sets is not a list of lists of strings")
    return problems


def parse_issue(raw: object, position: int) -> dict:
    """Return the grader's view of one issue. Never raises; see the module docstring."""
    if not isinstance(raw, dict):
        issue_id = f"#{position}"
        return {
            "issue_id": issue_id, "type": None, "affected_ids": [], "expected_amount_minor": None,
            "currency": None, "evidence_sets": [], "matchable": False,
            "warnings": [f"issue {issue_id!r} is not an object; it can never be matched"],
        }
    issue_id, warnings = _issue_id(raw, position)
    problems = _field_problems(raw)
    warnings.extend(f"issue {issue_id!r}: {p}; it can never be matched" for p in problems)
    evidence_sets = raw.get("evidence_sets")
    if not problems and not evidence_sets:
        warnings.append(f"issue {issue_id!r} has no evidence sets and can never be matched")
    return {
        "issue_id": issue_id,
        "type": raw.get("type"),
        "affected_ids": list(raw["affected_ids"]) if _is_list_of_str(raw.get("affected_ids")) else [],
        "expected_amount_minor": raw.get("expected_amount_minor"),
        "currency": raw.get("currency"),
        "evidence_sets": [list(s) for s in evidence_sets] if not problems and evidence_sets else [],
        "matchable": not problems,
        "warnings": warnings,
    }


def parse_truth(value: object) -> list[dict]:
    """Turn decoded truth JSON (wrapper object or bare list) into a list of issues, in file order."""
    if isinstance(value, dict):
        if "issues" not in value:
            raise ValueError("truth.json object has no issues list")
        value = value["issues"]
    if not isinstance(value, list):
        raise ValueError("truth.json issues must be a list")
    return [parse_issue(raw, position) for position, raw in enumerate(value)]


def load_truth(episode_dir: str | os.PathLike) -> list[dict]:
    """Read `<episode_dir>/truth.json` and return its issues in file order."""
    path = Path(episode_dir) / TRUTH_FILE
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
    except FileNotFoundError:
        raise ValueError("episode folder has no truth.json") from None
    except UnicodeDecodeError:
        raise ValueError("truth.json is not valid UTF-8") from None
    except OSError:
        raise ValueError("cannot read truth.json") from None
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"truth.json is not valid JSON: {exc}") from None
    return parse_truth(value)

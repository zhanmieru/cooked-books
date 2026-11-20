"""Score a submission (EVALUATION_CONTRACT.md, sections 5 and 6).

`score_findings` is the pure core. `score_episode` reads an episode folder
(public files through the loader, then `truth.json`), scores, appends one line
to the grader log and returns exactly the seven contract keys.

The scorer is generic: it uses truth, the public IDs, the policy tolerances
and the findings, nothing else, and works on folders the jury built by hand
(five records, empty or missing `policy.json`, a bare-list `truth.json`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from cookedbooks.episode.loader import load_public
from cookedbooks.findings import validate_findings
from cookedbooks.grader import grader_log
from cookedbooks.grader.matcher import match
from cookedbooks.grader.reward import compute_metrics, structural_error_output, to_output
from cookedbooks.grader.truth import load_truth


def check_inspection_steps(inspection_steps: object) -> int:
    """A non-negative int (not bool). Anything else is a caller bug: ValueError."""
    if not isinstance(inspection_steps, int) or isinstance(inspection_steps, bool) or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative integer")
    return inspection_steps


def resolve_tolerances(raw: object) -> tuple[dict[str, int], list[str]]:
    """Clean `policy.amount_tolerance_minor`: keep non-negative ints, drop the rest (tau 0) with a warning.

    A missing tolerance is 0 without a warning.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        return {}, ["amount_tolerance_minor is not an object; all tolerances are 0"]
    tolerances: dict[str, int] = {}
    warnings: list[str] = []
    for key in sorted(raw, key=str):
        value = raw[key]
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            tolerances[key] = value
        else:
            warnings.append(f"amount_tolerance_minor[{key!r}] is not a non-negative integer; using 0")
    return tolerances, warnings


def _truth_warnings(issues: Sequence[Mapping]) -> list[str]:
    """Warnings the truth parser attached to issues, plus repeated issue_ids."""
    warnings: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        issue_id = issue["issue_id"]
        if issue_id in seen:
            warnings.append(f"issue_id {issue_id!r} appears more than once in truth")
        seen.add(issue_id)
        if "warnings" in issue:
            warnings.extend(issue["warnings"])
        elif not issue["evidence_sets"]:
            warnings.append(f"issue {issue_id!r} has no evidence sets and can never be matched")
    return warnings


def score_findings(
    *,
    issues: Sequence[Mapping],
    existing_ids: AbstractSet[str],
    tolerances: object,
    findings: object,
    inspection_steps: int,
    manifest_episode_id: object = None,
) -> dict:
    """Validate and score one submission. No I/O.

    `findings` is the whole submitted object (`{"episode_id", "findings"}`).
    `tolerances` is the raw `policy.amount_tolerance_minor` value (or None).
    Returns a dict with:
    - `metrics`: the seven output keys (`structural_error_output()` if invalid);
    - `exact`: the `reward.Metrics` with exact fractions, or None if invalid;
    - `pairs`: sorted `[issue_id, finding_index]` lists;
    - `unmatched`: `{"finding_index", "reason", "issue_id"}` per unmatched finding;
    - `warnings`: sorted-by-occurrence strings;
    - `structural_error`: the reason code from `validate_findings`, or None.
    Raises ValueError for a bad `inspection_steps`.
    """
    steps = check_inspection_steps(inspection_steps)
    error = validate_findings(findings)
    if error is not None:
        return {
            "metrics": structural_error_output(),
            "exact": None,
            "pairs": [],
            "unmatched": [],
            "warnings": [],
            "structural_error": error,
        }

    clean_tolerances, warnings = resolve_tolerances(tolerances)
    warnings.extend(_truth_warnings(issues))
    submitted_id = findings["episode_id"]
    if manifest_episode_id is not None and submitted_id != manifest_episode_id:
        warnings.append("submitted episode_id differs from manifest episode_id")

    items = findings["findings"]
    result = match(items, issues, existing_ids, clean_tolerances)
    tp = result.tp
    exact = compute_metrics(tp, len(items) - tp, len(issues) - tp, steps)
    return {
        "metrics": to_output(exact),
        "exact": exact,
        "pairs": [[issue_id, index] for issue_id, index in result.pairs],
        "unmatched": [
            {"finding_index": u.finding_index, "reason": u.reason, "issue_id": u.issue_id}
            for u in result.unmatched
        ],
        "warnings": warnings,
        "structural_error": None,
    }


def score_episode(episode_dir: str | os.PathLike, findings: object, inspection_steps: int) -> dict:
    """Score `findings` against `<episode_dir>/truth.json`; return exactly the seven contract keys.

    Raises ValueError for a bad `inspection_steps` or an unreadable episode folder.
    A structurally invalid submission is not an exception: it scores
    `return = -1.0` with the other six keys None.
    """
    steps = check_inspection_steps(inspection_steps)
    public = load_public(episode_dir)
    issues = load_truth(episode_dir)
    manifest = public.manifest
    policy = public.policy
    result = score_findings(
        issues=issues,
        existing_ids=public.existing_ids(),
        tolerances=policy.get("amount_tolerance_minor"),
        findings=findings,
        inspection_steps=steps,
        manifest_episode_id=manifest.get("episode_id"),
    )
    entry = {
        "episode_id": manifest.get("episode_id"),
        "inspection_steps": steps,
        "metrics": result["metrics"],
        "pairs": result["pairs"],
        "unmatched": result["unmatched"],
        "warnings": result["warnings"],
        "structural_error": result["structural_error"],
    }
    grader_log.write_entry(entry, episode_dir)
    return dict(result["metrics"])

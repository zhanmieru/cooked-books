"""The perfect answer, built from truth (EVALUATION_CONTRACT.md, section 2, truth.json).

One finding per issue in `issue_id` order (code points), fields copied from
the issue, `evidence_ids` = the issue's first evidence set, a fixed short
explanation, and the manifest's `episode_id`. Issues that can never be
matched (malformed in truth.json, or without evidence sets) are skipped: a
finding for them could only be a false positive. Judge side only: it reads `truth.json`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from cookedbooks.episode.loader import load_public
from cookedbooks.grader.truth import load_truth

ORACLE_EXPLANATION = "Oracle finding copied from ground truth."


def oracle_from_issues(issues: Sequence[Mapping], episode_id: str) -> dict:
    """Build the oracle submission for `issues` (no I/O)."""
    order = sorted(range(len(issues)), key=lambda k: (issues[k]["issue_id"], k))
    findings = []
    for k in order:
        issue = issues[k]
        evidence_sets = issue.get("evidence_sets") or []
        if issue.get("matchable", True) is False or not evidence_sets:
            continue
        findings.append({
            "type": issue["type"],
            "affected_ids": list(issue["affected_ids"]),
            "claimed_amount_minor": issue["expected_amount_minor"],
            "currency": issue["currency"],
            "evidence_ids": list(evidence_sets[0]),
            "explanation": ORACLE_EXPLANATION,
        })
    return {"episode_id": episode_id, "findings": findings}


def oracle_findings(episode_dir: str | os.PathLike) -> dict:
    """The oracle submission for the episode in `episode_dir`."""
    episode_id = load_public(episode_dir).manifest.get("episode_id")
    if not isinstance(episode_id, str):
        episode_id = ""
    return oracle_from_issues(load_truth(episode_dir), episode_id)

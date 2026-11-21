"""Generated truths never contain "twin" issues that one finding could match.

With twins, a repeated finding can legitimately earn a second TP (jury example
M11), and "a duplicate is always a FP" stops being true. The jury's rules allow
that, but our generator should never produce it: every issue must be
distinguishable by a finding. Checked on every recipe in configs/matrix/ for
several seeds (default 10, about 15 s in all). More: COOKEDBOOKS_TRUTH_SEEDS=100.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.episode.loader import load_public
from cookedbooks.grader.matcher import can_share_finding
from cookedbooks.grader.score import resolve_tolerances
from cookedbooks.grader.truth import load_truth

REPO = Path(__file__).resolve().parents[2]
RECIPES = [f"D{k:02d}" for k in range(1, 21)]
SEEDS = int(os.environ.get("COOKEDBOOKS_TRUTH_SEEDS", "10"))


def recipe(name: str) -> dict:
    return json.loads((REPO / "configs" / "matrix" / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", RECIPES)
def test_no_finding_can_match_two_issues(tmp_path, name):
    for seed in range(SEEDS):
        folder = tmp_path / f"{name}-{seed}"
        judge_adapter.generate_episode(seed, recipe(name), str(folder))
        public = load_public(folder)
        tolerances, warnings = resolve_tolerances(public.policy.get("amount_tolerance_minor"))
        assert warnings == []
        issues = load_truth(folder)
        assert all(issue["matchable"] and not issue["warnings"] for issue in issues)

        keys = [(issue["type"], frozenset(issue["affected_ids"])) for issue in issues]
        assert len(set(keys)) == len(keys), f"{name} seed {seed}: two issues share type and affected IDs"

        for a, b in itertools.combinations(issues, 2):
            assert not can_share_finding(a, b, public.existing_ids(), tolerances), (
                f"{name} seed {seed}: one finding can match {a['issue_id']!r} and {b['issue_id']!r}")


def test_check_is_not_vacuous(tmp_path):
    """The recipes do put several issues in one truth, so the pairwise check has work to do."""
    sizes = []
    for name in RECIPES:
        folder = tmp_path / name
        judge_adapter.generate_episode(0, recipe(name), str(folder))
        sizes.append(len(load_truth(folder)))
    assert sum(1 for n in sizes if n >= 2) >= 10

"""evidence_solver against the real environment on every recipe D01-D20.

The solver sees an episode only through `judge_adapter.make_env()` in closed
mode; its answer is scored by `judge_adapter.score_episode`. Target, measured
and asserted: precision 1 and recall 1 for every error type and both
scenarios, so the return is exactly `1 - pages_read / 500`, and the action
and time budgets are never reached.

Default: 20 seeds per recipe (about 20 s). More: COOKEDBOOKS_SOLVER_SEEDS=300.
The 300-seed sweep (6 000 episodes, 9 900 errors) was clean when this was written.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.verify.attacks import exact_return, expected_counts

REPO = Path(__file__).resolve().parents[2]
SOLVER_PATH = REPO / "cookedbooks" / "policies" / "evidence_solver.py"
RECIPES = [f"D{k:02d}" for k in range(1, 21)]
SEEDS = int(os.environ.get("COOKEDBOOKS_SOLVER_SEEDS", "20"))
MAX_PAGES = 70          # 100 actions available; D20 needs 51 pages, the 12-month worst case 63
MAX_SECONDS = 5.0       # 60 available


def load_solver():
    name = "evidence_solver_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOLVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader_log.jsonl"))


def recipe(name: str) -> dict:
    return json.loads((REPO / "configs" / "matrix" / f"{name}.json").read_text(encoding="utf-8"))


def solve(episode_dir: str | Path, seed: int = 0) -> dict:
    """Run the solver through the environment; return metrics, steps, answer and timing."""
    env = judge_adapter.make_env()
    start = time.monotonic()
    obs, _ = env.reset(seed, {"mode": "closed", "episode_dir": str(episode_dir)})
    policy = load_solver().make_policy(seed)
    actions = []
    while True:
        action = policy.act(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0 and info == {}
        if terminated or truncated:
            break
    elapsed = time.monotonic() - start
    assert terminated and not truncated, "the solver must end the episode with its own submit"
    assert obs["error"] is None and obs["result"] == {"accepted": True}
    assert all(a["name"] == "list_records" for a in actions[:-1]) and actions[-1]["name"] == "submit"
    answer = env.final_findings
    metrics = judge_adapter.score_episode(str(episode_dir), answer, env.inspection_steps)
    return {"metrics": metrics, "steps": env.inspection_steps, "answer": answer, "seconds": elapsed}


def truth_issues(episode_dir: Path) -> list[dict]:
    raw = json.loads((episode_dir / "truth.json").read_text(encoding="utf-8"))
    return raw["issues"] if isinstance(raw, dict) else raw


def existing_and_tolerances(episode_dir: Path) -> tuple[frozenset[str], dict]:
    from cookedbooks.episode.loader import load_public

    public = load_public(episode_dir)
    return public.existing_ids(), public.policy.get("amount_tolerance_minor") or {}


def per_type_counts(episode_dir: Path, answer: dict) -> Counter:
    """TP/FP/FN per error type, from an independent matching restricted to each type."""
    issues = truth_issues(episode_dir)
    existing, tolerances = existing_and_tolerances(episode_dir)
    counts: Counter = Counter()
    for kind in ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error"):
        sub = {"findings": [f for f in answer["findings"] if f["type"] == kind]}
        tp, fp, fn = expected_counts(sub, [i for i in issues if i["type"] == kind], existing, tolerances)
        counts[(kind, "tp")] += tp
        counts[(kind, "fp")] += fp
        counts[(kind, "fn")] += fn
    return counts


def check_perfect(run: dict, where: str) -> None:
    m = run["metrics"]
    assert (m["fp"], m["fn"]) == (0, 0), f"{where}: {m}"
    assert m["return"] == float(exact_return(m["tp"], 0, 0, run["steps"])), where
    assert run["steps"] <= MAX_PAGES and run["seconds"] < MAX_SECONDS, where


@pytest.mark.parametrize("name", RECIPES)
def test_solver_is_perfect_on_recipe(name, tmp_path, capsys):
    """Every error found with canonical fields and evidence, no look-alike reported, within budgets."""
    config = recipe(name)
    totals: Counter = Counter()
    returns: list[Fraction] = []
    pages: list[int] = []
    for seed in range(SEEDS):
        folder = tmp_path / f"{name}-{seed}"
        judge_adapter.generate_episode(seed, config, str(folder))
        run = solve(folder, seed)
        check_perfect(run, f"{name} seed {seed}")
        m = run["metrics"]
        totals.update({"tp": m["tp"], "fp": m["fp"], "fn": m["fn"]})
        totals.update(per_type_counts(folder, run["answer"]))
        returns.append(exact_return(m["tp"], m["fp"], m["fn"], run["steps"]))
        pages.append(run["steps"])
    mean = sum(returns) / len(returns)
    assert mean == 1 - Fraction(sum(pages), 500 * len(pages))
    for kind in ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error"):
        assert totals[(kind, "fp")] == 0 and totals[(kind, "fn")] == 0, (name, kind)
    with capsys.disabled():
        print(f"\n{name}: seeds {SEEDS}, TP {totals['tp']} FP {totals['fp']} FN {totals['fn']}, "
              f"mean return {float(mean):.4f}, pages {min(pages)}-{max(pages)}")


SIZE_CONFIGS = {
    "small_200_rows": {"n_ledger_rows": 200, "n_documents": 40, "currencies": ["KZT", "USD"],
                       "errors": {"duplicate_invoice": 1, "unallocated_payment": {"missing": 1, "partial": 1},
                                  "fx_error": {"wrong_date": 1, "wrong_pair": 0}, "cutoff_error": {"late": 1}},
                       "distractors": {"boundary": 2, "overpayment": 1}},
    "large_2000_rows_everything": {
        "n_ledger_rows": 2000, "n_documents": 200, "currencies": ["KZT", "USD", "EUR"],
        "errors": {"duplicate_invoice": 2, "unallocated_payment": {"missing": 2, "partial": 2},
                   "fx_error": {"wrong_date": 2, "wrong_pair": 2}, "cutoff_error": {"late": 2, "early": 2}},
        "scenarios": {"S1": 2, "S2": 2},
        "distractors": {"split_payment": 2, "refund": 2, "recurring_amount": 2, "boundary": 8, "advance": 2,
                        "overpayment": 2}},
}
# The largest books the generator accepts: 2000 rows, 200 documents, three currencies, twelve months of rates.
SIZE_CONFIGS["worst_case_12_months"] = {**SIZE_CONFIGS["large_2000_rows_everything"], "period_months": 12}


@pytest.mark.parametrize("label", sorted(SIZE_CONFIGS))
def test_solver_on_200_and_2000_rows(label, tmp_path):
    for seed in range(max(3, SEEDS // 4)):
        folder = tmp_path / f"{label}-{seed}"
        judge_adapter.generate_episode(seed, SIZE_CONFIGS[label], str(folder))
        run = solve(folder, seed)
        check_perfect(run, f"{label} seed {seed}")
        assert run["metrics"]["tp"] == len(truth_issues(folder)) > 0


TRANSFORM_CASES = [("D11", "hex"), ("D16", "int"), ("D17", "awkward"), ("D20", "shuffle"), ("D19", "awkward")]


@pytest.mark.parametrize("name,style", TRANSFORM_CASES)
def test_solver_on_relabelled_and_permuted_copy(name, style, tmp_path):
    """Same counts and the same return on a copy with every ID replaced and every table reordered."""
    transform = pytest.importorskip("cookedbooks.verify.transform")
    for seed in range(2):
        source = tmp_path / f"{name}-{seed}"
        judge_adapter.generate_episode(seed, recipe(name), str(source))
        copy_dir = tmp_path / f"{name}-{seed}-{style}"
        mapping = transform.transform_episode(source, copy_dir, permute=True, relabel=True, seed=seed + 7, style=style)
        original, relabelled = solve(source, seed), solve(copy_dir, seed)
        check_perfect(relabelled, f"{name} seed {seed} {style}")
        assert relabelled["metrics"] == original["metrics"] and relabelled["steps"] == original["steps"]
        if not transform.is_ambiguous(mapping):
            restored = transform.restore_findings(relabelled["answer"], mapping)
            key = lambda f: (f["type"], frozenset(f["affected_ids"]), f["claimed_amount_minor"], f["currency"],
                             frozenset(f["evidence_ids"]))
            assert {key(f) for f in restored["findings"]} == {key(f) for f in original["answer"]["findings"]}

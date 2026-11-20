"""The generator against the independent verifier, end to end through the public entry points.

For every recipe D01-D11 and a grid of other configs, over many seeds:
- `cookedbooks.verifier.verify_episode(dir).ok` (the verifier was written from
  DESIGN.md alone and never imports the generator);
- truth.json holds exactly the requested number of issues;
- the oracle answer, submitted through `judge_adapter.make_env()` in closed
  mode as the very first step, is accepted, and `judge_adapter.score_episode`
  gives it exactly 1.0 at 0 inspection steps.

Default: SEEDS seeds per config (about 30-40 s in all). A larger sweep:
`COOKEDBOOKS_SWEEP=500 pytest -q tests/integration` runs that many seeds per
config and reports rejection counts by verifier problem code.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.verifier import verify_episode

SEEDS = 30
SWEEP = int(os.environ.get("COOKEDBOOKS_SWEEP", "0") or 0)
MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
RECIPES = [f"D{n:02d}" for n in range(1, 12)]
ALL_DISTRACTORS = {"split_payment": 2, "refund": 2, "recurring_amount": 2, "boundary": 3, "advance": 2,
                   "overpayment": 2}

GRID: dict[str, dict] = {
    "kzt_200_rows_max_errors": {
        "n_ledger_rows": 200, "n_documents": 50, "currencies": ["KZT", "USD"],
        "errors": {"duplicate_invoice": 4, "unallocated_payment": {"missing": 4, "partial": 4}},
        "distractors": {"split_payment": 1, "advance": 1, "refund": 1},
    },
    "kzt_small_each_error": {
        "n_ledger_rows": 200, "n_documents": 30, "currencies": ["KZT"],
        "errors": {"duplicate_invoice": 1, "unallocated_payment": {"missing": 1, "partial": 1}},
    },
    "multi_2000_max_errors_all_distractors": {
        "n_ledger_rows": 2000, "n_documents": 200, "currencies": ["KZT", "USD", "EUR"],
        "errors": {"duplicate_invoice": 4, "unallocated_payment": {"missing": 4, "partial": 4}},
        "distractors": ALL_DISTRACTORS,
    },
    "jpy_base_lima_half_even": {
        "base_currency": "JPY", "currencies": ["JPY", "USD"], "timezone": "America/Lima", "rounding": "half_even",
        "errors": {"duplicate_invoice": 2, "unallocated_payment": 2},
        "distractors": {"split_payment": 2, "refund": 1},
    },
    "kolkata_kwd_recurring_boundary": {
        "base_currency": "KZT", "currencies": ["KZT", "KWD", "EUR"], "timezone": "Asia/Kolkata",
        "errors": {"duplicate_invoice": 3, "unallocated_payment": {"partial": 2}},
        "distractors": {"recurring_amount": 3, "boundary": 3},
    },
    "utc_usd_missing_advances": {
        "base_currency": "USD", "currencies": ["USD", "KZT"], "timezone": "UTC",
        "errors": {"unallocated_payment": {"missing": 4}},
        "distractors": {"advance": 3, "refund": 2},
    },
    "tokyo_one_month_december": {
        "period_start": "2026-12-01", "period_months": 1, "timezone": "Asia/Tokyo",
        "currencies": ["KZT", "EUR", "JPY"],
        "errors": {"duplicate_invoice": 2, "unallocated_payment": {"missing": 1, "partial": 1}},
        "distractors": ALL_DISTRACTORS,
    },
    "kzt_2000_duplicates_recurring_even": {
        "n_ledger_rows": 2000, "n_documents": 120, "currencies": ["KZT"], "rounding": "half_even",
        "errors": {"duplicate_invoice": 4}, "distractors": {"recurring_amount": 4},
    },
    "bogota_clean_all_distractors": {
        "currencies": ["KZT", "USD"], "timezone": "America/Bogota", "distractors": ALL_DISTRACTORS,
    },
    "dubai_200_rows_multi": {
        "n_ledger_rows": 200, "n_documents": 40, "currencies": ["EUR", "USD", "KZT"], "base_currency": "EUR",
        "timezone": "Asia/Dubai", "errors": {"duplicate_invoice": 2, "unallocated_payment": {"partial": 2}},
        "distractors": {"split_payment": 1},
    },
    "overpayments_next_to_partial": {
        "currencies": ["KZT", "USD"], "errors": {"unallocated_payment": {"partial": 4, "missing": 2}},
        "distractors": {"overpayment": 4, "advance": 2, "refund": 1},
    },
    "overpayments_only_clean": {
        "base_currency": "JPY", "currencies": ["JPY", "KWD", "EUR"], "timezone": "Asia/Kolkata",
        "distractors": {"overpayment": 5},
    },
}


def random_config(seed: int) -> dict:
    """A config drawn per seed: 0-4 of each error and variant, each distractor on or off."""
    rng = random.Random(f"grid:{seed}")
    base = rng.choice(["KZT", "USD", "JPY", "KWD", "EUR"])
    others = [c for c in ("KZT", "USD", "JPY", "KWD", "EUR") if c != base]
    currencies = [base, *rng.sample(others, rng.randint(0, 2))]
    rows, docs = rng.choice([(200, 50), (600, 90), (2000, 200)])
    most = 1 if rows == 200 else 2  # 200 rows cannot hold 12 errors and ten look-alikes
    return {
        "n_ledger_rows": rows, "n_documents": docs, "base_currency": base, "currencies": currencies,
        "timezone": rng.choice(["Asia/Almaty", "UTC", "America/Lima", "Asia/Kolkata", "Asia/Tokyo"]),
        "rounding": rng.choice(["half_up", "half_even"]),
        "errors": {"duplicate_invoice": rng.randint(0, 4),
                   "unallocated_payment": {"missing": rng.randint(0, 4), "partial": rng.randint(0, 4)}},
        "distractors": {kind: rng.randint(0, 1) * rng.randint(1, most) for kind in ALL_DISTRACTORS},
    }


def requested_issues(config: dict) -> int:
    total = 0
    for value in config.get("errors", {}).values():
        total += value if isinstance(value, int) else sum(value.values())
    return total + 2 * sum(config.get("scenarios", {}).values())


def check_episode(folder: Path, seed: int, config: dict) -> list[str]:
    """Generate, verify and score one episode; returns verifier problem codes (empty when it agrees)."""
    shutil.rmtree(folder, ignore_errors=True)
    judge_adapter.generate_episode(seed, config, str(folder))
    report = verify_episode(folder)
    if not report.ok:
        return sorted(report.codes()) or ["<no code>"]
    truth = json.loads((folder / "truth.json").read_text())
    assert len(truth["issues"]) == requested_issues(config), (seed, config)

    findings = oracle_findings(folder)
    env = judge_adapter.make_env()
    observation, info = env.reset(seed, {"mode": "closed", "episode_dir": str(folder), "team_config": {}})
    assert info == {} and observation["remaining_steps"] == 100
    result, reward, terminated, truncated, step_info = env.step({"name": "submit", "args": {"findings": findings}})
    assert result["error"] is None and result["result"] == {"accepted": True}
    assert (reward, terminated, truncated, step_info) == (0.0, True, False, {})
    metrics = judge_adapter.score_episode(str(folder), findings, 0)
    assert metrics["return"] == 1.0, (seed, config, metrics)
    assert metrics["fp"] == 0 and metrics["fn"] == 0 and metrics["tp"] == len(truth["issues"])
    return []


def run_config(tmp_path: Path, name: str, config_for, seeds: range) -> None:
    rejected: Counter = Counter()
    examples: dict[str, tuple[int, dict]] = {}
    for seed in seeds:
        config = config_for(seed)
        codes = check_episode(tmp_path / name, seed, config)
        if codes:
            rejected["episodes"] += 1
            for code in codes:
                rejected[code] += 1
                examples.setdefault(code, (seed, config))
    if SWEEP:
        print(f"\nSWEEP {name}: {len(seeds)} episodes, {rejected['episodes']} rejected, "
              f"codes {dict(sorted((k, v) for k, v in rejected.items() if k != 'episodes'))}")
    if rejected:
        detail = ""
        seed, config = next(iter(examples.values()))
        folder = tmp_path / "repro"
        judge_adapter.generate_episode(seed, config, str(folder))
        detail = verify_episode(folder).render(20)
        pytest.fail(f"{name}: verifier rejected {rejected['episodes']} of {len(seeds)} episodes "
                    f"{dict(rejected)}; first: seed {seed} config {json.dumps(config)}\n{detail}")


def seed_range() -> range:
    return range(SWEEP or SEEDS)


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader_log.jsonl"))


def _recipe_test(recipe: str):
    def test(tmp_path):
        config = json.loads((MATRIX / f"{recipe}.json").read_text())
        run_config(tmp_path, recipe, lambda seed: config, seed_range())
    test.__name__ = f"test_{recipe}_recipe_agrees_with_verifier"
    test.__doc__ = f"Recipe {recipe}: every seed passes the verifier and the oracle scores 1.0."
    return test


for _recipe in RECIPES:
    globals()[f"test_{_recipe}_recipe_agrees_with_verifier"] = _recipe_test(_recipe)


@pytest.mark.parametrize("name", sorted(GRID))
def test_grid_config_agrees_with_verifier(tmp_path, name):
    run_config(tmp_path, name, lambda seed: GRID[name], seed_range())


def test_random_configs_agree_with_verifier(tmp_path):
    run_config(tmp_path, "random", random_config, seed_range())


def test_recipes_ask_for_what_the_jury_lists():
    """D06-D08 duplicates only, each next to recurring look-alikes; D09 missing, D10 partial, D11 both."""
    recipes = {r: json.loads((MATRIX / f"{r}.json").read_text()) for r in RECIPES}
    for r in ("D06", "D07", "D08"):
        errors = recipes[r]["errors"]
        assert set(errors) == {"duplicate_invoice"} and errors["duplicate_invoice"] >= 1
    assert len({json.dumps(recipes[r], sort_keys=True) for r in ("D06", "D07", "D08")}) == 3
    assert any(len(recipes[r]["currencies"]) == 1 for r in ("D06", "D07", "D08"))
    assert any(len(recipes[r]["currencies"]) > 1 for r in ("D06", "D07", "D08"))
    assert all(recipes[r]["distractors"].get("recurring_amount") for r in ("D06", "D07", "D08"))
    variants = {r: {v for v, n in recipes[r]["errors"]["unallocated_payment"].items() if n}
                for r in ("D09", "D10", "D11")}
    assert variants == {"D09": {"missing"}, "D10": {"partial"}, "D11": {"missing", "partial"}}
    for r in ("D09", "D10", "D11"):
        assert set(recipes[r]["errors"]) == {"unallocated_payment"}
        assert set(recipes[r]["distractors"]) & {"split_payment", "advance", "refund"}
    for r in ("D10", "D11"):  # partial errors sit next to legitimate overpayments
        assert recipes[r]["distractors"].get("overpayment")

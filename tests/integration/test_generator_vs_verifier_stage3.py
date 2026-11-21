"""fx_error, cutoff_error, S1 and S2 against the independent verifier, through the public entry points.

For every recipe D12-D20, a grid of configs mixing all four error types, both
scenarios and all look-alikes (currency setups with exponents 0/2/3 as base and
as foreign currency, every supported time zone, both rounding modes, small and
large sizes) and a per-seed random config, over SEEDS seeds each:

- `judge_adapter.generate_episode` writes the episode, and its
  `provenance.json` says attempt 0: the very first generator attempt was
  accepted (the adapter may retry rejected attempts; a retry here would hide a
  disagreement, so it counts as a failure);
- `cookedbooks.verifier.verify_episode(dir).ok` (the verifier was written
  from DESIGN.md alone and never imports the generator);
- truth.json holds exactly the requested number of issues;
- the oracle answer, submitted through `judge_adapter.make_env()` in closed
  mode as the very first step, is accepted, and `judge_adapter.score_episode`
  gives it exactly 1.0 at 0 inspection steps.

When the verifier rejects, the failure message names the codes, the first
reproducing (seed, config) and the verifier's report of attempt 0.

Default: SEEDS seeds per config. A larger sweep:
`COOKEDBOOKS_SWEEP=300 pytest -q -s tests/integration/test_generator_vs_verifier_stage3.py`
runs that many seeds per config and prints rejection counts by problem code.
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
from cookedbooks.world.generate import generate_episode as generate_attempt

SEEDS = 30
SWEEP = int(os.environ.get("COOKEDBOOKS_SWEEP", "0") or 0)
MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
RECIPES = [f"D{n:02d}" for n in range(12, 21)]
ZONES = ("UTC", "America/Bogota", "America/Lima", "Asia/Almaty", "Asia/Dubai", "Asia/Kolkata", "Asia/Riyadh",
         "Asia/Shanghai", "Asia/Tashkent", "Asia/Tokyo")
CODES = ("KZT", "USD", "EUR", "JPY", "KWD")
ALL_DISTRACTORS = {"split_payment": 1, "refund": 1, "recurring_amount": 1, "boundary": 3, "advance": 1,
                   "overpayment": 1}


def errors(dup: int = 0, missing: int = 0, partial: int = 0, wrong_date: int = 0, wrong_pair: int = 0,
           late: int = 0, early: int = 0) -> dict:
    return {"duplicate_invoice": dup, "unallocated_payment": {"missing": missing, "partial": partial},
            "fx_error": {"wrong_date": wrong_date, "wrong_pair": wrong_pair},
            "cutoff_error": {"late": late, "early": early}}


ALL_KINDS = errors(1, 1, 1, 1, 1, 1, 1)

GRID: dict[str, dict] = {
    "almaty_kzt_usd_eur_900_everything": {
        "n_ledger_rows": 900, "n_documents": 130, "currencies": ["KZT", "USD", "EUR"],
        "errors": errors(2, 1, 1, 2, 2, 2, 1), "scenarios": {"S1": 1, "S2": 1},
        "distractors": {**ALL_DISTRACTORS, "boundary": 4},
    },
    "lima_jpy_base_half_even": {
        "base_currency": "JPY", "currencies": ["JPY", "USD"], "timezone": "America/Lima", "rounding": "half_even",
        "n_ledger_rows": 600, "n_documents": 90, "errors": ALL_KINDS, "scenarios": {"S1": 1, "S2": 1},
        "distractors": ALL_DISTRACTORS,
    },
    "kolkata_kwd_base_eur_jpy": {
        "base_currency": "KWD", "currencies": ["KWD", "EUR", "JPY"], "timezone": "Asia/Kolkata",
        "n_ledger_rows": 800, "n_documents": 110, "errors": errors(1, 0, 1, 2, 2, 1, 1),
        "scenarios": {"S1": 2, "S2": 1}, "distractors": ALL_DISTRACTORS,
    },
    "utc_usd_base_kzt": {
        "base_currency": "USD", "currencies": ["USD", "KZT"], "timezone": "UTC",
        "errors": errors(0, 1, 0, 3, 1, 1, 1), "scenarios": {"S1": 1, "S2": 1},
        "distractors": {"boundary": 4, "overpayment": 1},
    },
    "tokyo_one_month_december": {
        "period_start": "2026-12-01", "period_months": 1, "timezone": "Asia/Tokyo",
        "currencies": ["KZT", "EUR", "JPY"], "errors": ALL_KINDS, "scenarios": {"S1": 1, "S2": 1},
        "distractors": ALL_DISTRACTORS,
    },
    "bogota_200_rows": {
        "n_ledger_rows": 200, "n_documents": 40, "currencies": ["KZT", "USD"], "timezone": "America/Bogota",
        "errors": errors(wrong_date=1, late=1, early=1), "scenarios": {"S1": 1}, "distractors": {"boundary": 2},
    },
    "dubai_eur_base_30_issues": {
        "base_currency": "EUR", "currencies": ["EUR", "USD", "KZT"], "timezone": "Asia/Dubai",
        "n_ledger_rows": 800, "n_documents": 140, "errors": errors(4, 2, 2, 4, 4, 3, 3),
        "scenarios": {"S1": 2, "S2": 2}, "distractors": {"boundary": 3, "overpayment": 1, "recurring_amount": 1},
    },
    "riyadh_kzt_jpy_kwd_half_even": {
        "currencies": ["KZT", "JPY", "KWD"], "timezone": "Asia/Riyadh", "rounding": "half_even",
        "errors": errors(1, 0, 1, 2, 2, 1, 1), "scenarios": {"S2": 2}, "distractors": ALL_DISTRACTORS,
    },
    "shanghai_jpy_base_kwd": {
        "base_currency": "JPY", "currencies": ["JPY", "KWD"], "timezone": "Asia/Shanghai",
        "n_ledger_rows": 600, "n_documents": 80, "errors": errors(wrong_date=2, wrong_pair=2),
        "scenarios": {"S1": 1, "S2": 1}, "distractors": {"boundary": 4, "recurring_amount": 1},
    },
    "tashkent_kzt_only_cutoffs": {
        "currencies": ["KZT"], "timezone": "Asia/Tashkent", "errors": errors(1, 1, 1, late=3, early=3),
        "distractors": {"boundary": 4, "split_payment": 1, "advance": 1, "refund": 1},
    },
}


def random_config(seed: int) -> dict:
    """A config drawn per seed: every error, variant and scenario 0-2 times, look-alikes on or off."""
    rng = random.Random(f"stage3:{seed}")
    base = rng.choice(CODES)
    currencies = [base, *rng.sample([c for c in CODES if c != base], rng.randint(0, 2))]
    rows, docs = rng.choice([(200, 50), (600, 90), (1200, 150), (2000, 200)])
    most = 1 if rows == 200 else 2
    foreign = 1 if len(currencies) > 1 else 0
    config = {
        "n_ledger_rows": rows, "n_documents": docs, "base_currency": base, "currencies": currencies,
        "timezone": rng.choice(ZONES), "rounding": rng.choice(["half_up", "half_even"]),
        "errors": errors(*(rng.randint(0, most) for _ in range(3)),
                         rng.randint(0, most) * foreign, rng.randint(0, most) * foreign,
                         rng.randint(0, most), rng.randint(0, most)),
        "scenarios": {"S1": rng.randint(0, 1) * foreign, "S2": rng.randint(0, 1) * foreign},
        "distractors": {kind: rng.randint(0, 1) * rng.randint(1, most) for kind in ALL_DISTRACTORS},
    }
    if rng.randrange(4) == 0:
        config.update(period_start=rng.choice(["2026-01-01", "2026-12-01", "2027-02-01"]),
                      period_months=rng.choice([1, 2, 3]))
    return config


def requested_issues(config: dict) -> int:
    total = 0
    for value in config.get("errors", {}).values():
        total += value if isinstance(value, int) else sum(value.values())
    return total + 2 * sum(config.get("scenarios", {}).values())


def check_episode(folder: Path, seed: int, config: dict) -> list[str]:
    """Generate, verify and score one episode; returns problem codes of attempt 0 (empty when it agrees)."""
    shutil.rmtree(folder, ignore_errors=True)
    judge_adapter.generate_episode(seed, config, str(folder))
    provenance = json.loads((folder / "provenance.json").read_text())
    report = verify_episode(folder)
    if provenance["attempt"] != 0 or not report.ok:
        shutil.rmtree(folder, ignore_errors=True)
        generate_attempt(seed, config, folder)
        return sorted(verify_episode(folder).codes()) or ["<attempt 0 rejected, no code>"]
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
        seed, config = next(iter(examples.values()))
        folder = tmp_path / "repro"
        generate_attempt(seed, config, folder)
        detail = verify_episode(folder).render(20)
        pytest.fail(f"{name}: verifier rejected {rejected['episodes']} of {len(seeds)} episodes "
                    f"{dict(rejected)}; first: seed {seed} config {json.dumps(config)}\n{detail}")


def seed_range() -> range:
    return range(SWEEP or SEEDS)


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader_log.jsonl"))


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipe_agrees_with_verifier(tmp_path, recipe):
    """Recipes D12-D20: every seed passes the verifier at attempt 0 and the oracle scores 1.0."""
    config = json.loads((MATRIX / f"{recipe}.json").read_text())
    run_config(tmp_path, recipe, lambda seed: config, seed_range())


@pytest.mark.parametrize("name", sorted(GRID))
def test_grid_config_agrees_with_verifier(tmp_path, name):
    run_config(tmp_path, name, lambda seed: GRID[name], seed_range())


def test_random_configs_agree_with_verifier(tmp_path):
    run_config(tmp_path, "random", random_config, seed_range())


def test_grid_covers_every_setup():
    """The grid and the recipes reach every zone, both roundings, exponents 0/2/3 as base and foreign."""
    configs = [*GRID.values(), *(json.loads((MATRIX / f"{r}.json").read_text()) for r in RECIPES)]
    assert {c.get("timezone", "Asia/Almaty") for c in configs} == set(ZONES)
    assert {c.get("rounding", "half_up") for c in configs} == {"half_up", "half_even"}
    exponent = {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3}
    assert {exponent[c.get("base_currency", "KZT")] for c in configs} == {0, 2, 3}
    assert {exponent[f] for c in configs for f in c["currencies"] if f != c.get("base_currency", "KZT")} == {0, 2, 3}
    mixed = [c for c in GRID.values() if all(v for v in (c["errors"]["duplicate_invoice"],
             sum(c["errors"]["unallocated_payment"].values()), sum(c["errors"]["fx_error"].values()),
             sum(c["errors"]["cutoff_error"].values()), c.get("scenarios", {}).get("S1"),
             c.get("scenarios", {}).get("S2")))]
    assert len(mixed) >= 4
    assert any(set(c["distractors"]) == set(ALL_DISTRACTORS) for c in mixed)


def test_recipes_ask_for_what_the_jury_lists():
    """D12 wrong_date, D13 wrong_pair, D14 late, D15 early (with boundary look-alikes); D16-D18 S1; D19-D20 S2."""
    recipes = {r: json.loads((MATRIX / f"{r}.json").read_text()) for r in RECIPES}

    def asked(config: dict) -> dict[str, int]:
        found = {}
        for kind, value in config.get("errors", {}).items():
            for variant, count in (value.items() if isinstance(value, dict) else [(None, value)]):
                if count:
                    found[f"{kind}.{variant}" if variant else kind] = count
        found.update({name: n for name, n in config.get("scenarios", {}).items() if n})
        return found

    expected = {"D12": "fx_error.wrong_date", "D13": "fx_error.wrong_pair", "D14": "cutoff_error.late",
                "D15": "cutoff_error.early", "D16": "S1", "D17": "S1", "D18": "S1", "D19": "S2", "D20": "S2"}
    for recipe, only in expected.items():
        got = asked(recipes[recipe])
        assert list(got) == [only], (recipe, got)
        if only in ("S1", "S2"):
            assert got[only] == 1
    for recipe in ("D12", "D13", "D14", "D15"):
        assert recipes[recipe]["distractors"].get("boundary", 0) >= 3
    assert recipes["D14"]["timezone"] != recipes["D15"]["timezone"]
    for group in (("D16", "D17", "D18"), ("D19", "D20")):
        setups = {json.dumps({k: v for k, v in recipes[r].items() if k not in ("scenarios", "distractors")},
                             sort_keys=True) for r in group}
        assert len(setups) == len(group)
        assert len({recipes[r]["timezone"] for r in group}) == len(group)

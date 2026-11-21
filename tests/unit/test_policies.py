"""The four baseline policies (PLAN.md section 7) on every jury recipe.

Policies are driven in this process here, straight through `judge_adapter`,
because these tests are about policy behaviour; the process boundary has its
own tests in tests/unit/test_runner.py.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.grader.oracle import oracle_findings

ROOT = Path(__file__).resolve().parents[2]
POLICIES_DIR = ROOT / "cookedbooks" / "policies"
BASELINES = ("random_explorer", "simple_recon", "flag_all", "oracle")
RECIPES = tuple(f"D{k:02d}" for k in range(1, 21))
CLEAN_RECIPES = RECIPES[:5]
SEED_BASE = 910000  # not a seed of the jury-recipe tests


@pytest.fixture(scope="module", autouse=True)
def grader_log(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path_factory.mktemp("log") / "grader.jsonl"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def recipe_dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("recipes")
    out = {}
    for k, name in enumerate(RECIPES, start=1):
        config = json.loads((ROOT / "configs" / "matrix" / f"{name}.json").read_text())
        judge_adapter.generate_episode(SEED_BASE + k, config, str(root / name))
        out[name] = root / name
    return out


def drive(policy_name: str, episode_dir: Path, seed: int = 0) -> dict:
    """Run a policy in this process; return actions, observations, metrics and flags."""
    options = {"answer": oracle_findings(episode_dir)} if policy_name == "oracle" else {}
    policy = importlib.import_module("cookedbooks.policies." + policy_name).make_policy(seed, **options)
    env = judge_adapter.make_env()
    observation, _ = env.reset(None, {"mode": "closed", "episode_dir": str(episode_dir)})
    actions, observations = [], [observation]
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(observation)
        actions.append(json.loads(json.dumps(action)))
        observation, reward, terminated, truncated, _ = env.step(action)
        assert reward == 0.0
        observations.append(observation)
    metrics = judge_adapter.score_episode(str(episode_dir), env.final_findings, env.inspection_steps)
    return {"actions": actions, "observations": observations, "metrics": metrics, "terminated": terminated,
            "truncated": truncated, "steps": env.inspection_steps, "findings": env.final_findings}


@pytest.fixture(scope="module")
def runs(recipe_dirs) -> dict[tuple[str, str], dict]:
    return {(policy, name): drive(policy, folder) for policy in BASELINES for name, folder in recipe_dirs.items()}


def _truth(folder: Path) -> list[dict]:
    return json.loads((folder / "truth.json").read_text())["issues"]


# --- the interface and the import rule ------------------------------------------------


@pytest.mark.parametrize("path", sorted(POLICIES_DIR.glob("*.py")), ids=lambda p: p.stem)
def test_policy_modules_import_only_stdlib_and_policies(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not used"
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top == "cookedbooks":
                assert name == "cookedbooks.policies" or name.startswith("cookedbooks.policies."), name
            else:
                assert top in sys.stdlib_module_names or top == "__future__", name


@pytest.mark.parametrize("policy", BASELINES)
def test_make_policy_returns_an_object_with_act(policy):
    module = importlib.import_module("cookedbooks.policies." + policy)
    assert callable(getattr(module.make_policy(0), "act", None))


# --- every policy, every recipe --------------------------------------------------------


@pytest.mark.parametrize("policy", BASELINES)
def test_policy_is_deterministic_for_a_seed(policy, recipe_dirs):
    for name in ("D01", "D11", "D17"):
        first = drive(policy, recipe_dirs[name], seed=5)
        second = drive(policy, recipe_dirs[name], seed=5)
        assert first["actions"] == second["actions"]
        assert first["metrics"] == second["metrics"]


def test_random_explorer_depends_on_its_seed(recipe_dirs):
    runs = [drive("random_explorer", recipe_dirs["D06"], seed=s)["actions"] for s in range(4)]
    assert len({json.dumps(r) for r in runs}) > 1


@pytest.mark.parametrize("policy", BASELINES)
def test_policy_submits_within_the_budget_on_every_recipe(policy, runs):
    for name in RECIPES:
        run = runs[(policy, name)]
        assert run["terminated"] and not run["truncated"], name
        assert len(run["actions"]) <= 100 and run["actions"][-1]["name"] == "submit", name
        assert run["observations"][-1]["error"] is None, name  # a structurally valid answer
        assert run["metrics"]["return"] is not None and run["metrics"]["f1"] is not None, name


# --- what each baseline is meant to show ---------------------------------------------------


def test_oracle_scores_exactly_one_at_zero_steps(runs):
    for name in RECIPES:
        run = runs[("oracle", name)]
        assert run["steps"] == 0 and len(run["actions"]) == 1
        assert run["metrics"]["return"] == 1.0 and run["metrics"]["f1"] == 1.0, name


def test_flag_all_sends_thirty_valid_findings_over_all_four_types(runs):
    findings = runs[("flag_all", "D06")]["findings"]["findings"]
    assert len(findings) == 30
    assert {f["type"] for f in findings} == {"duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error"}


def test_flag_all_never_beats_silence_on_clean_books(runs):
    silence = 1.0  # an empty answer at step 0 on clean books: F1 = 1, no steps
    for name in CLEAN_RECIPES:
        assert runs[("flag_all", name)]["metrics"]["return"] < silence
        assert runs[("flag_all", name)]["metrics"]["return"] == -1.0


def test_flag_all_scores_near_minus_one_on_books_with_errors(runs, recipe_dirs):
    for name in RECIPES:
        if _truth(recipe_dirs[name]):
            assert runs[("flag_all", name)]["metrics"]["return"] <= -0.9, name


def test_simple_recon_is_between_flag_all_and_oracle_on_average(runs):
    def mean(policy: str) -> float:
        return sum(runs[(policy, name)]["metrics"]["return"] for name in RECIPES) / len(RECIPES)

    assert mean("flag_all") < mean("simple_recon") < mean("oracle")
    assert any(runs[("simple_recon", name)]["metrics"]["tp"] > 0 for name in RECIPES)


def test_simple_recon_is_fooled_by_a_look_alike(runs, recipe_dirs):
    """On clean books with look-alikes it reports something, and the report names a look-alike."""
    fooled = []
    for name in CLEAN_RECIPES:
        run = runs[("simple_recon", name)]
        if run["metrics"]["fp"] == 0:
            continue
        provenance = json.loads((recipe_dirs[name] / "provenance.json").read_text())
        look_alike_ids = {i for d in provenance["distractors"] for i in d["ids"]}
        if any(set(f["affected_ids"]) & look_alike_ids for f in run["findings"]["findings"]):
            fooled.append(name)
    assert fooled, "no look-alike fooled simple_recon"


def test_simple_recon_never_reports_fx_errors(runs):
    for name in RECIPES:
        assert all(f["type"] != "fx_error" for f in runs[("simple_recon", name)]["findings"]["findings"])


def test_random_explorer_exercises_error_paths_without_crashing(runs):
    codes = set()
    for name in RECIPES:
        for observation in runs[("random_explorer", name)]["observations"][1:-1]:
            if observation["error"] is not None:
                codes.add(observation["error"]["code"])
    assert {"unknown_action", "unknown_table", "unknown_id", "bad_arguments", "calc_error"} <= codes
    for name in RECIPES:
        findings = runs[("random_explorer", name)]["findings"]["findings"]
        assert 1 <= len(findings) <= 3

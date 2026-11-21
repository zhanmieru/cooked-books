"""Reward attacks on real generated episodes of every recipe (cookedbooks.verify.attacks).

The jury's 20 matcher unit tests on tiny fixtures live in
tests/jury/test_block3_reward.py; here every attack of the catalogue runs
through the real environment and grader on one episode of each recipe D01-D20
(more with COOKEDBOOKS_ATTACK_SEEDS), and the invariants are asserted one by
one so a failure names the attack and the episode.
"""

from __future__ import annotations

import functools
import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.verify import attacks
from cookedbooks.verify.attacks import (
    ATTACKS, STRUCTURAL_ATTACKS, EpisodeView, exact_return, expected_counts, is_structurally_valid, run_attacks,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES = [f"D{k:02d}" for k in range(1, 21)]
CLEAN = {"D01", "D02", "D03", "D04", "D05"}
SEEDS = int(os.environ.get("COOKEDBOOKS_ATTACK_SEEDS", "1"))
# Inside the grammar and the 200-character limit, so they are answered exactly; everything else is calc_error.
CALC_VALID = {"(" * 99 + "1" + ")" * 99: ("1", None), "-" * 199 + "1": ("-1", None), "+-" * 99 + "1": ("-1", None)}
PADDING = ("right_answer_x30", "right_answer_plus_29_padding", "right_answer_plus_lookalikes")


@pytest.fixture(scope="module")
def episodes(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("attack-episodes")
    out = {}
    for name in RECIPES:
        config = json.loads((REPO / "configs" / "matrix" / f"{name}.json").read_text(encoding="utf-8"))
        for seed in range(SEEDS):
            folder = root / f"{name}-{seed}"
            judge_adapter.generate_episode(seed, config, str(folder))
            out[f"{name}/{seed}"] = str(folder)
    return out


@pytest.fixture(scope="module")
def results(episodes, tmp_path_factory) -> dict[str, list[attacks.AttackResult]]:
    log = tmp_path_factory.mktemp("grader-log") / "log.jsonl"
    old = os.environ.get("COOKEDBOOKS_GRADER_LOG")
    os.environ["COOKEDBOOKS_GRADER_LOG"] = str(log)
    try:
        return {key: run_attacks(folder) for key, folder in episodes.items()}
    finally:
        if old is None:
            os.environ.pop("COOKEDBOOKS_GRADER_LOG", None)
        else:
            os.environ["COOKEDBOOKS_GRADER_LOG"] = old


def by_name(results, name: str) -> list[tuple[str, attacks.AttackResult]]:
    return [(key, r) for key, rs in results.items() for r in rs if r.attack == name]


def one(results, key: str, name: str) -> attacks.AttackResult:
    return next(r for r in results[key] if r.attack == name)


@functools.lru_cache(maxsize=None)
def truth_count(folder: str) -> int:
    return len(EpisodeView.load(folder).truth)


def is_clean(key: str) -> bool:
    return key.split("/")[0] in CLEAN


# --- every attack keeps every invariant ------------------------------------------------------


@pytest.mark.parametrize("name", list(ATTACKS))
def test_attack_breaks_no_invariant(results, name):
    """Independent re-count, not above the oracle, -1 for invalid, nothing after the end, no leak."""
    broken = [(key, r.problems) for key, r in by_name(results, name) if r.problems]
    assert not broken, broken[:3]


def test_catalogue_covers_the_requested_attacks():
    assert len(ATTACKS) >= 60
    for required in ("silence", "random_accusation", "mass_flag_first_records", "right_answer_x30",
                     "right_answer_plus_29_padding", "amount_plus_one", "amount_rounding_variant",
                     "amount_in_other_currency", "amount_sign_flipped", "amount_with_other_currency_code",
                     "currency_lower_case", "currency_padded", "evidence_from_other_issue", "evidence_missing_one",
                     "evidence_plus_nonexistent", "evidence_padded_to_limit", "business_keys_in_evidence",
                     "affected_superset", "affected_subset", "two_issues_merged", "lookalikes_accused",
                     "everything_except_solution", "injection_essay_1000", "struct_findings_bare_list",
                     "episode_id_of_other_episode", "submit_twice_better_second", "act_after_end",
                     "burn_budget_submit_at_100", "burn_budget_submit_at_101", "time_limit_exceeded",
                     "path_probe_every_tool", "calculator_abuse"):
        assert required in ATTACKS


# --- the jury's invariants, one by one --------------------------------------------------------


def test_complete_answer_beats_silence_on_books_with_errors(results):
    for key in results:
        if is_clean(key):
            continue
        oracle, silence = one(results, key, "oracle (reference)"), one(results, key, "silence")
        assert oracle.ret == float(exact_return(truth_count(oracle.episode_dir), 0, 0, 0)) == 1.0
        assert silence.ret == -1.0, key
        assert oracle.ret > silence.ret


def test_silence_beats_any_accusation_on_clean_books(results):
    checked = 0
    for key, rs in results.items():
        if not is_clean(key):
            continue
        silence = one(results, key, "silence")
        assert silence.ret == 1.0
        for r in rs:
            final_has_findings = r.expected is not None and (r.expected[1] > 0)
            if final_has_findings:
                assert r.ret < float(exact_return(0, 0, 0, r.inspection_steps)), (key, r.attack)
                checked += 1
    assert checked >= 5 * 8 * SEEDS


def test_lookalikes_are_accused_on_every_recipe_that_has_them(results):
    for key in results:
        r = one(results, key, "lookalikes_accused")
        if r.expected and r.expected[1]:
            assert r.expected[0] == 0, (key, "a look-alike accusation matched a truth issue")
            assert r.ret < one(results, key, "silence").ret or not is_clean(key)


def test_repeats_and_padding_never_raise_the_return(results):
    for key in results:
        oracle = one(results, key, "oracle (reference)")
        for name in PADDING:
            r = one(results, key, name)
            assert r.ret <= oracle.ret, (key, name)
            k = truth_count(r.episode_dir)
            if name == "right_answer_x30" and k:
                assert (r.metrics["tp"], r.metrics["fp"], r.metrics["fn"]) == (k, 30 - k, 0)
                assert r.ret == float(exact_return(k, 30 - k, 0, 0)) and exact_return(k, 30 - k, 0, 0) < 1
            if name == "right_answer_plus_29_padding" and k:
                assert r.metrics["tp"] == k and r.metrics["fp"] == 30 - k
                assert r.ret == float(exact_return(k, 30 - k, 0, 0))


def test_no_attack_scores_above_the_oracle_at_the_same_steps(results):
    for key, rs in results.items():
        for r in rs:
            assert r.ret <= r.oracle_return, (key, r.attack)
            assert r.oracle_return == float(exact_return(truth_count(r.episode_dir), 0, 0,
                                                         r.inspection_steps))


@pytest.mark.parametrize("name", list(STRUCTURAL_ATTACKS) + ["injection_essay_1001", "invalid_then_valid_submit"])
def test_structurally_invalid_attacks_end_at_exactly_minus_one(results, name):
    for key, r in by_name(results, name):
        assert r.metrics == {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                             "f1": None, "return": -1}, key
        assert r.expected is None


def test_equal_answers_are_separated_by_exactly_0002_per_step(episodes, tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    for key, folder in episodes.items():
        view = EpisodeView.load(folder)
        k = len(view.truth)
        padded = attacks.a_evidence_padded_to_limit(view).final  # a different but equal answer
        previous = None
        for steps in (0, 1, 2, 17, 49, 50, 99, 100):
            ret = judge_adapter.score_episode(folder, view.oracle, steps)["return"]
            assert ret == judge_adapter.score_episode(folder, padded, steps)["return"]
            expected = exact_return(k, 0, 0, steps)
            assert ret == float(expected) and expected == 1 - Fraction(steps, 500)
            if previous is not None:
                assert previous[1] - expected == Fraction(steps - previous[0], 500)
            previous = (steps, expected)


def test_nothing_after_the_end_changes_the_recorded_result(results):
    for key in results:
        twice = one(results, key, "submit_twice_better_second")
        assert twice.ret == one(results, key, "silence").ret
        after = one(results, key, "act_after_end")
        assert after.inspection_steps == 0 and after.metrics["tp"] == 0
        assert [o["error"]["code"] for o in after.observations[2:]] == ["episode_over"] * 4
        invalid = one(results, key, "invalid_then_valid_submit")
        assert invalid.ret == -1.0


def test_submit_observation_is_the_same_for_right_and_wrong_answers(results):
    """The closed-mode observations of a perfect, a wrong and an empty answer are identical."""
    for key in results:
        seen = {json.dumps(one(results, key, name).observations, sort_keys=True)
                for name in ("oracle (reference)", "silence", "amount_plus_one", "everything_except_solution",
                             "injection_claims_correct", "lookalikes_accused")}
        assert len(seen) == 1, key


def test_budget_boundary_submit_100_vs_101(results):
    for key in results:
        k = truth_count(one(results, key, "silence").episode_dir)
        at_100 = one(results, key, "burn_budget_submit_at_100")
        assert at_100.inspection_steps == 99 and at_100.ret == float(exact_return(k, 0, 0, 99))
        at_101 = one(results, key, "burn_budget_submit_at_101")
        assert at_101.inspection_steps == 100 and at_101.ret == float(exact_return(0, 0, k, 100))
        assert at_101.observations[100]["error"] is None and at_101.observations[101]["error"]["code"] == "episode_over"
        failing = one(results, key, "burn_budget_with_failing_actions")
        assert failing.inspection_steps == 99 and failing.ret == at_100.ret


def test_time_limit_gives_the_empty_answer(results):
    for key in results:
        r = one(results, key, "time_limit_exceeded")
        k = truth_count(r.episode_dir)
        assert r.observations[-1]["error"]["code"] == "time_limit"
        assert r.inspection_steps == 1 and r.ret == float(exact_return(0, 0, k, 1))


def test_path_probes_and_calculator_abuse_only_get_fixed_errors(results):
    for key in results:
        probe = one(results, key, "path_probe_every_tool")
        codes = {o["error"]["code"] for o in probe.observations[1:-1]}
        assert codes <= {"unknown_table", "unknown_id", "bad_arguments", "calc_error"}, key
        assert all(o["result"] is None for o in probe.observations[1:-1])
        calc = one(results, key, "calculator_abuse")
        outcomes = {e: (o["result"], (o["error"] or {}).get("code"))
                    for e, o in zip(attacks.CALC_ATTACKS, calc.observations[1:-1])}
        assert len(outcomes) == len(attacks.CALC_ATTACKS)
        for expression, outcome in outcomes.items():
            assert outcome == CALC_VALID.get(expression, (None, "calc_error")), (key, expression, outcome)


# --- the checker itself is not vacuous --------------------------------------------------------


def test_independent_counter_follows_contract_examples():
    issue = {"type": "duplicate_invoice", "affected_ids": ["R1", "R2"], "expected_amount_minor": 10000,
             "currency": "KZT", "evidence_sets": [["D1", "D2"]]}
    issue_b = {**issue, "affected_ids": ["R3"]}
    existing = frozenset({"R1", "R2", "R3", "D1", "D2", "D3"})
    good = {"type": "duplicate_invoice", "affected_ids": ["R2", "R1", "R1"], "claimed_amount_minor": 10000,
            "currency": "KZT", "evidence_ids": ["D2", "D1", "D3"], "explanation": ""}
    assert expected_counts({"findings": [good]}, [issue], existing, {}) == (1, 0, 0)      # M01, M03
    assert expected_counts({"findings": [{**good, "evidence_ids": ["D1", "D2", "U"]}]}, [issue], existing, {}) \
        == (0, 1, 1)                                                                       # M04
    assert expected_counts({"findings": [good, good]}, [issue], existing, {}) == (1, 1, 0)  # M12
    both = {**good, "affected_ids": ["R3"]}
    assert expected_counts({"findings": [good, both]}, [issue, issue_b], existing, {}) == (2, 0, 0)
    assert expected_counts({"findings": [{**good, "claimed_amount_minor": 10001}]}, [issue], existing,
                           {"duplicate_invoice": 1}) == (1, 0, 0)                          # M08
    assert exact_return(1, 1, 0, 10) == Fraction(47, 150)
    assert exact_return(1, 19, 0, 10) == Fraction(-871, 1050)
    assert not is_structurally_valid({"episode_id": "e", "findings": [{**good, "claimed_amount_minor": True}]})
    assert not is_structurally_valid({"episode_id": "e", "findings": [{**good, "evidence_ids": ["D1"] * 11}]})
    assert is_structurally_valid({"episode_id": "e", "findings": [{**good, "type": "nonsense"}]})


class LeakyEnv:
    """Wraps the real environment and puts a truth issue_id into every error message."""

    def __init__(self, inner, marker: str) -> None:
        self.inner, self.marker = inner, marker

    def reset(self, *args, **kwargs):
        return self.inner.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(action)
        if obs["error"] is not None:
            obs = {**obs, "error": {**obs["error"], "message": f"near {self.marker}"}}
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_checker_catches_a_leaky_environment_and_a_generous_grader(episodes, monkeypatch, tmp_path):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    folder = episodes["D16/0"]
    marker = EpisodeView.load(folder).truth[0]["issue_id"]
    real_make_env, real_score = judge_adapter.make_env, judge_adapter.score_episode
    monkeypatch.setattr(judge_adapter, "make_env", lambda: LeakyEnv(real_make_env(), marker))
    (probe,) = run_attacks(folder, ["path_probe_every_tool"])
    assert any("fixed code" in p for p in probe.problems) and any(marker in p for p in probe.problems)

    monkeypatch.setattr(judge_adapter, "make_env", real_make_env)

    def generous(episode_dir, findings, steps):
        metrics = real_score(episode_dir, findings, steps)
        return {**metrics, "return": 1.0} if findings and findings.get("findings") else metrics

    monkeypatch.setattr(judge_adapter, "score_episode", generous)
    (flood,) = run_attacks(folder, ["everything_except_solution"])
    assert any("exact formula" in p for p in flood.problems)


def test_cli_prints_the_table(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    assert attacks.main(["--seeds", "1", "--recipes", "D06"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["attack", "episodes", "mean", "max", "oracle", "verdict"]
    assert "everything_except_solution" in out and "0 with a broken invariant" in out


NAIVE = ("naive_same_amount_duplicates", "naive_any_advance_is_unallocated", "naive_fx_written_date_any_diff",
         "naive_cutoff_written_dates", "naive_all_four_heuristics")


def test_honest_solver_matches_the_oracle_and_beats_every_public_heuristic(results):
    """Tricks that read the same public data but skip a rule lose to the rules (same reading cost or more)."""
    totals = {name: Fraction(0) for name in NAIVE}
    solver_total = Fraction(0)
    for key in results:
        solver = one(results, key, "evidence_solver (honest)")
        assert solver.ret == solver.oracle_return and solver.metrics["fp"] == solver.metrics["fn"] == 0, key
        for name in NAIVE:
            naive = one(results, key, name)
            assert naive.ret <= solver.ret, (key, name)
            if not is_clean(key):
                totals[name] += Fraction(naive.ret)
        if not is_clean(key):
            solver_total += Fraction(solver.ret)
    assert all(total < solver_total for total in totals.values()), totals

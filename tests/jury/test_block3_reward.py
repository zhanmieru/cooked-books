"""Jury block 3: matcher and reward tests 1-20 (R01-R20) and examples M01-M12.

docs/jury-checks.md, sections 4 and 7. Every test goes through
`judge_adapter.score_episode` on a folder written by the test factory, the way
the jury feeds us folders it built by hand.

Fixture for R01-R20: one issue A of type T on record R1, 10000 KZT, evidence
{D1, D2}, tau 0, inspection_steps 10. For M01-M12 A's affected_ids are
{R1, R2}. D3 exists, U does not.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Callable

import pytest

import judge_adapter
from cookedbooks.findings import ERROR_TYPES
from tests.helpers.factory import EpisodeFixture, build_mini

T = "duplicate_invoice"
STEPS = 10
METRIC_KEYS = {"tp", "fp", "fn", "precision", "recall", "f1", "return"}


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch) -> Path:
    """Send the grader log to the test's temp folder."""
    path = tmp_path / "logs" / "grader_log.jsonl"
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(path))
    return path


def make_episode(tmp_path: Path, make_issues: Callable[[EpisodeFixture], list[dict]], *,
                 policy: dict | None = None) -> EpisodeFixture:
    """The mini fixture's public data with a hand-written truth.json (and policy.json).

    `make_issues` gets the fixture (for its IDs) and returns the issues to write.
    """
    episode = build_mini(tmp_path / "ep")
    issues = make_issues(episode)
    truth = {"schema_version": "1.0", "episode_id": episode.episode_id, "issues": issues}
    (episode.path / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    if policy is not None:
        (episode.path / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return episode


def issue(issue_id: str, affected: list[str], evidence_sets: list[list[str]], *,
          type_: str = T, amount: int = 10000, currency: str = "KZT") -> dict:
    return {"issue_id": issue_id, "type": type_, "affected_ids": affected,
            "expected_amount_minor": amount, "currency": currency, "evidence_sets": evidence_sets}


def finding(affected: list[str], evidence: list[str], *, type_: str = T, amount: object = 10000,
            currency: str = "KZT", explanation: str = "test") -> dict:
    return {"type": type_, "affected_ids": affected, "claimed_amount_minor": amount,
            "currency": currency, "evidence_ids": evidence, "explanation": explanation}


def submit(episode: EpisodeFixture, findings: list[dict], steps: int = STEPS) -> dict:
    result = judge_adapter.score_episode(
        str(episode.path), {"episode_id": episode.episode_id, "findings": findings}, steps)
    assert set(result) == METRIC_KEYS
    return result


def assert_scores(result: dict, tp: int, fp: int, fn: int, precision: Fraction,
                  recall: Fraction, f1: Fraction, ret: Fraction) -> None:
    assert (result["tp"], result["fp"], result["fn"]) == (tp, fp, fn)
    for key in ("tp", "fp", "fn"):
        assert type(result[key]) is int
    for key, expected in (("precision", precision), ("recall", recall), ("f1", f1), ("return", ret)):
        assert isinstance(result[key], float)
        assert abs(result[key] - float(expected)) <= 1e-9, key


def assert_structural(result: dict, log: Path) -> dict:
    assert result == {"tp": None, "fp": None, "fn": None, "precision": None,
                      "recall": None, "f1": None, "return": -1.0}
    entry = last_log_entry(log)
    assert entry["structural_error"]
    return entry


def last_log_entry(log: Path) -> dict:
    return json.loads(log.read_text(encoding="utf-8").splitlines()[-1])


def last_log_entry_reason(tmp_path: Path) -> str:
    entry = last_log_entry(tmp_path / "logs" / "grader_log.jsonl")
    return entry["unmatched"][0]["reason"]


def one_issue(tmp_path: Path) -> tuple[EpisodeFixture, dict]:
    """Episode with issue A on R1 and the correct finding F."""
    episode = make_episode(tmp_path, lambda e: [issue("A", [e.r1], [[e.d1, e.d2]])])
    return episode, finding([episode.r1], [episode.d1, episode.d2])


ZERO = Fraction(0)
ONE = Fraction(1)
MINUS_ONE = Fraction(-1)


# --- R01-R20 ----------------------------------------------------------------


def test_R01_one_correct_finding(tmp_path):
    episode, f = one_issue(tmp_path)
    assert_scores(submit(episode, [f]), 1, 0, 0, ONE, ONE, ONE, Fraction(98, 100))


def test_R02_empty_answer(tmp_path):
    episode, _ = one_issue(tmp_path)
    assert_scores(submit(episode, []), 0, 0, 1, ZERO, ZERO, ZERO, MINUS_ONE)


def test_R03_empty_truth_empty_answer(tmp_path):
    episode = make_episode(tmp_path, lambda e: [])
    assert_scores(submit(episode, []), 0, 0, 0, ZERO, ZERO, ONE, Fraction(98, 100))


def test_R04_empty_truth_one_false_finding(tmp_path):
    episode = make_episode(tmp_path, lambda e: [])
    f = finding([episode.r1], [episode.d1, episode.d2])
    assert_scores(submit(episode, [f]), 0, 1, 0, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "no_issues"


def test_R05_two_identical_correct_findings(tmp_path):
    episode, f = one_issue(tmp_path)
    assert_scores(submit(episode, [f, dict(f)]), 1, 1, 0, Fraction(1, 2), ONE,
                  Fraction(2, 3), Fraction(47, 150))


def test_R06_wrong_type(tmp_path):
    episode, f = one_issue(tmp_path)
    f["type"] = "cutoff_error"
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "wrong_type"


def test_R07_other_existing_record_in_affected_ids(tmp_path):
    episode, f = one_issue(tmp_path)
    f["affected_ids"] = [episode.r2]
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "wrong_ids"


def test_R08_amount_10001(tmp_path):
    episode, f = one_issue(tmp_path)
    f["claimed_amount_minor"] = 10001
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "amount_off"


def test_R09_other_currency(tmp_path):
    episode, f = one_issue(tmp_path)
    f["currency"] = "USD"
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "wrong_currency"


def test_R10_evidence_d1_only(tmp_path):
    episode, f = one_issue(tmp_path)
    f["evidence_ids"] = [episode.d1]
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "evidence_insufficient"


def test_R11_unknown_id_instead_of_d2(tmp_path):
    episode, f = one_issue(tmp_path)
    f["evidence_ids"] = [episode.d1, episode.u]
    assert_scores(submit(episode, [f]), 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)
    assert last_log_entry_reason(tmp_path) == "evidence_unknown"


def test_R12_five_records_times_four_types(tmp_path):
    episode, _ = one_issue(tmp_path)
    assert len(episode.row_ids) == 5
    findings = [finding([row_id], [episode.d1, episode.d2], type_=type_)
                for row_id in episode.row_ids for type_ in ERROR_TYPES]
    assert len(findings) == 20
    assert_scores(submit(episode, findings), 1, 19, 0, Fraction(1, 20), ONE,
                  Fraction(2, 21), Fraction(-871, 1050))


def two_issues(tmp_path: Path) -> tuple[EpisodeFixture, dict]:
    episode = make_episode(tmp_path, lambda e: [
        issue("A", [e.r1], [[e.d1, e.d2]]),
        issue("B", [e.r2], [[e.d3]], type_="cutoff_error", amount=5000),
    ])
    return episode, finding([episode.r1], [episode.d1, episode.d2])


def test_R13_two_issues_correct_finding_for_a_only(tmp_path):
    episode, f = two_issues(tmp_path)
    assert_scores(submit(episode, [f]), 1, 0, 1, ONE, Fraction(1, 2),
                  Fraction(2, 3), Fraction(47, 150))


def test_R14_two_issues_correct_for_a_and_one_false(tmp_path):
    episode, f = two_issues(tmp_path)
    false = finding([episode.r3], [episode.d3], type_="fx_error", amount=1)
    assert_scores(submit(episode, [f, false]), 1, 1, 1, Fraction(1, 2), Fraction(1, 2),
                  Fraction(1, 2), Fraction(-2, 100))


def _correct_with_steps(tmp_path: Path, steps: int, expected: Fraction) -> None:
    episode, f = one_issue(tmp_path)
    assert_scores(submit(episode, [f], steps), 1, 0, 0, ONE, ONE, ONE, expected)


def test_R15_zero_inspection_steps(tmp_path):
    _correct_with_steps(tmp_path, 0, Fraction(1))


def test_R16_fifty_inspection_steps(tmp_path):
    _correct_with_steps(tmp_path, 50, Fraction(90, 100))


def test_R17_hundred_inspection_steps(tmp_path):
    _correct_with_steps(tmp_path, 100, Fraction(80, 100))


def test_R18_thirty_one_findings(tmp_path, grader_log):
    episode, f = one_issue(tmp_path)
    entry = assert_structural(submit(episode, [dict(f) for _ in range(31)]), grader_log)
    assert entry["structural_error"] == "too_many_findings"


def test_R19_eleven_distinct_evidence_ids(tmp_path, grader_log):
    episode, f = one_issue(tmp_path)
    distinct = sorted(set(episode.row_ids) | set(episode.doc_ids) | set(episode.invoice_ids)
                      | set(episode.fx_ids) | {episode.u, "extra"})
    f["evidence_ids"] = distinct[:11]
    assert len(set(f["evidence_ids"])) == 11
    entry = assert_structural(submit(episode, [f]), grader_log)
    assert entry["structural_error"] == "too_many_evidence_ids"


def test_R20_explanation_of_1001_ascii_chars(tmp_path, grader_log):
    episode, f = one_issue(tmp_path)
    f["explanation"] = "a" * 1001
    entry = assert_structural(submit(episode, [f]), grader_log)
    assert entry["structural_error"] == "explanation_too_long"


# --- M01-M12 ----------------------------------------------------------------


def example(tmp_path: Path, *, policy: dict | None = None) -> tuple[EpisodeFixture, dict]:
    """Issue A on {R1, R2} with evidence {D1, D2}; the correct finding."""
    episode = make_episode(tmp_path, lambda e: [issue("A", [e.r1, e.r2], [[e.d1, e.d2]])],
                           policy=policy)
    return episode, finding([episode.r1, episode.r2], [episode.d1, episode.d2])


def assert_tp(result: dict) -> None:
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    assert abs(result["return"] - 0.98) <= 1e-9


def assert_miss(result: dict) -> None:
    assert_scores(result, 0, 1, 1, ZERO, ZERO, ZERO, MINUS_ONE)


def test_M01_affected_ids_order_and_repeats(tmp_path):
    episode, f = example(tmp_path)
    f["affected_ids"] = [episode.r2, episode.r1, episode.r1]
    assert_tp(submit(episode, [f]))


def test_M02_extra_existing_affected_record(tmp_path):
    episode, f = example(tmp_path)
    f["affected_ids"] = [episode.r1, episode.r2, episode.r3]
    assert_miss(submit(episode, [f]))


def test_M03_extra_existing_evidence(tmp_path):
    episode, f = example(tmp_path)
    f["evidence_ids"] = [episode.d2, episode.d1, episode.d3]
    assert_tp(submit(episode, [f]))


def test_M04_unknown_evidence_next_to_minimal_set(tmp_path):
    episode, f = example(tmp_path)
    f["evidence_ids"] = [episode.d1, episode.d2, episode.u]
    assert_miss(submit(episode, [f]))


def test_M05_eleven_evidence_ids_ten_repeats(tmp_path, grader_log):
    episode, f = example(tmp_path)
    f["evidence_ids"] = [episode.d1] * 10 + [episode.d2]
    entry = assert_structural(submit(episode, [f]), grader_log)
    assert entry["structural_error"] == "too_many_evidence_ids"


@pytest.mark.parametrize("currency", ["kzt", "Kzt", "KZT ", " KZT"])
def test_M06_currency_case_or_space(tmp_path, currency):
    episode, f = example(tmp_path)
    f["currency"] = currency
    assert_miss(submit(episode, [f]))


def test_M07_amount_10001_with_zero_tolerance(tmp_path):
    episode, f = example(tmp_path)
    f["claimed_amount_minor"] = 10001
    assert_miss(submit(episode, [f]))


def test_M08_tolerance_one(tmp_path):
    episode, f = example(tmp_path, policy={"amount_tolerance_minor": {T: 1}})
    f["claimed_amount_minor"] = 10001
    assert_tp(submit(episode, [f]))
    f["claimed_amount_minor"] = 9999
    assert_tp(submit(episode, [f]))
    f["claimed_amount_minor"] = 10002
    assert_miss(submit(episode, [f]))


@pytest.mark.parametrize("amount", ["10000", True, False, 10000.0])
def test_M09_amount_as_string_or_bool(tmp_path, grader_log, amount):
    episode, f = example(tmp_path)
    f["claimed_amount_minor"] = amount
    entry = assert_structural(submit(episode, [f]), grader_log)
    assert entry["structural_error"] == "bad_amount"


def twin_issues(tmp_path: Path) -> EpisodeFixture:
    """A and B with identical type, ids, amount and currency; A needs D1, B needs D2."""
    # B is written first: file order must not matter.
    return make_episode(tmp_path, lambda e: [issue("B", [e.r1, e.r2], [[e.d2]]),
                                             issue("A", [e.r1, e.r2], [[e.d1]])])


def test_M10_no_greedy_matching(tmp_path, grader_log):
    episode = twin_issues(tmp_path)
    affected = [episode.r1, episode.r2]
    f0 = finding(affected, [episode.d1, episode.d2])  # fits A and B
    f1 = finding(affected, [episode.d1])  # fits A only
    result = submit(episode, [f0, f1])
    assert_scores(result, 2, 0, 0, ONE, ONE, ONE, Fraction(98, 100))
    assert last_log_entry(grader_log)["pairs"] == [["A", 1], ["B", 0]]


def test_M11_both_findings_fit_both_issues(tmp_path, grader_log):
    episode = twin_issues(tmp_path)
    affected = [episode.r1, episode.r2]
    f0 = finding(affected, [episode.d1, episode.d2])
    f1 = finding(affected, [episode.d2, episode.d1])
    result = submit(episode, [f0, f1])
    assert_scores(result, 2, 0, 0, ONE, ONE, ONE, Fraction(98, 100))
    assert last_log_entry(grader_log)["pairs"] == [["A", 0], ["B", 1]]


def test_M12_two_identical_correct_findings(tmp_path, grader_log):
    episode, f = example(tmp_path)
    result = submit(episode, [f, dict(f)])
    assert_scores(result, 1, 1, 0, Fraction(1, 2), ONE, Fraction(2, 3), Fraction(47, 150))
    entry = last_log_entry(grader_log)
    assert entry["pairs"] == [["A", 0]]
    assert entry["unmatched"] == [{"finding_index": 1, "issue_id": "A", "reason": "duplicate"}]

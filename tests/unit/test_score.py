"""score_findings / score_episode, truth loading, the grader log and the oracle."""

from __future__ import annotations

import json
import os
import random
import stat
import tempfile
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import judge_adapter
from cookedbooks.findings import ERROR_TYPES
from cookedbooks.grader import grader_log
from cookedbooks.grader.oracle import ORACLE_EXPLANATION, oracle_findings, oracle_from_issues
from cookedbooks.grader.reward import STEP_COST
from cookedbooks.grader.score import resolve_tolerances, score_episode, score_findings
from cookedbooks.grader.truth import load_truth, parse_truth
from tests.helpers.factory import default_manifest, write_episode

METRIC_KEYS = ["tp", "fp", "fn", "precision", "recall", "f1", "return"]


@pytest.fixture(autouse=True)
def log_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "logs" / "grader.jsonl"
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(path))
    return path


def entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_truth(episode_dir: Path, value: object) -> None:
    (episode_dir / "truth.json").write_text(json.dumps(value), encoding="utf-8")


# --- score_episode ------------------------------------------------------------


def test_oracle_scores_one_at_zero_steps(mini_episode):
    oracle = oracle_findings(mini_episode.path)
    assert oracle["episode_id"] == mini_episode.episode_id
    assert oracle["findings"] == [{
        "type": "duplicate_invoice", "affected_ids": ["7", "kX"], "claimed_amount_minor": 10000,
        "currency": "KZT", "evidence_ids": ["7", "kX"], "explanation": ORACLE_EXPLANATION}]
    result = score_episode(mini_episode.path, oracle, 0)
    assert result == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
                      "f1": 1.0, "return": 1.0}
    assert list(result) == METRIC_KEYS


def test_adapter_delegates(mini_episode):
    oracle = oracle_findings(mini_episode.path)
    assert judge_adapter.score_episode(str(mini_episode.path), oracle, 10)["return"] == 0.98


def test_oracle_on_clean_episode(clean_episode):
    oracle = oracle_findings(clean_episode.path)
    assert oracle == {"episode_id": "ep-clean", "findings": []}
    assert score_episode(clean_episode.path, oracle, 0)["f1"] == 1.0


def test_oracle_orders_by_issue_id_code_points(tmp_path):
    issues = [
        {"issue_id": "b", "type": "fx_error", "affected_ids": ["7"], "expected_amount_minor": 1,
         "currency": "KZT", "evidence_sets": [["7", "a"], ["7"]]},
        {"issue_id": "B", "type": "cutoff_error", "affected_ids": ["kX"], "expected_amount_minor": 2,
         "currency": "KZT"},
    ]
    issues.append(dict(issues[0], issue_id="A", type="cutoff_error"))
    oracle = oracle_from_issues(parse_truth(issues), "e")
    # "B" has no evidence sets and is skipped; "A" < "b" in code points.
    assert [f["type"] for f in oracle["findings"]] == ["cutoff_error", "fx_error"]
    assert [f["evidence_ids"] for f in oracle["findings"]] == [["7", "a"], ["7", "a"]]


@pytest.mark.parametrize("steps", [-1, True, False, 1.0, "10", None])
def test_bad_inspection_steps_is_a_caller_error(mini_episode, steps):
    with pytest.raises(ValueError):
        score_episode(mini_episode.path, oracle_findings(mini_episode.path), steps)


def test_structural_error_result_and_log(mini_episode, log_file):
    result = score_episode(mini_episode.path, {"findings": []}, 3)
    assert result == {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                      "f1": None, "return": -1.0}
    [entry] = entries(log_file)
    assert entry["structural_error"] == "missing_episode_id"
    assert entry["episode_id"] == "ep-mini"
    assert entry["metrics"] == result


def test_episode_id_mismatch_is_only_a_warning(mini_episode, log_file):
    oracle = oracle_findings(mini_episode.path)
    oracle["episode_id"] = "someone-else"
    assert score_episode(mini_episode.path, oracle, 0)["return"] == 1.0
    assert entries(log_file)[0]["warnings"] == ["submitted episode_id differs from manifest episode_id"]


def test_missing_or_empty_policy_means_zero_tolerance(mini_episode):
    finding = oracle_findings(mini_episode.path)
    finding["findings"][0]["claimed_amount_minor"] = 10001
    (mini_episode.path / "policy.json").unlink()
    assert score_episode(mini_episode.path, finding, 0)["tp"] == 0
    (mini_episode.path / "policy.json").write_text("{}", encoding="utf-8")
    assert score_episode(mini_episode.path, finding, 0)["tp"] == 0
    (mini_episode.path / "policy.json").write_text(
        json.dumps({"amount_tolerance_minor": {"duplicate_invoice": 1}}), encoding="utf-8")
    assert score_episode(mini_episode.path, finding, 0)["tp"] == 1


@pytest.mark.parametrize("bad", [True, -1, 1.5, "1", None, [1]])
def test_invalid_tolerance_is_zero_and_logged(mini_episode, log_file, bad):
    (mini_episode.path / "policy.json").write_text(
        json.dumps({"amount_tolerance_minor": {"duplicate_invoice": bad}}), encoding="utf-8")
    finding = oracle_findings(mini_episode.path)
    finding["findings"][0]["claimed_amount_minor"] = 10001
    assert score_episode(mini_episode.path, finding, 0)["tp"] == 0
    assert any("amount_tolerance_minor" in w for w in entries(log_file)[0]["warnings"])


def test_resolve_tolerances():
    assert resolve_tolerances(None) == ({}, [])
    assert resolve_tolerances({"fx_error": 3, "cutoff_error": 0}) == (
        {"cutoff_error": 0, "fx_error": 3}, [])
    tolerances, warnings = resolve_tolerances([1, 2])
    assert tolerances == {} and len(warnings) == 1


def test_five_record_hand_built_folder_with_bare_list_truth(tmp_path):
    ledger = [{"row_id": f"r{k}", "timestamp": "2026-08-01T10:00:00+05:00", "account": "AR",
               "debit_minor": 100, "credit_minor": 0, "currency": "KZT", "entity_id": "c",
               "invoice_id": None, "payment_id": None} for k in range(5)]
    folder = write_episode(tmp_path / "hand", manifest=default_manifest("ep-hand"), ledger=ledger)
    (folder / "policy.json").unlink()
    write_truth(folder, [{"issue_id": "X", "type": "cutoff_error", "affected_ids": ["r1"],
                          "expected_amount_minor": 100, "currency": "KZT",
                          "evidence_sets": [["r1", "r2"]]}])
    good = {"type": "cutoff_error", "affected_ids": ["r1"], "claimed_amount_minor": 100,
            "currency": "KZT", "evidence_ids": ["r2", "r1"], "explanation": ""}
    result = score_episode(folder, {"episode_id": "ep-hand", "findings": [good]}, 0)
    assert result["return"] == 1.0
    assert oracle_findings(folder)["findings"][0]["evidence_ids"] == ["r1", "r2"]


def test_issue_without_evidence_sets_counts_as_fn(mini_episode, log_file):
    write_truth(mini_episode.path, [{"issue_id": "1", "type": "duplicate_invoice",
                                     "affected_ids": ["7", "kX"], "expected_amount_minor": 10000,
                                     "currency": "KZT"}])
    finding = {"type": "duplicate_invoice", "affected_ids": ["7", "kX"],
               "claimed_amount_minor": 10000, "currency": "KZT", "evidence_ids": ["7", "kX"],
               "explanation": ""}
    result = score_episode(mini_episode.path, {"episode_id": "ep-mini", "findings": [finding]}, 0)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    entry = entries(log_file)[0]
    assert entry["unmatched"][0]["reason"] == "evidence_insufficient"
    assert any("no evidence sets" in w for w in entry["warnings"])


def test_score_findings_exposes_pairs_and_exact_metrics():
    issue = {"issue_id": "A", "type": "fx_error", "affected_ids": ["x"], "expected_amount_minor": 5,
             "currency": "USD", "evidence_sets": [["x"]]}
    finding = {"type": "fx_error", "affected_ids": ["x"], "claimed_amount_minor": 5,
               "currency": "USD", "evidence_ids": ["x"], "explanation": ""}
    result = score_findings(issues=[issue], existing_ids=frozenset({"x"}), tolerances={},
                            findings={"episode_id": "e", "findings": [finding, finding]},
                            inspection_steps=10)
    assert result["pairs"] == [["A", 0]]
    assert result["unmatched"] == [{"finding_index": 1, "reason": "duplicate", "issue_id": "A"}]
    assert result["exact"].ret == Fraction(47, 150)
    assert result["structural_error"] is None


# --- truth.json ---------------------------------------------------------------


def test_load_truth_wrapper_and_bare_list(mini_episode):
    [issue] = load_truth(mini_episode.path)
    assert issue["issue_id"] == "1" and issue["evidence_sets"] == [["7", "kX"]]
    write_truth(mini_episode.path, [dict(issue, extra="ignored")])
    assert load_truth(mini_episode.path) == [issue]


def test_load_truth_keeps_strings_exact(mini_episode):
    write_truth(mini_episode.path, {"issues": [{
        "issue_id": " 01", "type": "Weird ", "affected_ids": ["q ", "Q"],
        "expected_amount_minor": -3, "currency": "kzt", "evidence_sets": [[], ["q "]]}]})
    [issue] = load_truth(mini_episode.path)
    assert issue == {"issue_id": " 01", "type": "Weird ", "affected_ids": ["q ", "Q"],
                     "expected_amount_minor": -3, "currency": "kzt", "evidence_sets": [[], ["q "]],
                     "matchable": True, "warnings": []}


@pytest.mark.parametrize("text", [
    "", "not json", "{\"issues\": NaN}", "{}", "{\"issues\": {}}", "{\"issues\": null}", "42",
    "\"issues\"", "null",
])
def test_not_a_truth_file_raises_value_error(mini_episode, text):
    (mini_episode.path / "truth.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_truth(mini_episode.path)


GOOD = {"issue_id": "A", "type": "t", "affected_ids": ["7"], "expected_amount_minor": 1,
        "currency": "K", "evidence_sets": [["7"]]}


@pytest.mark.parametrize(("change", "field"), [
    ({"type": 5}, "type"),
    ({"type": None}, "type"),
    ({"currency": ["K"]}, "currency"),
    ({"affected_ids": "7"}, "affected_ids"),
    ({"affected_ids": [7]}, "affected_ids"),
    ({"expected_amount_minor": True}, "expected_amount_minor"),
    ({"expected_amount_minor": 1.0}, "expected_amount_minor"),
    ({"expected_amount_minor": "1"}, "expected_amount_minor"),
    ({"evidence_sets": ["7"]}, "evidence_sets"),
    ({"evidence_sets": [["7", 7]]}, "evidence_sets"),
    ({"evidence_sets": {"a": ["7"]}}, "evidence_sets"),
    ({"type": "__delete__"}, "type"),
    ({"expected_amount_minor": "__delete__"}, "expected_amount_minor"),
])
def test_malformed_issue_is_kept_unmatchable_with_warning(change, field):
    raw = dict(GOOD, **change)
    raw = {k: v for k, v in raw.items() if v != "__delete__"}
    [issue] = parse_truth([raw])
    assert issue["issue_id"] == "A"
    assert issue["matchable"] is False
    assert issue["evidence_sets"] == []
    assert len(issue["warnings"]) == 1 and field in issue["warnings"][0]


@pytest.mark.parametrize("raw", [1, "x", None, ["A"], True])
def test_issue_that_is_not_an_object_is_kept_unmatchable(raw):
    issues = parse_truth([GOOD, raw])
    assert [i["issue_id"] for i in issues] == ["A", "#1"]
    assert issues[1]["matchable"] is False
    assert "not an object" in issues[1]["warnings"][0]


def test_issue_id_rules():
    issues = parse_truth([dict(GOOD, issue_id=2), dict(GOOD, issue_id=None),
                          {k: v for k, v in GOOD.items() if k != "issue_id"},
                          dict(GOOD, issue_id=True), dict(GOOD, issue_id=1.5), GOOD])
    assert [i["issue_id"] for i in issues] == ["2", "#1", "#2", "#3", "#4", "A"]
    assert all(i["matchable"] for i in issues)
    assert all(len(i["warnings"]) == 1 for i in issues[:5]) and issues[5]["warnings"] == []
    # IDs inside the issue are still exact strings, never converted.
    assert issues[0]["affected_ids"] == ["7"]


def jury_issue(issue_id: object) -> dict:
    return {"issue_id": issue_id, "type": "duplicate_invoice", "affected_ids": ["7"],
            "expected_amount_minor": 10000, "currency": "KZT", "evidence_sets": [["7", "kX"]]}


def jury_finding(**change) -> dict:
    f = {"type": "duplicate_invoice", "affected_ids": ["7"], "claimed_amount_minor": 10000,
         "currency": "KZT", "evidence_ids": ["7", "kX"], "explanation": ""}
    f.update(change)
    return f


def test_integer_issue_ids_order_as_strings_and_score_jury_numbers(mini_episode, log_file):
    # Issue 2 needs D1, issue 10 needs D2. Both findings fit both, so the pairs show the
    # order: "10" < "2" in code points takes finding 0.
    write_truth(mini_episode.path, [dict(jury_issue(2), evidence_sets=[["7"]]),
                                    dict(jury_issue(10), evidence_sets=[["kX"]])])
    both = jury_finding()
    result = score_episode(mini_episode.path, {"episode_id": "ep-mini", "findings": [both, both]}, 10)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert abs(result["return"] - 0.98) <= 1e-9
    entry = entries(log_file)[0]
    assert entry["pairs"] == [["10", 0], ["2", 1]]
    assert sum("integer issue_id" in w for w in entry["warnings"]) == 2
    # Only one finding (the R13 shape): TP 1, FN 1, return 47/150.
    result = score_episode(mini_episode.path, {"episode_id": "ep-mini", "findings": [both]}, 10)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert abs(result["return"] - float(Fraction(47, 150))) <= 1e-9
    assert entries(log_file)[1]["pairs"] == [["10", 0]]


def test_one_good_one_malformed_issue(mini_episode, log_file):
    write_truth(mini_episode.path, {"issues": [jury_issue("A"),
                                               dict(jury_issue("B"), expected_amount_minor="10000")]})
    oracle = oracle_findings(mini_episode.path)
    assert len(oracle["findings"]) == 1  # the malformed issue is skipped
    result = score_episode(mini_episode.path, oracle, 10)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert abs(result["return"] - float(Fraction(47, 150))) <= 1e-9
    warnings = entries(log_file)[0]["warnings"]
    assert any("'B'" in w and "expected_amount_minor" in w for w in warnings)


def test_all_malformed_truth_still_scores(mini_episode, log_file):
    write_truth(mini_episode.path, [5, {"type": None}])
    assert oracle_findings(mini_episode.path)["findings"] == []
    result = score_episode(mini_episode.path, {"episode_id": "ep-mini", "findings": [jury_finding()]}, 0)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 2)
    entry = entries(log_file)[0]
    assert entry["unmatched"][0]["reason"] == "issue_malformed"
    assert len(entry["warnings"]) >= 2


def test_oracle_skips_issue_without_evidence_sets(mini_episode):
    write_truth(mini_episode.path, [jury_issue("A"), dict(jury_issue("B"), evidence_sets=[])])
    oracle = oracle_findings(mini_episode.path)
    assert len(oracle["findings"]) == 1
    result = score_episode(mini_episode.path, oracle, 0)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_missing_and_non_utf8_truth(mini_episode):
    (mini_episode.path / "truth.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError):
        load_truth(mini_episode.path)
    (mini_episode.path / "truth.json").unlink()
    with pytest.raises(ValueError):
        load_truth(mini_episode.path)


def test_null_or_empty_evidence_sets_accepted():
    base = {"issue_id": "A", "type": "t", "affected_ids": [], "expected_amount_minor": 1,
            "currency": "K"}
    assert parse_truth([dict(base, evidence_sets=None)])[0]["evidence_sets"] == []
    assert parse_truth([dict(base, evidence_sets=[])])[0]["evidence_sets"] == []


# --- grader log ---------------------------------------------------------------


def test_log_goes_to_env_path_not_episode_folder(mini_episode, log_file):
    before = sorted(p.name for p in mini_episode.path.iterdir())
    score_episode(mini_episode.path, oracle_findings(mini_episode.path), 7)
    score_episode(mini_episode.path, {"episode_id": "ep-mini", "findings": []}, 7)
    assert sorted(p.name for p in mini_episode.path.iterdir()) == before
    first, second = entries(log_file)
    assert first["pairs"] == [["1", 0]]
    assert first["inspection_steps"] == 7
    assert set(first) == {"episode_id", "inspection_steps", "metrics", "pairs", "unmatched",
                          "warnings", "structural_error"}
    assert second["metrics"]["fn"] == 1


def test_log_path_inside_episode_folder_is_refused(mini_episode, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(mini_episode.path / "grader.jsonl"))
    assert grader_log.log_path(mini_episode.path) == grader_log.default_log_path()
    before = sorted(p.name for p in mini_episode.path.iterdir())
    score_episode(mini_episode.path, oracle_findings(mini_episode.path), 0)
    assert sorted(p.name for p in mini_episode.path.iterdir()) == before


def test_default_log_path(monkeypatch):
    monkeypatch.delenv("COOKEDBOOKS_GRADER_LOG", raising=False)
    assert grader_log.log_path() == Path(tempfile.gettempdir()) / "cookedbooks" / "grader_log.jsonl"


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_read_only_log_location_does_not_break_scoring(mini_episode, tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(locked / "sub" / "log.jsonl"))
        result = score_episode(mini_episode.path, oracle_findings(mini_episode.path), 0)
        assert result["return"] == 1.0
    finally:
        locked.chmod(stat.S_IRWXU)


def test_log_path_that_is_a_directory_does_not_break_scoring(mini_episode, tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path))
    assert score_episode(mini_episode.path, oracle_findings(mini_episode.path), 0)["tp"] == 1


def test_log_has_no_timestamps(mini_episode, log_file):
    oracle = oracle_findings(mini_episode.path)
    score_episode(mini_episode.path, oracle, 5)
    score_episode(mini_episode.path, oracle, 5)
    first, second = log_file.read_text(encoding="utf-8").splitlines()
    assert first == second


# --- Hypothesis properties ------------------------------------------------------

POOL = ("7", "kX", "Kx", "5", "a", "a ", "0042")
EXISTING = frozenset(POOL)
CURRENCIES = ("KZT", "USD")
id_lists = st.lists(st.sampled_from(POOL), min_size=1, max_size=3)


@st.composite
def truths(draw) -> list[dict]:
    n = draw(st.integers(min_value=0, max_value=4))
    ids = draw(st.lists(st.sampled_from(("A", "B", "b", "10", "9")), min_size=n, max_size=n,
                        unique=True))
    return [{
        "issue_id": issue_id,
        "type": draw(st.sampled_from(ERROR_TYPES[:2])),
        "affected_ids": draw(id_lists),
        "expected_amount_minor": draw(st.integers(min_value=0, max_value=3)),
        "currency": draw(st.sampled_from(CURRENCIES)),
        "evidence_sets": draw(st.lists(id_lists, min_size=1, max_size=2)),
    } for issue_id in ids]


@st.composite
def findings_for(draw, issues: list[dict]) -> list[dict]:
    """Findings that are often near-copies of oracle findings, sometimes random."""
    oracle = oracle_from_issues(issues, "e")["findings"]
    out = []
    for _ in range(draw(st.integers(min_value=0, max_value=6))):
        if oracle and draw(st.booleans()):
            f = dict(draw(st.sampled_from(oracle)))
            field = draw(st.sampled_from(("none", "amount", "currency", "evidence", "affected")))
            if field == "amount":
                f["claimed_amount_minor"] += draw(st.integers(min_value=-2, max_value=2))
            elif field == "currency":
                f["currency"] = draw(st.sampled_from(CURRENCIES))
            elif field == "evidence":
                f["evidence_ids"] = f["evidence_ids"] + draw(st.lists(st.sampled_from(POOL + ("U",)), max_size=2))
            elif field == "affected":
                f["affected_ids"] = draw(id_lists)
        else:
            f = {"type": draw(st.sampled_from(ERROR_TYPES[:2])), "affected_ids": draw(id_lists),
                 "claimed_amount_minor": draw(st.integers(min_value=0, max_value=3)),
                 "currency": draw(st.sampled_from(CURRENCIES)),
                 "evidence_ids": draw(st.lists(st.sampled_from(POOL), max_size=4)),
                 "explanation": ""}
        out.append(f)
    return out


@st.composite
def cases(draw):
    issues = draw(truths())
    return (issues, draw(findings_for(issues)),
            draw(st.fixed_dictionaries({}, optional={"duplicate_invoice": st.integers(0, 2),
                                                     "unallocated_payment": st.integers(0, 2)})),
            draw(st.integers(min_value=0, max_value=120)))


def run(issues, items, tolerances, steps) -> dict:
    return score_findings(issues=issues, existing_ids=EXISTING, tolerances=tolerances,
                          findings={"episode_id": "e", "findings": items}, inspection_steps=steps)


def key(result: dict) -> tuple:
    m = result["metrics"]
    return m["tp"], m["fp"], m["fn"], result["exact"].ret


@settings(max_examples=200, deadline=None)
@given(cases(), st.randoms(use_true_random=False))
def test_permuting_findings_changes_nothing(case, rnd: random.Random):
    issues, items, tolerances, steps = case
    shuffled = list(items)
    rnd.shuffle(shuffled)
    assert key(run(issues, items, tolerances, steps)) == key(run(issues, shuffled, tolerances, steps))


@settings(max_examples=200, deadline=None)
@given(cases(), st.data())
def test_duplicate_finding_never_increases_return(case, data):
    issues, items, tolerances, steps = case
    if not items:
        return
    extra = data.draw(st.sampled_from(items))
    before = run(issues, items, tolerances, steps)["exact"].ret
    after = run(issues, items + [dict(extra)], tolerances, steps)["exact"].ret
    assert after <= before


@settings(max_examples=200, deadline=None)
@given(cases(), st.data())
def test_extra_finding_never_improves_oracle(case, data):
    issues, items, tolerances, steps = case
    oracle = oracle_from_issues(issues, "e")["findings"]
    base = run(issues, oracle, tolerances, steps)["exact"].ret
    for extra in items[:3]:
        position = data.draw(st.integers(min_value=0, max_value=len(oracle)))
        assert run(issues, oracle[:position] + [extra] + oracle[position:], tolerances, steps)["exact"].ret <= base


@settings(max_examples=200, deadline=None)
@given(cases())
def test_oracle_achieves_maximum_return(case):
    issues, items, tolerances, steps = case
    oracle = oracle_from_issues(issues, "e")["findings"]
    best = run(issues, oracle, tolerances, steps)
    assert best["exact"].f1 == 1
    assert best["exact"].ret == max(Fraction(-1), 1 - STEP_COST * steps)
    assert run(issues, items, tolerances, steps)["exact"].ret <= best["exact"].ret


@settings(max_examples=200, deadline=None)
@given(cases())
def test_one_more_step_costs_exactly_0002_when_unclipped(case):
    issues, items, tolerances, steps = case
    now = run(issues, items, tolerances, steps)["exact"]
    later = run(issues, items, tolerances, steps + 1)["exact"]
    unclipped = 2 * now.f1 - 1 - STEP_COST * (steps + 1)
    if -1 <= unclipped <= 1 and now.ret == 2 * now.f1 - 1 - STEP_COST * steps:
        assert now.ret - later.ret == Fraction(2, 1000)
    else:
        assert later.ret <= now.ret

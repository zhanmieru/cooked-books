"""Matcher: the five conditions, exact strings, and the deterministic maximum matching."""

from __future__ import annotations

import itertools
import time

import pytest

from cookedbooks.grader.matcher import (
    UNMATCHED_REASONS,
    UnmatchedFinding,
    can_share_finding,
    choose_pairs,
    fits,
    match,
    mismatch_reason,
)

EXISTING = frozenset({"7", "kX", "5", "Kx", "B7", "a", "0042", "b7", "2026-08-03:USD:KZT"})


def issue(issue_id="A", *, type_="duplicate_invoice", affected=("7",), amount=10000,
          currency="KZT", evidence_sets=(("7", "kX"),)) -> dict:
    return {"issue_id": issue_id, "type": type_, "affected_ids": list(affected),
            "expected_amount_minor": amount, "currency": currency,
            "evidence_sets": [list(s) for s in evidence_sets]}


def finding(*, type_="duplicate_invoice", affected=("7",), amount=10000, currency="KZT",
            evidence=("7", "kX")) -> dict:
    return {"type": type_, "affected_ids": list(affected), "claimed_amount_minor": amount,
            "currency": currency, "evidence_ids": list(evidence), "explanation": ""}


def reason(f: dict, i: dict | None = None, tolerance: int = 0) -> str | None:
    return mismatch_reason(f, i or issue(), EXISTING, tolerance)


# --- the five conditions ----------------------------------------------------


def test_correct_finding_fits():
    assert reason(finding()) is None
    assert fits(finding(), issue(), EXISTING, 0)


def test_condition_1_type():
    assert reason(finding(type_="cutoff_error")) == "wrong_type"
    assert reason(finding(type_="Duplicate_invoice")) == "wrong_type"
    assert reason(finding(type_="made_up")) == "wrong_type"
    # Truth may carry any type string; equal strings match.
    assert reason(finding(type_="weird"), issue(type_="weird")) is None


def test_condition_2_affected_ids_as_set():
    two = issue(affected=("7", "B7"))
    assert reason(finding(affected=("B7", "7", "7")), two) is None
    assert reason(finding(affected=("7",)), two) == "wrong_ids"
    assert reason(finding(affected=("7", "B7", "a")), two) == "wrong_ids"
    assert reason(finding(affected=()), two) == "wrong_ids"
    assert reason(finding(affected=()), issue(affected=())) is None


@pytest.mark.parametrize("claimed", ["7 ", " 7", "07", "7.0", "kx", "KX"])
def test_condition_2_exact_strings(claimed):
    assert reason(finding(affected=(claimed,))) == "wrong_ids"


def test_condition_2_case_matters_between_kx_and_kX():
    i = issue(affected=("kX",))
    assert reason(finding(affected=("kX",)), i) is None
    assert reason(finding(affected=("Kx",)), i) == "wrong_ids"


def test_condition_2_trailing_space_id_in_truth():
    i = issue(affected=("q ",), evidence_sets=(("q ",),))
    existing = EXISTING | {"q "}
    assert mismatch_reason(finding(affected=("q ",), evidence=("q ",)), i, existing, 0) is None
    assert mismatch_reason(finding(affected=("q",), evidence=("q ",)), i, existing, 0) == "wrong_ids"


def test_condition_3_currency():
    assert reason(finding(currency="USD")) == "wrong_currency"
    assert reason(finding(currency="kzt")) == "wrong_currency"
    assert reason(finding(currency="KZT ")) == "wrong_currency"


def test_condition_4_amount_and_tolerance_boundaries():
    assert reason(finding(amount=10001)) == "amount_off"
    assert reason(finding(amount=9999)) == "amount_off"
    assert reason(finding(amount=10001), tolerance=1) is None
    assert reason(finding(amount=9999), tolerance=1) is None
    assert reason(finding(amount=10002), tolerance=1) == "amount_off"
    assert reason(finding(amount=9998), tolerance=1) == "amount_off"
    assert reason(finding(amount=-10000)) == "amount_off"
    assert reason(finding(amount=0), tolerance=10000) is None


def test_condition_5_evidence():
    assert reason(finding(evidence=("kX", "7", "5"))) is None  # extra existing ID is fine
    assert reason(finding(evidence=("7",))) == "evidence_insufficient"
    assert reason(finding(evidence=())) == "evidence_insufficient"
    assert reason(finding(evidence=("7", "kX", "U"))) == "evidence_unknown"
    assert reason(finding(evidence=("7", "U"))) == "evidence_unknown"
    assert reason(finding(evidence=("7", "kx"))) == "evidence_unknown"


def test_condition_5_any_one_set_is_enough():
    i = issue(evidence_sets=(("7", "B7"), ("7", "a")))
    assert reason(finding(evidence=("7", "a")), i) is None
    assert reason(finding(evidence=("B7", "7")), i) is None
    assert reason(finding(evidence=("B7", "a")), i) == "evidence_insufficient"


def test_condition_5_issue_without_sets_never_matches():
    assert reason(finding(), issue(evidence_sets=())) == "evidence_insufficient"


def test_condition_5_id_from_any_id_space_exists():
    i = issue(evidence_sets=(("Kx", "2026-08-03:USD:KZT", "0042"),))
    assert reason(finding(evidence=("Kx", "2026-08-03:USD:KZT", "0042")), i) is None


def test_explanation_is_ignored():
    f = finding()
    f["explanation"] = "nonsense"
    assert reason(f) is None


# --- matching ---------------------------------------------------------------


def test_empty_cases():
    assert match([], [], EXISTING, {}).pairs == ()
    result = match([finding()], [], EXISTING, {})
    assert result.pairs == ()
    assert result.unmatched == (UnmatchedFinding(0, "no_issues", None),)
    assert match([], [issue()], EXISTING, {}).pairs == ()


def test_duplicate_is_reported():
    result = match([finding(), finding()], [issue()], EXISTING, {})
    assert result.pairs == (("A", 0),)
    assert result.unmatched == (UnmatchedFinding(1, "duplicate", "A"),)


def test_tolerance_is_per_issue_type():
    issues = [issue("A"), issue("B", type_="fx_error")]
    findings = [finding(amount=10001), finding(type_="fx_error", amount=10001)]
    result = match(findings, issues, EXISTING, {"fx_error": 1})
    assert result.pairs == (("B", 1),)


def test_no_greedy_matching():
    # F0 fits A and B, F1 fits A only: the maximum matching needs F1->A.
    issues = [issue("A", evidence_sets=(("7",),)), issue("B", evidence_sets=(("kX",),))]
    findings = [finding(evidence=("7", "kX")), finding(evidence=("7",))]
    assert match(findings, issues, EXISTING, {}).pairs == (("A", 1), ("B", 0))


def test_issue_id_code_point_order_not_file_order():
    # "B" < "a" in code points; both issues fit both findings.
    issues = [issue("a", evidence_sets=(("7",),)), issue("B", evidence_sets=(("7",),))]
    findings = [finding(), finding()]
    assert match(findings, issues, EXISTING, {}).pairs == (("B", 0), ("a", 1))


def test_numeric_issue_ids_sort_as_strings():
    issues = [issue("10"), issue("9")]
    assert match([finding(), finding()], issues, EXISTING, {}).pairs == (("10", 0), ("9", 1))


def test_issue_left_unmatched_when_no_maximum_matching_covers_it():
    # Issue 0 fits finding 0; issue 1 fits findings 0 and 1; issue 2 fits finding 1.
    # Max size 2. Issue 0 takes finding 0, issue 1 then takes finding 1, issue 2 stays out.
    assert choose_pairs([0, 1, 2], {0: [0], 1: [0, 1], 2: [1]}) == [(0, 0), (1, 1)]
    # Issue 1 fits only finding 0, which issue 0 needs: issue 1 is skipped, not forced.
    assert choose_pairs([0, 1, 2], {0: [0], 1: [0], 2: [1]}) == [(0, 0), (2, 1)]


def brute_force_pairs(order, edges, n_findings):
    """Lexicographically smallest sorted pair sequence among all maximum matchings."""
    best_size, best = -1, None
    options = [[None] + list(edges[i]) for i in order]
    for choice in itertools.product(*options):
        used = [f for f in choice if f is not None]
        if len(used) != len(set(used)):
            continue
        pairs = sorted((rank, f) for rank, f in enumerate(choice) if f is not None)
        if len(pairs) > best_size or (len(pairs) == best_size and pairs < best):
            best_size, best = len(pairs), pairs
    return best


def test_choose_pairs_matches_brute_force_on_all_small_graphs():
    n_issues, n_findings = 3, 3
    cells = [(i, f) for i in range(n_issues) for f in range(n_findings)]
    for mask in range(1 << len(cells)):
        edges = {i: [] for i in range(n_issues)}
        for bit, (i, f) in enumerate(cells):
            if mask >> bit & 1:
                edges[i].append(f)
        order = list(range(n_issues))
        got = sorted(choose_pairs(order, edges))
        assert got == brute_force_pairs(order, edges, n_findings), edges


def test_thirty_by_thirty_is_fast():
    issues = [issue(f"I{k:02d}", evidence_sets=(("7",),)) for k in range(30)]
    findings = [finding() for _ in range(30)]
    start = time.perf_counter()
    result = match(findings, issues, EXISTING, {})
    assert time.perf_counter() - start < 2
    assert result.pairs == tuple((f"I{k:02d}", k) for k in range(30))


def test_thirty_by_thirty_sparse_chain_is_fast():
    # Issue k fits findings k and k+1 (reversed priority makes augmenting paths long).
    order = list(range(30))
    edges = {k: [f for f in (k, k + 1) if f < 30] for k in order}
    start = time.perf_counter()
    pairs = choose_pairs(order, edges)
    assert time.perf_counter() - start < 2
    assert pairs == [(k, k) for k in range(30)]


# --- best-candidate reason --------------------------------------------------


def test_best_candidate_is_issue_where_finding_gets_furthest():
    issues = [
        issue("A", type_="fx_error"),                     # fails at type
        issue("B", affected=("B7",)),                     # fails at ids
        issue("C", amount=5),                             # fails at amount
    ]
    result = match([finding()], issues, EXISTING, {})
    assert result.unmatched == (UnmatchedFinding(0, "amount_off", "C"),)


def test_best_candidate_ties_go_to_smaller_issue_id():
    issues = [issue("b", currency="USD"), issue("B", currency="EUR")]
    result = match([finding()], issues, EXISTING, {})
    assert result.unmatched == (UnmatchedFinding(0, "wrong_currency", "B"),)


def test_all_reasons_documented():
    assert set(UNMATCHED_REASONS) == {
        "wrong_type", "wrong_ids", "wrong_currency", "amount_off", "evidence_insufficient",
        "evidence_unknown", "issue_malformed", "duplicate", "no_issues"}


def test_malformed_issue_fits_nothing_and_is_never_preferred():
    bad = {"issue_id": "0", "type": ["x"], "affected_ids": [], "expected_amount_minor": None,
           "currency": None, "evidence_sets": [], "matchable": False}
    assert reason(finding(), bad) == "issue_malformed"
    result = match([finding(currency="USD")], [bad, issue("Z")], EXISTING, {})
    assert result.unmatched == (UnmatchedFinding(0, "wrong_currency", "Z"),)
    result = match([finding()], [bad], EXISTING, {})
    assert result.pairs == ()
    assert result.unmatched == (UnmatchedFinding(0, "issue_malformed", "0"),)


# --- twin issues ------------------------------------------------------------


def test_can_share_finding_each_condition():
    a = issue("A", evidence_sets=(("7",),))
    assert can_share_finding(a, issue("B", evidence_sets=(("kX",),)), EXISTING, {})
    assert can_share_finding(a, issue("B", affected=("7", "7")), EXISTING, {})
    assert not can_share_finding(a, issue("B", type_="fx_error"), EXISTING, {})
    assert not can_share_finding(a, issue("B", affected=("7", "B7")), EXISTING, {})
    assert not can_share_finding(a, issue("B", currency="USD"), EXISTING, {})
    assert not can_share_finding(a, issue("B", amount=10001), EXISTING, {})
    # One claim within tau of both: |10000 - 10002| <= 2 * 1.
    assert can_share_finding(a, issue("B", amount=10002), EXISTING, {"duplicate_invoice": 1})
    assert not can_share_finding(a, issue("B", amount=10003), EXISTING, {"duplicate_invoice": 1})
    # The union of one set from each must exist and fit in 10 evidence IDs.
    assert not can_share_finding(a, issue("B", evidence_sets=(("U",),)), EXISTING, {})
    many = tuple(f"m{k}" for k in range(10))
    assert not can_share_finding(a, issue("B", evidence_sets=(many,)), EXISTING | set(many), {})
    assert can_share_finding(a, issue("B", evidence_sets=(many, ("kX",))), EXISTING | set(many), {})
    assert not can_share_finding(a, issue("B", evidence_sets=()), EXISTING, {})
    assert not can_share_finding(a, dict(issue("B"), matchable=False), EXISTING, {})


def test_can_share_finding_agrees_with_fits_on_the_witness():
    a = issue("A", amount=10000, evidence_sets=(("7",),))
    b = issue("B", amount=10002, evidence_sets=(("kX",),))
    tolerances = {"duplicate_invoice": 1}
    witness = finding(amount=10001, evidence=("7", "kX"))
    assert can_share_finding(a, b, EXISTING, tolerances)
    assert fits(witness, a, EXISTING, 1) and fits(witness, b, EXISTING, 1)

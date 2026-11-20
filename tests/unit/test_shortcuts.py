"""Lazy shortcuts stay weak: `cookedbooks.verify.shortcuts` on a fixed-seed run.

Bounds (measured at 60 seeds per config: worst trivial precision 0.588,
population lift/kind 0.7-1.7, "customer has 3+ invoices" lift/kind 1.0, the
two advances-share features 0.470 vs 0.456). With 20 seeds:

1. No trivial feature that flags at least 66 documents (200 at 60 seeds) has
   precision above 0.7.
2. Population features (customer size, late in the period, written offset,
   file position and ID controls) have lift within [0.4, 2.5] over the base
   rate of their record kind.
3. Invoices of customers with 3+ invoices have lift/kind at least 0.5.
4. The advances share of a payment that also credits AR does not separate
   errors from look-alikes: precision above and below 20 % differs by < 0.15.
5. The rule features keep precision 1.0 (the rules still find every error
   they flag), so the bounds above are not met by hiding the errors.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from cookedbooks.verify.shortcuts import Measurement, measure, render

SEEDS = 20
MIN_FLAGGED = 200 * SEEDS // 60
POPULATION = {
    ("invoice", "customer has exactly 1 invoice"),
    ("invoice", "customer has exactly 2 invoices"),
    ("invoice", "customer has 3+ invoices"),
    ("invoice", "issued in the last 10 days of the period or later"),
    ("invoice", "issued_at written with another offset"),
    ("invoice", "control: in the first half of invoices.jsonl"),
    ("invoice", "control: doc_id ends in a low character"),
    ("payment", "customer has at most 1 invoice"),
    ("payment", "customer has 3+ invoices"),
    ("payment", "control: in the first half of payments.jsonl"),
    ("payment", "control: doc_id ends in a low character"),
}


@pytest.fixture(scope="module")
def measured() -> Measurement:
    return measure(SEEDS)


def test_no_trivial_feature_is_precise(measured):
    loud = [r for r in measured.rows if r.group == "trivial" and r.flagged >= MIN_FLAGGED
            and r.precision > Fraction(7, 10)]
    assert not loud, render(measured)


def test_population_features_sit_near_lift_one(measured):
    seen = set()
    for r in measured.rows:
        if (r.kind, r.name) in POPULATION and r.flagged >= MIN_FLAGGED:
            seen.add((r.kind, r.name))
            assert Fraction(2, 5) <= r.lift_kind <= Fraction(5, 2), (r, render(measured))
    assert len(seen) >= len(POPULATION) - 1, sorted(POPULATION - seen)


def test_big_customers_carry_errors(measured):
    assert measured.row("customer has 3+ invoices", "invoice").lift_kind >= Fraction(1, 2)


def test_advances_share_does_not_separate(measured):
    high = measured.row("advances share >= 20% (with a Cr AR row)")
    low = measured.row("advances share < 20% (with a Cr AR row)")
    assert abs(high.precision - low.precision) < Fraction(15, 100), render(measured)


def test_rules_still_find_the_errors(measured):
    rules = [r for r in measured.rows if r.group == "rule"]
    assert len(rules) == 3 and all(r.precision == 1 for r in rules), render(measured)

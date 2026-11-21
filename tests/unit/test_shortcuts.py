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

6. Features aimed at fx_error and cutoff_error, bounded from the first
   measurement with their injectors (recipes D12-D20 and `mix-fx-cutoff`;
   measured precision at 20 / 40 / 60 seeds in brackets): "posted earlier
   than issued_at" (0.049 / 0.048 / 0.048) and "posted more than 2 days
   after issued_at" (0.042 / 0.047 / 0.047) and "base amount over 2x away
   from the customer's other invoices" (0.053 / 0.055 / 0.057) have lift/kind
   within [0.4, 2.5]; "on different sides of a period bound by UTC date"
   (0.270 / 0.271 / 0.270), "issued outside 08:00-20:00 local" (0.199 /
   0.191 / 0.191) and "issued within 2 days of a period bound" (0.180 /
   0.180 / 0.179) have precision at most 0.40; "differs from the table rate
   of the written date" (0.477 / 0.492 / 0.488) at most 0.60.
7. "posted amount matches no table rate within +-3 days" (precision 1.000)
   is a known, unremoved limit (KNOWN_LIMITS): comparing the implied rate
   with the pair's table is the audit procedure itself, and the clean books
   have no legitimate case of it (rates are published every business day).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from cookedbooks.verify.shortcuts import MIX_FX_CUTOFF_CONFIG, Measurement, default_configs, measure, render

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
LIFT_BOUNDED = {
    "posted earlier than issued_at",
    "posted more than 2 days after issued_at",
    "foreign: base amount over 2x away from the customer's other invoices",
}
PRECISION_BOUNDED = {
    "issued_at and posting on different sides of a period bound by UTC date": Fraction(40, 100),
    "issued outside 08:00-20:00 local": Fraction(40, 100),
    "issued within 2 days of a period bound": Fraction(40, 100),
    "foreign: posted amount differs from the table rate of the written date": Fraction(60, 100),
}
KNOWN_LIMITS = {"foreign: posted amount matches no table rate within +-3 days"}


@pytest.fixture(scope="module")
def measured() -> Measurement:
    return measure(SEEDS)


def test_no_trivial_feature_is_precise(measured):
    loud = [r for r in measured.rows if r.group == "trivial" and r.flagged >= MIN_FLAGGED
            and r.precision > Fraction(7, 10) and r.name not in KNOWN_LIMITS]
    assert not loud, render(measured)


def test_fx_and_cutoff_features_stay_weak(measured):
    for name in LIFT_BOUNDED:
        row = measured.row(name, "invoice")
        assert row.flagged >= MIN_FLAGGED and Fraction(2, 5) <= row.lift_kind <= Fraction(5, 2), \
            (row, render(measured))
    for name, bound in PRECISION_BOUNDED.items():
        row = measured.row(name, "invoice")
        assert row.flagged >= MIN_FLAGGED and row.precision <= bound, (row, render(measured))


def test_known_limits_are_still_measured(measured):
    """The exemption in 7. covers a feature the probe still reports, not a dead name."""
    for name in KNOWN_LIMITS:
        assert measured.row(name, "invoice").flagged > 0, render(measured)


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


def test_recipes_are_found_automatically():
    configs = default_configs()
    assert {"D06", "D07", "D08", "D09", "D10", "D11", "mix"} <= set(configs)
    assert not {"D01", "D02", "D03", "D04", "D05"} & set(configs)
    assert configs["mix-fx-cutoff"] == MIX_FX_CUTOFF_CONFIG


def test_config_without_injector_is_skipped_and_reported():
    import judge_adapter

    def generate(seed, config, folder):
        if config.get("errors", {}).get("fx_error"):
            raise NotImplementedError("no injector is registered for fx_error")
        judge_adapter.generate_episode(seed, config, folder)

    m = measure(2, {"plain": {"errors": {"duplicate_invoice": 1}}, "fx": {"errors": {"fx_error": 1}}}, generate)
    assert m.episodes == 2 and m.skipped == {"fx": "no injector is registered for fx_error"}
    assert "fx: no injector is registered for fx_error" in render(m)

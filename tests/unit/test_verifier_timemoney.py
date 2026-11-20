"""Exact time and money helpers of the verifier."""

from datetime import date
from fractions import Fraction

import pytest

from cookedbooks.verifier import money, timeutil


@pytest.mark.parametrize("text, seconds", [
    ("Z", 0), ("+00:00", 0), ("+05:00", 18000), ("+05:30", 19800), ("-05:00", -18000), ("-00:00", 0),
])
def test_parse_offset(text, seconds):
    assert timeutil.parse_offset(text) == seconds


@pytest.mark.parametrize("bad", ["+5:00", "05:00", "+05", "UTC", "+05:60", "+24:00", "", None, "+０５:00"])
def test_parse_offset_rejects(bad):
    with pytest.raises(ValueError):
        timeutil.parse_offset(bad)


def test_same_instant_different_offsets():
    a = timeutil.parse_instant("2026-09-30T21:00:00+00:00")
    assert a == timeutil.parse_instant("2026-10-01T02:00:00+05:00")
    assert a == timeutil.parse_instant("2026-09-30T21:00:00Z")
    assert a == timeutil.parse_instant("2026-10-01T02:30:00+05:30")
    assert a == timeutil.parse_instant("2026-09-30T16:00:00-05:00")


@pytest.mark.parametrize("bad", [
    "2026-09-30T21:00:00", "2026-09-30 21:00:00+00:00", "2026-09-30T21:00+00:00",
    "2026-09-30T21:00:00.5+00:00", "2026-02-30T00:00:00Z", "2026-09-30T24:00:00Z", "2026-09-30", 20260930,
])
def test_parse_instant_rejects(bad):
    with pytest.raises(ValueError):
        timeutil.parse_instant(bad)


def test_local_date_at_offsets():
    t = timeutil.parse_instant("2026-08-13T21:00:00+00:00")
    assert timeutil.local_date(t, timeutil.parse_offset("+05:00")) == date(2026, 8, 14)
    assert timeutil.local_date(t, timeutil.parse_offset("+05:30")) == date(2026, 8, 14)
    assert timeutil.local_date(t, timeutil.parse_offset("-05:00")) == date(2026, 8, 13)
    assert timeutil.local_date(t, 0) == date(2026, 8, 13)
    early = timeutil.parse_instant("2026-08-14T03:00:00+00:00")
    assert timeutil.local_date(early, timeutil.parse_offset("-05:00")) == date(2026, 8, 13)
    half = timeutil.parse_instant("2026-08-13T18:29:59+00:00")
    assert timeutil.local_date(half, timeutil.parse_offset("+05:30")) == date(2026, 8, 13)
    assert timeutil.local_date(half + 1, timeutil.parse_offset("+05:30")) == date(2026, 8, 14)


def test_period_bounds_inclusive():
    start = timeutil.parse_instant("2026-07-01T00:00:00+05:00")
    end = timeutil.parse_instant("2026-09-30T23:59:59+05:00")
    assert timeutil.in_period(start, start, end) and timeutil.in_period(end, start, end)
    assert timeutil.period_side(start - 1, start, end) == -1
    assert timeutil.period_side(end + 1, start, end) == 1
    assert timeutil.in_period(timeutil.parse_instant("2026-09-30T18:59:59Z"), start, end)
    assert not timeutil.in_period(timeutil.parse_instant("2026-09-30T19:00:00Z"), start, end)


def test_parse_rate_and_decimal():
    assert money.parse_rate("470.25") == Fraction(47025, 100)
    assert money.parse_decimal("0.005") == Fraction(1, 200)
    for bad in ["1e3", "-1.0", "0", "0.000", ".5", "5.", "", " 1.0", "1,5", 1.5, None, "١٢"]:
        with pytest.raises(ValueError):
            money.parse_rate(bad)


@pytest.mark.parametrize("value, up, even", [
    (Fraction(1, 2), 1, 0), (Fraction(3, 2), 2, 2), (Fraction(5, 2), 3, 2), (Fraction(-1, 2), -1, 0),
    (Fraction(-5, 2), -3, -2), (Fraction(49, 100), 0, 0), (Fraction(51, 100), 1, 1), (Fraction(7), 7, 7),
])
def test_rounding_modes(value, up, even):
    assert money.round_fraction(value, "half_up") == up
    assert money.round_fraction(value, "half_even") == even


def test_rounding_mode_unknown():
    with pytest.raises(ValueError):
        money.round_fraction(Fraction(1, 2), "down")


@pytest.mark.parametrize("src, rate, exp_src, exp_base, expected", [
    (100000, "470.25", 2, 2, 47025000),          # 1000.00 USD -> KZT
    (10000, "3.4567", 0, 2, 3456700),            # 10000 JPY (exp 0) -> KZT (exp 2)
    (1234, "1650.125", 3, 2, 203625),            # 1.234 KWD (exp 3) -> KZT: 2036.254... -> 203625
    (100, "150.5", 2, 0, 151),                   # 1.00 USD -> JPY: 150.5 -> 151 half_up
    (1, "0.5", 2, 2, 1),
])
def test_convert_half_up(src, rate, exp_src, exp_base, expected):
    assert money.convert(src, money.parse_rate(rate), exp_src, exp_base, "half_up") == expected


def test_convert_half_even_ties():
    assert money.convert(100, money.parse_rate("150.5"), 2, 0, "half_even") == 150
    assert money.convert(100, money.parse_rate("151.5"), 2, 0, "half_even") == 152
    assert money.convert(1, money.parse_rate("0.5"), 2, 2, "half_even") == 0


def test_materiality_bounds():
    rel = Fraction(1, 200)
    assert money.is_material(100, 20000, 100, rel)        # both exactly at the threshold
    assert not money.is_material(99, 19800, 100, rel)     # below min_minor
    assert not money.is_material(199, 40000, 100, rel)    # below 0.5 %
    assert money.is_material(200, 40000, 100, rel)

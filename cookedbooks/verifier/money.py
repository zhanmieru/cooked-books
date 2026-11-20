"""Exact money arithmetic: rates, rounding and FX conversion (DESIGN.md, section 3).

`base_minor = round(src_minor * rate * 10^(exp_base - exp_src))`, computed in
`Fraction`, rounded with `half_up` (ties away from zero) or `half_even`.
No float anywhere.
"""

from __future__ import annotations

import re
from fractions import Fraction

ROUNDING_MODES = ("half_up", "half_even")
_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def parse_decimal(text: object) -> Fraction:
    """A plain non-negative decimal string such as `"470.25"` or `"0.005"`."""
    if not isinstance(text, str) or _DECIMAL.fullmatch(text) is None:
        raise ValueError(f"not a decimal string: {text!r}")
    whole, _, frac = text.partition(".")
    return Fraction(int(whole + frac), 10 ** len(frac))


def parse_rate(text: object) -> Fraction:
    """An FX rate: a decimal string strictly greater than zero."""
    rate = parse_decimal(text)
    if rate <= 0:
        raise ValueError(f"rate must be positive: {text!r}")
    return rate


def round_fraction(value: Fraction, mode: str) -> int:
    """Round to an integer; ties go away from zero (half_up) or to even (half_even)."""
    if mode not in ROUNDING_MODES:
        raise ValueError(f"unknown rounding mode {mode!r}")
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude.numerator, magnitude.denominator)
    twice = 2 * remainder
    if twice > magnitude.denominator:
        quotient += 1
    elif twice == magnitude.denominator:
        if mode == "half_up" or quotient % 2 == 1:
            quotient += 1
    return sign * quotient


def convert(src_minor: int, rate: Fraction, exp_src: int, exp_base: int, mode: str) -> int:
    """Convert minor units of the source currency to minor units of the base currency."""
    return round_fraction(Fraction(src_minor) * rate * Fraction(10) ** (exp_base - exp_src), mode)


def is_material(diff: int, correct: int, min_minor: int, min_relative: Fraction) -> bool:
    """`diff >= min_minor` and `diff >= min_relative * correct` (both inclusive)."""
    return diff >= min_minor and Fraction(diff) >= min_relative * Fraction(abs(correct))

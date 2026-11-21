"""Currencies, the synthetic rate series (`fx.csv`) and the conversion rule of DESIGN.md section 3.

All arithmetic is exact `Decimal`. Rates are fictional: a seeded,
mean-reverting walk around invented levels, not market data.

Every published rate moves at least MIN_MOVE_BP basis points from the
previous published rate of its pair. So a conversion at a neighbouring
business day's rate, on an invoice of at least FX_TARGET_MIN_BASE_MINOR base
minor units, always differs from the correct one by more than
`fx_materiality` (DESIGN.md section 5): an fx_error is material by
construction, never "above zero but below the threshold".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

EXPONENTS: dict[str, int] = {"EUR": 2, "JPY": 0, "KWD": 3, "KZT": 2, "USD": 2}

# Invented value levels in one abstract unit; only their ratios matter.
_LEVELS: dict[str, Decimal] = {
    "EUR": Decimal("561.7"),
    "JPY": Decimal("3.184"),
    "KWD": Decimal("1587.4"),
    "KZT": Decimal("1"),
    "USD": Decimal("483.2"),
}
_REVERSION = Decimal("0.08")
_STEP_BP = 140  # daily shock up to +-1.40 % of the level
RATE_DIGITS = 5  # e.g. "470.25": exact half-way products happen, so the rounding mode matters
LOOKBACK_DAYS = 14  # rates start this many days before the export window
MIN_MOVE_BP = 65  # a published rate differs from the previous published one by at least 0.65 %
_ROUNDING = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}

# policy.fx_materiality (EVALUATION_CONTRACT.md section 2, policy.json)
FX_MIN_MINOR = 100
FX_MIN_RELATIVE = Decimal("0.005")
# fx_error targets convert to at least this many base minor units: with MIN_MOVE_BP above, a
# neighbouring business day's rate then misses the correct amount by > 0.6 % and > 200 minor units.
FX_TARGET_MIN_BASE_MINOR = 40_000


def is_material(posted: int, correct: int) -> bool:
    """policy.fx_materiality: |posted - correct| >= min_minor and >= min_relative x correct."""
    diff = abs(posted - correct)
    return diff >= FX_MIN_MINOR and Decimal(diff) >= FX_MIN_RELATIVE * correct


def convert(src_minor: int, rate: Decimal, src_currency: str, base_currency: str, rounding: str) -> int:
    """base_minor = round(src_minor * rate * 10^(exp_base - exp_src)) with policy rounding, exactly."""
    with localcontext() as ctx:
        ctx.prec = 80
        shift = Decimal(10) ** (EXPONENTS[base_currency] - EXPONENTS[src_currency])
        value = Decimal(src_minor) * rate * shift
        return int(value.quantize(Decimal(1), rounding=_ROUNDING[rounding]))


def _significant(value: Decimal, digits: int, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    exponent = value.adjusted() - (digits - 1)
    return value.quantize(Decimal(1).scaleb(exponent), rounding=rounding)


def _moved(previous: Decimal, rate: Decimal, rng: random.Random) -> Decimal:
    """`rate`, or the nearest RATE_DIGITS-digit rate at least MIN_MOVE_BP away from `previous`."""
    step = previous * MIN_MOVE_BP / 10000
    if abs(rate - previous) >= step:
        return rate
    up = rate > previous or (rate == previous and rng.randrange(2) == 0)
    target = previous + step if up else previous - step
    moved = _significant(target, RATE_DIGITS, ROUND_CEILING if up else ROUND_FLOOR)
    moved = _significant(moved, RATE_DIGITS)  # exact already; normalises 9.9999x -> 10.000
    if abs(moved - previous) < step:
        raise AssertionError("a rate move below the minimum survived rounding")
    return moved


@dataclass
class FxTable:
    """Rates per (date, source, target); `text` is exactly what fx.csv holds."""

    text: dict[tuple[date, str, str], str] = field(default_factory=dict)

    def rows(self) -> list[tuple[date, str, str, str]]:
        return [(d, s, t, rate) for (d, s, t), rate in self.text.items()]

    def rate(self, day: date, source: str, target: str) -> Decimal | None:
        text = self.text.get((day, source, target))
        return None if text is None else Decimal(text)

    def rate_on_or_before(self, day: date, source: str, target: str) -> tuple[date, Decimal]:
        """The policy rate: the given date, else the closest earlier date with a row."""
        for back in range(LOOKBACK_DAYS + 31):
            probe = day - timedelta(days=back)
            found = self.rate(probe, source, target)
            if found is not None:
                return probe, found
        raise LookupError(f"no {source}->{target} rate on or before {day}")

    @staticmethod
    def key(day: date, source: str, target: str) -> str:
        """The natural fx record key `date:SRC:TGT`."""
        return f"{day.isoformat()}:{source}:{target}"


def build_fx(rng: random.Random, first: date, last: date, base: str, sources: tuple[str, ...]) -> FxTable:
    """Business-day rates source->base for every source, from `first` to `last`."""
    table = FxTable()
    if not sources:
        return table
    walked = sorted({base, *sources})
    values = {c: _LEVELS[c] * (1 + Decimal(rng.randint(-300, 300)) / 10000) for c in walked}
    published: dict[str, Decimal] = {}
    with localcontext() as ctx:
        ctx.prec = 40
        day = first
        while day <= last:
            for code in walked:
                level = _LEVELS[code]
                shock = level * Decimal(rng.randint(-_STEP_BP, _STEP_BP)) / 10000
                values[code] = _significant(values[code] + (level - values[code]) * _REVERSION + shock, 12)
            if day.weekday() < 5:
                for source in sources:
                    rate = _significant(values[source] / values[base], RATE_DIGITS)
                    if source in published:
                        rate = _moved(published[source], rate, rng)
                        values[source] = rate * values[base]  # the walk continues from the published rate
                    published[source] = rate
                    table.text[(day, source, base)] = format(rate, "f")
            day += timedelta(days=1)
    return table

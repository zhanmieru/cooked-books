"""Currencies, the synthetic rate series (`fx.csv`) and the conversion rule of DESIGN.md section 3.

All arithmetic is exact `Decimal`. Rates are fictional: a seeded,
mean-reverting walk around invented levels, not market data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

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
_ROUNDING = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}


def convert(src_minor: int, rate: Decimal, src_currency: str, base_currency: str, rounding: str) -> int:
    """base_minor = round(src_minor * rate * 10^(exp_base - exp_src)) with policy rounding, exactly."""
    with localcontext() as ctx:
        ctx.prec = 80
        shift = Decimal(10) ** (EXPONENTS[base_currency] - EXPONENTS[src_currency])
        value = Decimal(src_minor) * rate * shift
        return int(value.quantize(Decimal(1), rounding=_ROUNDING[rounding]))


def _significant(value: Decimal, digits: int) -> Decimal:
    exponent = value.adjusted() - (digits - 1)
    return value.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_EVEN)


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
                    table.text[(day, source, base)] = format(rate, "f")
            day += timedelta(days=1)
    return table

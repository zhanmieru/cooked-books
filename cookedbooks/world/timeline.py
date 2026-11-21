"""The period, the export window and the time distributions documents are drawn from.

Boundary distance of a local date d (in days): how far d is from the nearest
period bound, counted on its own side. The first and last day of the period
and the days just outside it have distance 0.

Regular invoices (everything that is not a `boundary` distractor or an
injection target that asks for a near-bound time) are issued on weekdays at
local business hours, at boundary distance >= 3, written either with the
company offset or in UTC when the UTC date equals the local date. Their
posting is the issue instant, 1-2 local days later, or now and then 1-2 local
days earlier (a posting prepared just before the document is dated), always
at business hours; at distance >= 3 it never crosses a bound. A cutoff_error
moves a posting by one of these same delays across the period end.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from cookedbooks.world.rng import weighted_index
from cookedbooks.world.timeutil import DAY, HOUR, days, local_date, second_of_day, to_utc

WINDOW_MARGIN_DAYS = 10
REGULAR_MIN_DISTANCE = 3
NEAR_MAX_DISTANCE = 2
BUSINESS_START = 8 * HOUR
BUSINESS_END = 20 * HOUR  # exclusive
LAST_REGULAR_AFTER_END = 8  # regular invoices are issued at most E+8, so a 2-day delay stays in the window
# Posting delay in local days -> weight. Negative: posted before the document date.
POSTING_DELAY_WEIGHTS: dict[int, int] = {-2: 4, -1: 10, 0: 50, 1: 22, 2: 14}
_DELAYS = sorted(POSTING_DELAY_WEIGHTS)
_UTC_WRITTEN_PERCENT = 15


class Timeline:
    """Period [first 00:00:00, last 23:59:59] local, and a window 10 days wider on both sides."""

    def __init__(self, first: date, last: date, offset: int) -> None:
        self.offset = offset
        self.first = first
        self.last = last
        self.window_first = first - timedelta(days=WINDOW_MARGIN_DAYS)
        self.window_last = last + timedelta(days=WINDOW_MARGIN_DAYS)
        self.period_start_utc = to_utc(first, 0, offset)
        self.period_end_utc = to_utc(last, DAY - 1, offset)
        self.window_start_utc = to_utc(self.window_first, 0, offset)
        self.window_end_utc = to_utc(self.window_last, DAY - 1, offset)
        latest = last + timedelta(days=LAST_REGULAR_AFTER_END)
        span = days(self.window_first, latest)
        self.regular_dates = [d for d in span if d.weekday() < 5 and self.distance(d) >= REGULAR_MIN_DISTANCE]
        self.weekend_dates = [d for d in span if d.weekday() >= 5 and self.distance(d) >= REGULAR_MIN_DISTANCE]

    # --- facts ------------------------------------------------------------------

    def distance(self, day: date) -> int:
        """Boundary distance of a local date (see module docstring)."""
        to_start = (day - self.first).days if day >= self.first else (self.first - day).days - 1
        to_end = (self.last - day).days if day <= self.last else (day - self.last).days - 1
        return min(to_start, to_end)

    def in_period(self, instant: int) -> bool:
        return self.period_start_utc <= instant <= self.period_end_utc

    def local_date(self, instant: int) -> date:
        return local_date(instant, self.offset)

    def at(self, day: date, second: int) -> int:
        return to_utc(day, second, self.offset)

    # --- draws ------------------------------------------------------------------

    def business_second(self, rng: random.Random) -> int:
        return rng.randrange(BUSINESS_START, BUSINESS_END)

    def time_on(self, rng: random.Random, dates: list[date]) -> int:
        """A business-hours instant on one of `dates`."""
        return self.at(rng.choice(dates), self.business_second(rng))

    def regular_invoice_time(self, rng: random.Random, *, latest: date | None = None,
                             month: tuple[int, int] | None = None) -> int:
        """An issue instant from the regular distribution, optionally limited."""
        dates = [d for d in self.regular_dates
                 if (latest is None or d <= latest) and (month is None or (d.year, d.month) == month)]
        if not dates:
            raise ValueError("no regular date satisfies the constraint")
        return self.time_on(rng, dates)

    def regular_posting_time(self, rng: random.Random, issued: int) -> int:
        """Posting of a regular invoice: a delay drawn from POSTING_DELAY_WEIGHTS (see posting_at).

        A negative delay that would leave the export window is replaced by 0.
        """
        delay = _DELAYS[weighted_index(rng, [POSTING_DELAY_WEIGHTS[d] for d in _DELAYS])]
        if delay < 0 and self.local_date(issued) + timedelta(days=delay) < self.window_first:
            delay = 0
        return self.posting_at(rng, issued, delay)

    def posting_at(self, rng: random.Random, issued: int, delay: int) -> int:
        """A posting `delay` local days from `issued`: the issue instant for 0, else business hours
        on that day (strictly after `issued` for a positive delay, before it for a negative one)."""
        if delay == 0:
            return issued
        moved = self.at(self.local_date(issued) + timedelta(days=delay), self.business_second(rng))
        return max(moved, issued + 60) if delay > 0 else min(moved, issued - 60)

    def payment_time(self, rng: random.Random, after: int, *, max_days: int = 35,
                     before: int | None = None) -> int | None:
        """An instant 1 hour to `max_days` days after `after`, inside the window and strictly
        before `before` (if given); None if no room."""
        low = after + HOUR
        high = min(after + max_days * DAY, self.window_end_utc)
        if before is not None:
            high = min(high, before - 1)
        if low > high:
            return None
        return rng.randint(low, high)

    def written_offset(self, rng: random.Random, instant: int) -> int:
        """Company offset, or UTC (0) now and then when that does not change the calendar date."""
        if self.offset != 0 and rng.randrange(100) < _UTC_WRITTEN_PERCENT \
                and local_date(instant, 0) == self.local_date(instant):
            return 0
        return self.offset

    def alternative_offset(self, rng: random.Random) -> int:
        """An offset different from the company's, for timestamps written on another clock."""
        return 0 if self.offset != 0 else rng.choice([360, -300])

    def midnight_time(self, rng: random.Random, day: date, written: int) -> int:
        """An instant on local `day` whose calendar date at offset `written` is a different day."""
        gap = (written - self.offset) * 60
        second = rng.randrange(DAY - gap, DAY) if gap > 0 else rng.randrange(0, -gap)
        instant = self.at(day, second)
        assert local_date(instant, written) != day
        return instant

    def same_day_later(self, rng: random.Random, instant: int, *, up_to: int = 2 * HOUR) -> int:
        """`instant` or a little later, on the same local date."""
        room = min(up_to, DAY - 1 - second_of_day(instant, self.offset))
        return instant + rng.randint(0, room)

"""Exact time arithmetic with fixed UTC offsets (DESIGN.md, "Точные определения").

No time-zone database is used: an offset is a fixed number of seconds taken
from the text itself or from `manifest.utc_offset`. Instants are integers:
seconds since 1970-01-01T00:00:00Z. Only ASCII digits are accepted.
"""

from __future__ import annotations

import re
from datetime import date

_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_DAY = 86400

_OFFSET = re.compile(r"(?:(Z)|([+-])([0-9]{2}):([0-9]{2}))")
_TIMESTAMP = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(Z|[+-][0-9]{2}:[0-9]{2})"
)
_DATE = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})")


def parse_offset(text: object) -> int:
    """`Z`, `+05:00`, `-05:00`, `+05:30` -> offset in seconds east of UTC."""
    if not isinstance(text, str):
        raise ValueError("offset is not a string")
    match = _OFFSET.fullmatch(text)
    if match is None:
        raise ValueError(f"bad UTC offset {text!r}")
    if match.group(1):
        return 0
    hours, minutes = int(match.group(3)), int(match.group(4))
    if hours > 23 or minutes > 59:
        raise ValueError(f"bad UTC offset {text!r}")
    seconds = hours * 3600 + minutes * 60
    return -seconds if match.group(2) == "-" else seconds


def parse_date(text: object) -> date:
    """Strict `YYYY-MM-DD` calendar date."""
    if not isinstance(text, str):
        raise ValueError("date is not a string")
    match = _DATE.fullmatch(text)
    if match is None:
        raise ValueError(f"bad date {text!r}")
    year, month, day = (int(g) for g in match.groups())
    return date(year, month, day)  # raises ValueError for 2026-02-30


def parse_instant(text: object) -> int:
    """ISO-8601 timestamp with a mandatory offset, whole seconds -> UTC epoch seconds."""
    if not isinstance(text, str):
        raise ValueError("timestamp is not a string")
    match = _TIMESTAMP.fullmatch(text)
    if match is None:
        raise ValueError(f"bad timestamp {text!r} (need YYYY-MM-DDTHH:MM:SS and an offset)")
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"bad time of day in {text!r}")
    days = date(year, month, day).toordinal() - _EPOCH_ORDINAL
    return days * _DAY + hour * 3600 + minute * 60 + second - parse_offset(match.group(7))


def local_date(instant: int, offset_seconds: int) -> date:
    """Calendar day of `instant` seen at the fixed offset."""
    return date.fromordinal(_EPOCH_ORDINAL + (instant + offset_seconds) // _DAY)


def period_side(instant: int, start: int, end: int) -> int:
    """-1 before the period, 0 inside (bounds inclusive), +1 after."""
    if instant < start:
        return -1
    if instant > end:
        return 1
    return 0


def in_period(instant: int, start: int, end: int) -> bool:
    """`start <= instant <= end`."""
    return period_side(instant, start, end) == 0

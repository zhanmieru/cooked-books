"""Fixed-offset time arithmetic on integer UTC seconds. No time-zone database.

An instant is an int: seconds since 1970-01-01T00:00:00Z. Offsets are ints in
minutes east of UTC. Rendering is ISO-8601 with an explicit offset and whole
seconds, e.g. `2026-09-30T23:30:00+05:00`.
"""

from __future__ import annotations

from datetime import date, timedelta

DAY = 86_400
HOUR = 3_600
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()

# Zones without daylight saving time, mapped to their fixed offset in minutes.
TIMEZONES: dict[str, int] = {
    "UTC": 0,
    "America/Bogota": -300,
    "America/Lima": -300,
    "Asia/Almaty": 300,
    "Asia/Dubai": 240,
    "Asia/Kolkata": 330,
    "Asia/Riyadh": 180,
    "Asia/Shanghai": 480,
    "Asia/Tashkent": 300,
    "Asia/Tokyo": 540,
}


def zone_offset(name: str) -> int:
    """The fixed offset of a supported zone name; ValueError for anything else."""
    if not isinstance(name, str) or name not in TIMEZONES:
        supported = ", ".join(sorted(TIMEZONES))
        raise ValueError(f"team_config.timezone: unsupported zone {name!r}; supported: {supported}")
    return TIMEZONES[name]


def offset_text(minutes: int) -> str:
    """-300 -> '-05:00', 330 -> '+05:30'."""
    sign = "+" if minutes >= 0 else "-"
    value = abs(minutes)
    return f"{sign}{value // 60:02d}:{value % 60:02d}"


def to_utc(day: date, second_of_day: int, offset: int) -> int:
    """The instant of local `day` at `second_of_day` for a clock at `offset`."""
    return (day.toordinal() - _EPOCH_ORDINAL) * DAY + second_of_day - offset * 60


def local_date(instant: int, offset: int) -> date:
    """The calendar day of `instant` on a clock at `offset`."""
    return date.fromordinal(_EPOCH_ORDINAL + (instant + offset * 60) // DAY)


def second_of_day(instant: int, offset: int) -> int:
    return (instant + offset * 60) % DAY


def iso(instant: int, offset: int) -> str:
    """ISO-8601 text of `instant` written on a clock at `offset`."""
    day = local_date(instant, offset)
    second = second_of_day(instant, offset)
    clock = f"{second // HOUR:02d}:{second % HOUR // 60:02d}:{second % 60:02d}"
    return f"{day.isoformat()}T{clock}{offset_text(offset)}"


def days(first: date, last: date) -> list[date]:
    """Every date from `first` to `last`, both included."""
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]

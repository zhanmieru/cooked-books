"""Parse and validate `team_config` (EVALUATION_CONTRACT.md, section 8).

Every key is optional. An empty dict gives a clean single-currency company
with no distractors. An unknown key or a bad value raises ValueError with a
message naming the key; nothing is silently ignored. `config_version` and
`time_limit_s` are accepted (checked, then ignored: the environment consumes
the time limit).

Keys added by the generator (not in the contract example):
- `period_start`: first day of the reporting period, "YYYY-MM-DD", must be
  the 1st of a month. Default "2026-07-01".
- `period_months`: length of the period in months, 1-12. Default 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from cookedbooks.world.fx import EXPONENTS
from cookedbooks.world.timeutil import zone_offset

ERROR_VARIANTS: dict[str, tuple[str, ...]] = {
    "duplicate_invoice": (),
    "unallocated_payment": ("missing", "partial"),
    "fx_error": ("wrong_date", "wrong_pair"),
    "cutoff_error": ("late", "early"),
}
SCENARIOS: tuple[str, ...] = ("S1", "S2")
ISSUES_PER_SCENARIO = 2
DISTRACTOR_KINDS: tuple[str, ...] = ("split_payment", "refund", "recurring_amount", "boundary", "advance",
                                     "overpayment")
ROUNDING_MODES = ("half_up", "half_even")

ROWS_RANGE = (200, 2000)
DOCUMENTS_RANGE = (20, 200)
MAX_ISSUES = 30  # an oracle answer must fit in one submit (at most 30 findings)
MAX_DISTRACTORS_PER_KIND = 50

DEFAULTS: dict[str, Any] = {
    "config_version": "1",
    "n_ledger_rows": 600,
    "n_documents": 90,
    "base_currency": "KZT",
    "timezone": "Asia/Almaty",
    "rounding": "half_up",
    "fx_rate_date": "document_date",
    "period_start": "2026-07-01",
    "period_months": 3,
}
KNOWN_KEYS = (
    "config_version", "n_ledger_rows", "n_documents", "base_currency", "currencies",
    "timezone", "errors", "scenarios", "distractors", "rounding", "fx_rate_date",
    "time_limit_s", "period_start", "period_months",
)


@dataclass(frozen=True)
class WorldConfig:
    """A validated team_config. `errors` maps kind -> int (generator picks variants) or {variant: int}."""

    n_ledger_rows: int
    n_documents: int
    base_currency: str
    currencies: tuple[str, ...]
    timezone: str
    utc_offset: int
    errors: dict[str, int | dict[str, int]]
    scenarios: dict[str, int]
    distractors: dict[str, int]
    rounding: str
    fx_rate_date: str
    period_first: date
    period_last: date
    period_months: int
    normalized: dict[str, Any]

    @property
    def foreign_currencies(self) -> tuple[str, ...]:
        return tuple(code for code in self.currencies if code != self.base_currency)

    @property
    def generator_config(self) -> dict[str, Any]:
        """What shapes the books: `normalized` without `config_version` (a constant) and `time_limit_s`
        (the environment's), with "no errors of this kind" in one form. Two configs that mean the
        same thing have the same generator config. A bare positive number ("generator picks the
        variants") stays distinct from an explicit split.
        """
        errors = {}
        for kind, value in self.normalized["errors"].items():
            variants = ERROR_VARIANTS[kind]
            errors[kind] = {v: 0 for v in variants} if variants and value == 0 else value
        shaping = {k: v for k, v in self.normalized.items() if k not in ("config_version", "time_limit_s")}
        return {**shaping, "errors": errors}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _count(value: Any, where: str, maximum: int) -> int:
    if not _is_int(value) or not 0 <= value <= maximum:
        raise ValueError(f"team_config.{where} must be an integer from 0 to {maximum}, got {value!r}")
    return value


def _sub_object(value: Any, where: str, allowed: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"team_config.{where} must be an object")
    unknown = [key for key in value if key not in allowed]
    if unknown:
        raise ValueError(f"team_config.{where}: unknown key(s) {sorted(map(str, unknown))}; allowed: {list(allowed)}")
    return value


def _parse_errors(raw: Any) -> dict[str, int | dict[str, int]]:
    raw = _sub_object({} if raw is None else raw, "errors", tuple(ERROR_VARIANTS))
    errors: dict[str, int | dict[str, int]] = {}
    for kind, variants in ERROR_VARIANTS.items():
        value = raw.get(kind, 0 if not variants else {})
        if not variants or _is_int(value):
            errors[kind] = _count(value, f"errors.{kind}", MAX_ISSUES)
            continue
        value = _sub_object(value, f"errors.{kind}", variants)
        errors[kind] = {v: _count(value.get(v, 0), f"errors.{kind}.{v}", MAX_ISSUES) for v in variants}
    return errors


def _parse_currencies(raw: dict, base: str) -> tuple[str, ...]:
    value = raw.get("currencies", [base])
    if not isinstance(value, list) or not value:
        raise ValueError("team_config.currencies must be a non-empty list of currency codes")
    for code in value:
        if not isinstance(code, str) or code not in EXPONENTS:
            raise ValueError(f"team_config.currencies: unsupported currency {code!r}; supported: {sorted(EXPONENTS)}")
    if len(set(value)) != len(value):
        raise ValueError("team_config.currencies must not repeat a currency")
    if base not in value:
        raise ValueError(f"team_config.currencies must include the base currency {base!r}")
    if len(value) > 3:
        raise ValueError("team_config.currencies: at most the base currency and two foreign ones")
    return tuple(value)


def _parse_period(raw: dict) -> tuple[date, date, int]:
    text = raw.get("period_start", DEFAULTS["period_start"])
    try:
        first = date.fromisoformat(text) if isinstance(text, str) and len(text) == 10 else None
    except ValueError:
        first = None
    if first is None or first.day != 1 or not 2000 <= first.year <= 2100:
        raise ValueError(f"team_config.period_start must be the first day of a month as YYYY-MM-DD, got {text!r}")
    months = raw.get("period_months", DEFAULTS["period_months"])
    if not _is_int(months) or not 1 <= months <= 12:
        raise ValueError(f"team_config.period_months must be an integer from 1 to 12, got {months!r}")
    index = first.year * 12 + first.month - 1 + months
    after = date(index // 12, index % 12 + 1, 1)
    return first, date.fromordinal(after.toordinal() - 1), months


def parse_config(team_config: Any) -> WorldConfig:
    """Validate `team_config` and fill defaults. Raises ValueError with a clear message."""
    if team_config is None:
        team_config = {}
    if not isinstance(team_config, dict):
        raise ValueError("team_config must be a JSON object")
    unknown = [key for key in team_config if key not in KNOWN_KEYS]
    if unknown:
        raise ValueError(f"team_config: unknown key(s) {sorted(map(str, unknown))}")
    raw = team_config

    version = raw.get("config_version", DEFAULTS["config_version"])
    if not isinstance(version, str) or version != "1":
        raise ValueError(f'team_config.config_version must be "1", got {version!r}')
    if "time_limit_s" in raw:
        limit = raw["time_limit_s"]
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) \
                or not math.isfinite(limit) or not 0 < limit <= 60:
            raise ValueError("team_config.time_limit_s must be a number in (0, 60]")

    rows = raw.get("n_ledger_rows", DEFAULTS["n_ledger_rows"])
    if not _is_int(rows) or not ROWS_RANGE[0] <= rows <= ROWS_RANGE[1]:
        raise ValueError(f"team_config.n_ledger_rows must be an integer from {ROWS_RANGE[0]} to {ROWS_RANGE[1]}, got {rows!r}")
    documents = raw.get("n_documents", DEFAULTS["n_documents"])
    if not _is_int(documents) or not DOCUMENTS_RANGE[0] <= documents <= DOCUMENTS_RANGE[1]:
        raise ValueError(f"team_config.n_documents must be an integer from {DOCUMENTS_RANGE[0]} to {DOCUMENTS_RANGE[1]}, got {documents!r}")

    base = raw.get("base_currency", DEFAULTS["base_currency"])
    if not isinstance(base, str) or base not in EXPONENTS:
        raise ValueError(f"team_config.base_currency: unsupported currency {base!r}; supported: {sorted(EXPONENTS)}")
    currencies = _parse_currencies(raw, base)

    timezone = raw.get("timezone", DEFAULTS["timezone"])
    offset = zone_offset(timezone)

    rounding = raw.get("rounding", DEFAULTS["rounding"])
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"team_config.rounding must be one of {list(ROUNDING_MODES)}, got {rounding!r}")
    fx_rate_date = raw.get("fx_rate_date", DEFAULTS["fx_rate_date"])
    if fx_rate_date != "document_date":
        raise ValueError(f'team_config.fx_rate_date must be "document_date", got {fx_rate_date!r}')

    errors = _parse_errors(raw.get("errors"))
    scenarios_raw = _sub_object(raw.get("scenarios", {}), "scenarios", SCENARIOS)
    scenarios = {name: _count(scenarios_raw.get(name, 0), f"scenarios.{name}", MAX_ISSUES) for name in SCENARIOS}
    distractors_raw = _sub_object(raw.get("distractors", {}), "distractors", DISTRACTOR_KINDS)
    distractors = {
        kind: _count(distractors_raw.get(kind, 0), f"distractors.{kind}", MAX_DISTRACTORS_PER_KIND)
        for kind in DISTRACTOR_KINDS
    }
    issues = sum(v if isinstance(v, int) else sum(v.values()) for v in errors.values())
    issues += ISSUES_PER_SCENARIO * sum(scenarios.values())
    if issues > MAX_ISSUES:
        raise ValueError(f"team_config asks for {issues} issues; at most {MAX_ISSUES} fit in one answer")

    first, last, months = _parse_period(raw)
    normalized: dict[str, Any] = {
        "config_version": version,
        "n_ledger_rows": rows,
        "n_documents": documents,
        "base_currency": base,
        "currencies": list(currencies),
        "timezone": timezone,
        "errors": {k: (dict(v) if isinstance(v, dict) else v) for k, v in errors.items()},
        "scenarios": dict(scenarios),
        "distractors": dict(distractors),
        "rounding": rounding,
        "fx_rate_date": fx_rate_date,
        "period_start": first.isoformat(),
        "period_months": months,
    }
    if "time_limit_s" in raw:
        normalized["time_limit_s"] = raw["time_limit_s"]
    return WorldConfig(
        n_ledger_rows=rows, n_documents=documents, base_currency=base, currencies=currencies,
        timezone=timezone, utc_offset=offset, errors=errors, scenarios=scenarios,
        distractors=distractors, rounding=rounding, fx_rate_date=fx_rate_date,
        period_first=first, period_last=last, period_months=months, normalized=normalized,
    )

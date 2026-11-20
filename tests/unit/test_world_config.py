"""team_config parsing (EVALUATION_CONTRACT.md, section 8)."""

from __future__ import annotations

import pytest

from cookedbooks.world.config import parse_config
from cookedbooks.world.timeutil import TIMEZONES, zone_offset


def test_empty_config_is_a_clean_single_currency_company():
    config = parse_config({})
    assert config.currencies == ("KZT",) and config.foreign_currencies == ()
    assert config.n_ledger_rows == 600 and config.n_documents == 90
    assert all(v == 0 for v in config.distractors.values())
    assert config.errors["duplicate_invoice"] == 0
    assert config.errors["fx_error"] == {"wrong_date": 0, "wrong_pair": 0}
    assert str(config.period_first) == "2026-07-01" and str(config.period_last) == "2026-09-30"


def test_contract_example_parses():
    config = parse_config({
        "config_version": "1", "n_ledger_rows": 600, "n_documents": 90, "base_currency": "KZT",
        "currencies": ["KZT", "USD"], "timezone": "Asia/Almaty",
        "errors": {"duplicate_invoice": 0, "unallocated_payment": {"missing": 0, "partial": 0},
                   "fx_error": {"wrong_date": 0, "wrong_pair": 0}, "cutoff_error": {"late": 0, "early": 0}},
        "scenarios": {"S1": 0, "S2": 0},
        "distractors": {"split_payment": 0, "refund": 0, "recurring_amount": 0, "boundary": 0, "advance": 0},
        "rounding": "half_up", "fx_rate_date": "document_date", "time_limit_s": 60,
    })
    assert config.foreign_currencies == ("USD",)
    assert config.normalized["time_limit_s"] == 60


def test_error_kind_as_number_or_object():
    config = parse_config({"errors": {"fx_error": 2, "unallocated_payment": {"partial": 1}}})
    assert config.errors["fx_error"] == 2
    assert config.errors["unallocated_payment"] == {"missing": 0, "partial": 1}


def test_period_keys():
    config = parse_config({"period_start": "2026-11-01", "period_months": 2})
    assert str(config.period_first) == "2026-11-01" and str(config.period_last) == "2026-12-31"


def test_timezone_table():
    assert zone_offset("Asia/Almaty") == 300 and zone_offset("Asia/Kolkata") == 330
    assert zone_offset("America/Bogota") == -300 and zone_offset("UTC") == 0
    assert zone_offset("Asia/Tokyo") == 540 and zone_offset("Asia/Dubai") == 240
    assert "Europe/Berlin" not in TIMEZONES


@pytest.mark.parametrize("config, words", [
    ({"n_rows": 600}, "unknown key"),
    ({"errors": {"duplicate": 1}}, "unknown key"),
    ({"errors": {"fx_error": {"wrong_rate": 1}}}, "unknown key"),
    ({"distractors": {"refunds": 1}}, "unknown key"),
    ({"scenarios": {"S3": 1}}, "unknown key"),
    ({"n_ledger_rows": 199}, "n_ledger_rows"),
    ({"n_ledger_rows": 2001}, "n_ledger_rows"),
    ({"n_ledger_rows": 600.0}, "n_ledger_rows"),
    ({"n_ledger_rows": True}, "n_ledger_rows"),
    ({"n_documents": 19}, "n_documents"),
    ({"n_documents": 201}, "n_documents"),
    ({"base_currency": "GBP"}, "base_currency"),
    ({"base_currency": "usd"}, "base_currency"),
    ({"currencies": ["USD"]}, "base currency"),
    ({"currencies": ["KZT", "KZT"]}, "repeat"),
    ({"currencies": ["KZT", "USD", "EUR", "JPY"]}, "at most"),
    ({"currencies": []}, "currencies"),
    ({"currencies": "KZT"}, "currencies"),
    ({"timezone": "Europe/Berlin"}, "timezone"),
    ({"timezone": "+05:00"}, "timezone"),
    ({"rounding": "half_down"}, "rounding"),
    ({"fx_rate_date": "payment_date"}, "fx_rate_date"),
    ({"config_version": "2"}, "config_version"),
    ({"time_limit_s": 0}, "time_limit_s"),
    ({"time_limit_s": "60"}, "time_limit_s"),
    ({"errors": {"duplicate_invoice": -1}}, "duplicate_invoice"),
    ({"errors": {"duplicate_invoice": True}}, "duplicate_invoice"),
    ({"errors": {"duplicate_invoice": {"exact": 1}}}, "must be an integer"),
    ({"errors": {"cutoff_error": {"late": "1"}}}, "cutoff_error.late"),
    ({"errors": {"duplicate_invoice": 20, "fx_error": 11}}, "at most 30"),
    ({"scenarios": {"S1": 16}}, "at most 30"),
    ({"distractors": {"boundary": -1}}, "boundary"),
    ({"distractors": []}, "distractors"),
    ({"period_start": "2026-07-15"}, "period_start"),
    ({"period_start": "2026-7-1"}, "period_start"),
    ({"period_months": 0}, "period_months"),
])
def test_bad_values_raise_clear_value_error(config, words):
    with pytest.raises(ValueError, match=words):
        parse_config(config)


def test_not_a_dict():
    with pytest.raises(ValueError):
        parse_config([("n_documents", 20)])

"""Clean books satisfy C1-C6 for every seed and config; distractors appear exactly as configured.

The checks read only the exported folder (tests/helpers/world_checks.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import Books, books_from_folder, check_books, pattern_counts

SEEDS = range(50)
KINDS = ("split_payment", "refund", "recurring_amount", "boundary", "advance", "overpayment")
ALL_DISTRACTORS = {"split_payment": 2, "refund": 2, "recurring_amount": 2, "boundary": 4, "advance": 2,
                   "overpayment": 2}

CONFIGS = {
    "single_currency": {},
    "kzt_usd": {"currencies": ["KZT", "USD"]},
    "jpy_kwd_base_jpy": {"base_currency": "JPY", "currencies": ["JPY", "KWD"]},
    "jpy_kwd_base_kwd": {"base_currency": "KWD", "currencies": ["KWD", "JPY", "EUR"]},
    "usd_base_jpy_kwd": {"base_currency": "USD", "currencies": ["USD", "JPY", "KWD"]},
    "tz_kolkata": {"currencies": ["KZT", "EUR"], "timezone": "Asia/Kolkata", "distractors": {"boundary": 3}},
    "tz_bogota": {"currencies": ["KZT", "USD"], "timezone": "America/Bogota", "distractors": {"boundary": 3}},
    "tz_utc": {"currencies": ["KZT", "USD"], "timezone": "UTC", "distractors": {"boundary": 3}},
    "tz_tokyo": {"currencies": ["KZT", "USD"], "timezone": "Asia/Tokyo", "distractors": {"boundary": 3}},
    "tz_dubai": {"currencies": ["KZT", "USD"], "timezone": "Asia/Dubai", "distractors": {"boundary": 3}},
    "half_up": {"currencies": ["KZT", "USD", "JPY"], "rounding": "half_up"},
    "half_even": {"currencies": ["KZT", "USD", "JPY"], "rounding": "half_even"},
    "one_month": {"currencies": ["KZT", "USD"], "period_start": "2026-12-01", "period_months": 1,
                  "distractors": {"recurring_amount": 1, "boundary": 2}},
    **{f"only_{kind}": {"currencies": ["KZT", "USD"], "distractors": {kind: 3}} for kind in KINDS},
    "all_distractors": {"currencies": ["KZT", "USD", "EUR"], "distractors": ALL_DISTRACTORS},
    "all_distractors_single": {"distractors": ALL_DISTRACTORS},
}


def expected_counts(config: dict) -> dict[str, int]:
    return {kind: config.get("distractors", {}).get(kind, 0) for kind in KINDS}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_clean_books_over_50_seeds(tmp_path, name):
    config = CONFIGS[name]
    expected = expected_counts(config)
    for seed in SEEDS:
        folder = tmp_path / str(seed)
        generate_episode(seed, config, folder)
        books = books_from_folder(folder)
        assert check_books(books) == [], (name, seed)
        assert pattern_counts(books) == expected, (name, seed)
        provenance = json.loads((folder / "provenance.json").read_text())
        recorded = {kind: sum(1 for d in provenance["distractors"] if d["kind"] == kind) for kind in KINDS}
        assert recorded == expected, (name, seed)
        doc_ids = {i["doc_id"] for i in books.invoices} | {p["doc_id"] for p in books.payments}
        listed = [doc for d in provenance["distractors"] for doc in d["ids"]]
        assert set(listed) <= doc_ids and len(listed) == len(set(listed))
        assert json.loads((folder / "truth.json").read_text())["issues"] == []


def test_rounding_mode_matters_somewhere(tmp_path):
    """Half-way cases exist, so half_up and half_even books differ for the same seed."""
    differs = 0
    for seed in range(10):
        a, b = tmp_path / f"u{seed}", tmp_path / f"e{seed}"
        generate_episode(seed, CONFIGS["half_up"], a)
        generate_episode(seed, CONFIGS["half_even"], b)
        differs += (a / "ledger.csv").read_bytes() != (b / "ledger.csv").read_bytes()
    assert differs > 0


def test_exponents_zero_and_three_are_used(tmp_path):
    generate_episode(3, CONFIGS["jpy_kwd_base_kwd"], tmp_path / "ep")
    books = books_from_folder(tmp_path / "ep")
    assert books.exponents["KWD"] == 3 and books.exponents["JPY"] == 0
    assert {i["currency"] for i in books.invoices} >= {"KWD", "JPY"}


# --- the checker itself is not vacuous ------------------------------------------


def _mutated(folder: Path, mutate) -> list[str]:
    books = books_from_folder(folder)
    mutate(books)
    return check_books(Books(books.manifest, books.policy, {
        "ledger": books.ledger, "invoices": books.invoices, "payments": books.payments, "fx": books.fx}))


def test_checker_detects_broken_books(tmp_path):
    folder = tmp_path / "ep"
    generate_episode(1, {"currencies": ["KZT", "USD"], "distractors": {"advance": 1}}, folder)
    assert check_books(books_from_folder(folder)) == []

    def fx_wrong(b):
        inv = next(i for i in b.invoices if i["currency"] == "USD")
        for r in b.ledger:
            if r["invoice_id"] == inv["invoice_id"] and r["currency"] == "KZT":
                r["debit_minor" if r["debit_minor"] else "credit_minor"] += 500
    assert any(v.startswith("C3") for v in _mutated(folder, fx_wrong))

    def duplicate(b):
        b.invoices.append({**b.invoices[0], "doc_id": "x", "invoice_id": "y"})
    assert any(v.startswith("C5") for v in _mutated(folder, duplicate))

    def unbalanced(b):
        b.ledger[0]["debit_minor" if b.ledger[0]["debit_minor"] else "credit_minor"] += 1
    assert any(v.startswith("C1") for v in _mutated(folder, unbalanced))

    def late_posting(b):
        inv = next(i for i in b.invoices if b.in_period(__import__("tests.helpers.world_checks",
                                                                  fromlist=["instant"]).instant(i["issued_at"])))
        for r in b.ledger:
            if r["invoice_id"] == inv["invoice_id"] and r["payment_id"] is None:
                r["timestamp"] = "2026-10-05T10:00:00+05:00"
    assert any(v.startswith("C6") for v in _mutated(folder, late_posting))

    def open_invoice_for_advance(b):
        advance = next(p for p in b.payments if p["invoice_refs"] is None and p["amount_minor"] > 0)
        customer = b.payment_customer(advance)
        currency = advance["currency"]
        b.invoices.append({**b.invoices[0], "doc_id": "z", "invoice_id": "w", "entity_id": customer,
                           "currency": currency, "issued_at": "2026-06-21T00:00:00+05:00",
                           "po_ref": None, "line_items": [], "total_minor": 0})
        b.invoices[-1]["total_minor"] = 5
    assert any(v.startswith("C4") for v in _mutated(folder, open_invoice_for_advance))

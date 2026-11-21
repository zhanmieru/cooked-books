"""Scenarios S1 and S2 (DESIGN.md section 6), and every error kind together in one episode.

Checks read the public files, truth.json and provenance.json only; clean-state
checks come from tests/helpers/world_checks.py.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import (
    Books, check_books, convert, instant, local_day, revert_tables, written_day,
)
from tests.unit.test_inject_fx import check_fx_issue

KEY = ("entity_id", "po_ref", "service_period", "currency", "total_minor")
SETUPS = {
    "kzt_usd_almaty": {"currencies": ["KZT", "USD"]},
    "eur_base_tokyo_jpy_even": {"base_currency": "EUR", "currencies": ["EUR", "JPY", "KZT"], "timezone": "Asia/Tokyo",
                                "rounding": "half_even"},
    "kwd_base_lima": {"base_currency": "KWD", "currencies": ["KWD", "USD"], "timezone": "America/Lima"},
    "jpy_base_kolkata": {"base_currency": "JPY", "currencies": ["JPY", "EUR"], "timezone": "Asia/Kolkata"},
    "usd_base_utc": {"base_currency": "USD", "currencies": ["USD", "KZT"], "timezone": "UTC"},
}


def gen(tmp_path: Path, seed: int, config: dict, name: str | None = None) -> Path:
    folder = tmp_path / (name or f"s{seed}")
    generate_episode(seed, config, folder)
    return folder


def load(folder: Path) -> tuple[Books, dict, dict, dict]:
    episode = load_public(folder)
    books = Books(episode.manifest, episode.policy, episode.tables)
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    return books, truth, provenance, episode.tables


def rate_row_on_or_before(books: Books, currency: str, day) -> str:
    days = [r["date"] for r in books.fx if r["source_currency"] == currency and r["target_currency"] == books.base]
    return max(d for d in days if d <= day.isoformat())


def check_s1(books: Books, issues: dict, injection: dict) -> dict:
    """fx_error by the written (UTC) date and a partial allocation on one invoice near the period end."""
    assert injection["kind"] == "S1" and injection["scenario"] == "S1" and injection["variant"] is None
    fx_issue, pay_issue = (issues[i] for i in injection["issue_ids"])
    assert fx_issue["type"] == "fx_error" and pay_issue["type"] == "unallocated_payment"
    by_doc = {d["doc_id"]: d for d in books.invoices + books.payments}
    invoice = by_doc[fx_issue["affected_ids"][0]]
    payment = by_doc[pay_issue["affected_ids"][0]]
    assert pay_issue["affected_ids"] == [payment["doc_id"], invoice["doc_id"]]
    fx_changes = [c for c in injection["changes"] if c["key"] in
                  {r["row_id"] for r in books.posting[invoice["invoice_id"]]}]
    fact = check_fx_issue(books, fx_issue, fx_changes)
    # near the period end, written on another clock; for a company off UTC that clock is UTC
    local = local_day(invoice["issued_at"], books.offset)
    assert books.last_day - timedelta(days=2) <= local <= books.last_day + timedelta(days=3)
    written = written_day(invoice["issued_at"])
    assert written != local
    if books.offset != 0:
        assert invoice["issued_at"].endswith("+00:00")
    # converted by the written date's fx row, which is another row than the correct one
    used_row = rate_row_on_or_before(books, invoice["currency"], written)
    assert (used_row, invoice["currency"]) in fact["used"] and used_row != fact["key"].split(":")[0]
    # the partial allocation: remainder in the invoice currency, the only open invoice under it
    assert pay_issue["currency"] == invoice["currency"] == payment["currency"]
    rest = books.unallocated(payment)
    assert pay_issue["expected_amount_minor"] == rest == books.open_balance(invoice) > 0
    assert payment["invoice_refs"] == [invoice["invoice_id"]]
    ar = [r for r in books.pay_rows[payment["payment_id"]] if r["account"] == "AR"]
    assert len(ar) == 1 and 0 < ar[0]["credit_minor"] < payment["amount_minor"]
    customer = books.payment_customer(payment)
    open_matching = [i for i in books.invoices if i["entity_id"] == customer and i["currency"] == payment["currency"]
                     and instant(i["issued_at"]) <= instant(payment["timestamp"]) and books.open_balance(i) > 0]
    assert open_matching == [invoice]
    assert pay_issue["evidence_sets"] == [pay_issue["affected_ids"]]
    assert injection["effects"] == [
        {"control": "C3", "currency": books.base, "delta_minor": fx_issue["expected_amount_minor"]},
        {"control": "C4", "currency": invoice["currency"], "delta_minor": rest},
    ]
    ops = Counter((c["file"], c["op"]) for c in injection["changes"])
    assert ops == {("ledger.csv", "modify"): 3, ("ledger.csv", "add"): 1}
    return {"invoice": invoice, "payment": payment, "share": rest * 100 // payment["amount_minor"]}


def check_s2(books: Books, issues: dict, injection: dict) -> dict:
    """A duplicate pair of a foreign invoice whose copy is converted at another date's rate."""
    assert injection["kind"] == "S2" and injection["scenario"] == "S2" and injection["variant"] is None
    dup_issue, fx_issue = (issues[i] for i in injection["issue_ids"])
    assert dup_issue["type"] == "duplicate_invoice" and fx_issue["type"] == "fx_error"
    by_doc = {i["doc_id"]: i for i in books.invoices}
    original, copy = (by_doc[d] for d in dup_issue["affected_ids"])
    assert tuple(original[k] for k in KEY) == tuple(copy[k] for k in KEY) and original["po_ref"]
    assert sum(1 for i in books.invoices if tuple(i[k] for k in KEY) == tuple(copy[k] for k in KEY)) == 2
    assert copy["currency"] != books.base
    assert dup_issue["expected_amount_minor"] == copy["total_minor"] and dup_issue["currency"] == copy["currency"]
    assert dup_issue["evidence_sets"] == [dup_issue["affected_ids"]]
    assert fx_issue["affected_ids"] == [copy["doc_id"]]
    copy_rows = {r["row_id"] for r in books.posting[copy["invoice_id"]]}
    fx_changes = [c for c in injection["changes"] if c["key"] in copy_rows and c["op"] == "modify"]
    fact = check_fx_issue(books, fx_issue, fx_changes)
    # the copies differ in the base currency; the original is converted correctly
    original_revenue = next(r for r in books.posting[original["invoice_id"]] if r["account"] == "REVENUE")
    assert original_revenue["credit_minor"] == books.correct_base(original)[0] != fact["posted"]
    key_day = fact["key"].split(":")[0]
    assert any(s == copy["currency"] and d != key_day for d, s in fact["used"])
    assert injection["effects"] == [
        {"control": "C5", "currency": copy["currency"], "delta_minor": copy["total_minor"]},
        {"control": "C3", "currency": books.base, "delta_minor": fx_issue["expected_amount_minor"]},
    ]
    added = [c for c in injection["changes"] if c["op"] == "add"]
    assert [c["key"] for c in added if c["file"] == "invoices.jsonl"] == [copy["doc_id"]]
    assert all(c["op"] in ("add", "modify") for c in injection["changes"])
    # the copy is posted like an ordinary invoice: same side of the bounds, an ordinary delay
    posted = books.posting[copy["invoice_id"]][0]["timestamp"]
    assert books.in_period(instant(posted)) == books.in_period(instant(copy["issued_at"]))
    assert abs((local_day(posted, books.offset) - local_day(copy["issued_at"], books.offset)).days) <= 2
    return {"original": original, "copy": copy}


def check_episode(folder: Path) -> dict[str, list[dict]]:
    """Every scenario is exact; the books break exactly the claimed controls; revert is clean."""
    books, truth, provenance, tables = load(folder)
    issues = {i["issue_id"]: i for i in truth["issues"]}
    found: dict[str, list[dict]] = {"S1": [], "S2": []}
    for injection in provenance["injections"]:
        if injection["scenario"] == "S1":
            found["S1"].append(check_s1(books, issues, injection))
        elif injection["scenario"] == "S2":
            found["S2"].append(check_s2(books, issues, injection))
    claimed = Counter(e["control"] for inj in provenance["injections"] for e in inj["effects"])
    violations = Counter(v.split()[0] for v in check_books(books))
    assert violations == claimed, (violations, claimed)
    assert check_books(Books(books.manifest, books.policy, revert_tables(tables, provenance))) == []
    return found


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_s1_truth_changes_and_revert(tmp_path, setup):
    config = {**SETUPS[setup], "scenarios": {"S1": 2}, "distractors": {"boundary": 3, "overpayment": 1}}
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        found = check_episode(folder)
        assert len(found["S1"]) == 2
        truth = json.loads((folder / "truth.json").read_text())
        assert [i["type"] for i in truth["issues"]] == ["fx_error", "unallocated_payment"] * 2


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_s2_truth_changes_and_revert(tmp_path, setup):
    config = {**SETUPS[setup], "scenarios": {"S2": 2}, "distractors": {"recurring_amount": 2, "boundary": 2}}
    paid = Counter()
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, _, _, _ = load(folder)
        found = check_episode(folder)
        assert len(found["S2"]) == 2
        for pair in found["S2"]:
            paid[sum(books.open_balance(pair[k]) == 0 for k in ("original", "copy"))] += 1
    assert len(paid) >= 2  # neither / one / both copies paid, like ordinary duplicates


def test_s1_shares_the_advances_split_of_partial_and_overpayment(tmp_path):
    """The S1 remainder comes from the same 5-40 % share distribution as partial and overpayment."""
    shares = []
    for seed in range(40):
        for fact in check_episode(gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "scenarios": {"S1": 1}}))["S1"]:
            shares.append(fact["share"])
    assert 4 <= min(shares) and max(shares) <= 40 and len(set(shares)) > 15


def test_scenarios_count_as_two_issues(tmp_path):
    config = {"currencies": ["KZT", "USD"], "n_ledger_rows": 1200, "n_documents": 160,
              "errors": {"fx_error": 2}, "scenarios": {"S1": 2, "S2": 2}}
    folder = gen(tmp_path, 3, config)
    assert len(json.loads((folder / "truth.json").read_text())["issues"]) == 10
    with pytest.raises(ValueError, match="at most 30"):
        generate_episode(1, {"currencies": ["KZT", "USD"], "errors": {"fx_error": 1}, "scenarios": {"S1": 15}},
                         tmp_path / "x")


def test_scenarios_need_a_foreign_currency(tmp_path):
    for name in ("S1", "S2"):
        with pytest.raises(ValueError, match="foreign currency"):
            generate_episode(1, {"scenarios": {name: 1}}, tmp_path / "x")


ALL_KINDS = {
    "currencies": ["KZT", "USD", "EUR"], "n_ledger_rows": 2000, "n_documents": 200, "timezone": "Asia/Tashkent",
    "errors": {"duplicate_invoice": 2, "unallocated_payment": {"missing": 1, "partial": 2},
               "fx_error": {"wrong_date": 2, "wrong_pair": 2}, "cutoff_error": {"late": 2, "early": 2}},
    "scenarios": {"S1": 2, "S2": 2},
    "distractors": {"split_payment": 2, "refund": 1, "recurring_amount": 2, "boundary": 6, "advance": 1,
                    "overpayment": 2},
}


def test_every_kind_together_without_interference(tmp_path):
    """All errors and both scenarios in one episode: exact count, no shared documents outside a scenario,
    no look-alike touched, each injection breaks only its own controls."""
    for seed in range(8):
        folder = gen(tmp_path, seed, ALL_KINDS)
        books, truth, provenance, _ = load(folder)
        assert len(books.ledger) == 2000 and len(books.invoices) + len(books.payments) == 200
        assert len(truth["issues"]) == 2 + 3 + 4 + 4 + 2 * 4
        assert Counter(i["type"] for i in truth["issues"]) == {
            "duplicate_invoice": 4, "unallocated_payment": 5, "fx_error": 8, "cutoff_error": 4}
        found = check_episode(folder)
        assert len(found["S1"]) == 2 and len(found["S2"]) == 2
        issues = {i["issue_id"]: i for i in truth["issues"]}
        groups = [{d for iid in inj["issue_ids"] for d in issues[iid]["affected_ids"]}
                  for inj in provenance["injections"]]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                assert not groups[a] & groups[b], (a, b)
        lookalikes = {d for item in provenance["distractors"] for d in item["ids"]}
        assert not set().union(*groups) & lookalikes
        rows = {r["row_id"]: r for r in books.ledger}
        doc_of = {i["invoice_id"]: i["doc_id"] for i in books.invoices}
        doc_of.update({p["payment_id"]: p["doc_id"] for p in books.payments})
        for injection in provenance["injections"]:
            for change in injection["changes"]:
                if change["key"] in rows:
                    row = rows[change["key"]]
                    owner = doc_of.get(row["payment_id"] or row["invoice_id"])
                    assert owner not in lookalikes


def test_scenarios_are_deterministic(tmp_path):
    config = {"currencies": ["KZT", "USD"], "scenarios": {"S1": 1, "S2": 1}, "distractors": {"boundary": 2}}
    a, b = gen(tmp_path, 4, config, "a"), gen(tmp_path, 4, config, "b")
    for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "truth.json", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_footprints_keep_exact_sizes(tmp_path):
    for rows, docs in ((200, 40), (2000, 200)):
        config = {"n_ledger_rows": rows, "n_documents": docs, "currencies": ["KZT", "USD"],
                  "scenarios": {"S1": 2, "S2": 2}}
        for seed in range(4):
            books, truth, _, _ = load(gen(tmp_path, seed, config, f"{rows}_{seed}"))
            assert len(books.ledger) == rows and len(books.invoices) + len(books.payments) == docs
            assert len(truth["issues"]) == 8


def test_s1_written_date_rate_is_material_by_construction(tmp_path):
    """The UTC-date rate always misses the correct one by more than the threshold, never in between."""
    for seed in range(25):
        folder = gen(tmp_path, seed, {"base_currency": "JPY", "currencies": ["JPY", "USD", "KWD"],
                                      "timezone": "America/Bogota", "scenarios": {"S1": 1}})
        books, truth, _, _ = load(folder)
        fx_issue = next(i for i in truth["issues"] if i["type"] == "fx_error")
        invoice = next(i for i in books.invoices if i["doc_id"] == fx_issue["affected_ids"][0])
        correct, _ = books.correct_base(invoice)
        day = rate_row_on_or_before(books, invoice["currency"], written_day(invoice["issued_at"]))
        rate = Decimal(next(r["rate"] for r in books.fx if r["date"] == day
                            and r["source_currency"] == invoice["currency"]))
        exps = books.exponents
        wrong = convert(invoice["total_minor"], rate, exps[invoice["currency"]], exps[books.base],
                        books.policy["rounding"])
        assert abs(wrong - correct) == fx_issue["expected_amount_minor"] > 200
        assert abs(wrong - correct) * 1000 > 6 * correct

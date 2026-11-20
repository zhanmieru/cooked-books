"""The production duplicate_invoice injector, checked on generated files only.

Every check reads the public files, truth.json and provenance.json; the
clean-state checks come from tests/helpers/world_checks.py (written from
DESIGN.md, independent of the generator).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import (
    Books, books_from_folder, check_books, instant, local_day, revert_tables, written_day,
)

KEY = ("entity_id", "po_ref", "service_period", "currency", "total_minor")
SAME_FIELDS = ("entity_id", "currency", "total_minor", "line_items", "po_ref", "service_period")

SETUPS = {
    "kzt": {"currencies": ["KZT"]},
    "kzt_usd": {"currencies": ["KZT", "USD"]},
    "jpy_base": {"base_currency": "JPY", "currencies": ["JPY", "KWD"], "timezone": "America/Lima"},
    "kwd_base_even": {"base_currency": "KWD", "currencies": ["KWD", "EUR", "KZT"], "rounding": "half_even",
                      "timezone": "Asia/Kolkata"},
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


def key_of(invoice: dict) -> tuple:
    return tuple(invoice[f] for f in KEY)


def check_one_duplicate(books: Books, issue: dict, injection: dict) -> tuple[dict, dict]:
    """The exact truth issue and change set of one duplicate; returns (original, copy)."""
    by_doc = {i["doc_id"]: i for i in books.invoices}
    assert issue["type"] == "duplicate_invoice"
    assert len(issue["affected_ids"]) == 2 and issue["evidence_sets"] == [issue["affected_ids"]]
    original, copy = (by_doc[d] for d in issue["affected_ids"])
    assert key_of(original) == key_of(copy) and original["po_ref"] and original["service_period"]
    assert issue["expected_amount_minor"] == copy["total_minor"] and issue["currency"] == copy["currency"]
    for field in SAME_FIELDS:
        assert original[field] == copy[field], field
    assert original["doc_id"] != copy["doc_id"] and original["invoice_id"] != copy["invoice_id"]
    # exactly two invoices carry the key
    assert sum(1 for i in books.invoices if key_of(i) == key_of(copy)) == 2
    # the change set: the copy and its own posting, nothing else
    assert injection["kind"] == "duplicate_invoice" and injection["issue_ids"] == [issue["issue_id"]]
    assert injection["effects"] == [{"control": "C5", "currency": copy["currency"],
                                     "delta_minor": copy["total_minor"]}]
    changes = injection["changes"]
    assert all(c["op"] == "add" for c in changes)
    assert [c["key"] for c in changes if c["file"] == "invoices.jsonl"] == [copy["doc_id"]]
    assert [c["after"] for c in changes if c["file"] == "invoices.jsonl"] == [copy]
    posting = books.posting[copy["invoice_id"]]
    payers = [p for p in books.payments if copy["invoice_id"] in (p["invoice_refs"] or [])]
    added_payments = [c["after"] for c in changes if c["file"] == "payments.jsonl"]
    assert added_payments == payers and len(payers) <= 1
    payment_rows = [r for p in payers for r in books.pay_rows[p["payment_id"]]]
    assert sorted(c["key"] for c in changes if c["file"] == "ledger.csv") == \
        sorted(r["row_id"] for r in posting + payment_rows)
    assert len(changes) == 1 + (2 if copy["currency"] == books.base else 4) + 3 * len(payers)
    # the copy is either open or paid in full by one payment of its own, allocated in full
    if payers:
        pay = payers[0]
        assert pay["invoice_refs"] == [copy["invoice_id"]] and pay["amount_minor"] == copy["total_minor"]
        assert books.unallocated(pay) == 0 and books.open_balance(copy) == 0
        assert instant(pay["timestamp"]) > instant(posting[0]["timestamp"])
    else:
        assert books.applied(copy["invoice_id"]) == 0
    return original, copy


def check_copy_is_ordinary(books: Books, original: dict, copy: dict) -> None:
    """Issued like a regular invoice; posted on the same side; converted at its own date."""
    day = local_day(copy["issued_at"], books.offset)
    assert day.weekday() < 5 and books.distance(day) >= 3
    assert written_day(copy["issued_at"]) == day
    seconds = (instant(copy["issued_at"]) + books.offset * 60) % 86_400
    assert 8 * 3600 <= seconds < 20 * 3600
    year, month = map(int, copy["service_period"].split("-"))
    after = date(year + month // 12, month % 12 + 1, 1)
    assert (day.year, day.month) in {(year, month), (after.year, after.month)}
    due = local_day(copy["due_at"], books.offset)
    assert (due - day).days in (14, 30, 45) and copy["due_at"].endswith("T23:59:59" + books.manifest["utc_offset"])
    posted = instant(books.posting[copy["invoice_id"]][0]["timestamp"])
    assert 0 <= posted - instant(copy["issued_at"]) <= 3 * 86_400
    assert books.in_period(posted) == books.in_period(instant(copy["issued_at"]))
    if copy["currency"] != books.base:
        revenue = next(r for r in books.posting[copy["invoice_id"]] if r["account"] == "REVENUE")
        assert books.correct_base(copy)[0] == revenue["credit_minor"]
        assert books.correct_base(copy)[1].startswith(day.isoformat())


@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_duplicate_truth_changes_and_revert(tmp_path, setup):
    config = {**SETUPS[setup], "errors": {"duplicate_invoice": 2}}
    currencies_seen, paid_seen, earlier_seen, copy_paid = Counter(), Counter(), Counter(), Counter()
    for seed in range(12):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, tables = load(folder)
        assert len(truth["issues"]) == 2 and len(provenance["injections"]) == 2
        assert [i["issue_id"] for i in truth["issues"]] == ["ISS-0001", "ISS-0002"]
        docs = set()
        for issue, injection in zip(truth["issues"], provenance["injections"]):
            original, copy = check_one_duplicate(books, issue, injection)
            check_copy_is_ordinary(books, original, copy)
            docs.update(issue["affected_ids"])
            currencies_seen["base" if copy["currency"] == books.base else "foreign"] += 1
            paid_seen[books.open_balance(original) == 0] += 1
            copy_paid[books.open_balance(copy) == 0] += 1
            earlier_seen[instant(copy["issued_at"]) < instant(original["issued_at"])] += 1
        assert len(docs) == 4  # the two duplicates do not share a document
        # final books break C5 exactly twice and nothing else
        violations = check_books(books)
        assert len(violations) == 2 and all(v.startswith("C5") for v in violations), violations
        # reverting provenance gives clean books
        reverted = revert_tables(tables, provenance)
        assert check_books(Books(books.manifest, books.policy, reverted)) == []
    assert paid_seen[True] and paid_seen[False]  # the original is sometimes paid, sometimes open
    assert copy_paid[True] and copy_paid[False]  # and so is the copy
    assert earlier_seen[True] and earlier_seen[False]  # the copy is not always the later invoice
    if len(SETUPS[setup]["currencies"]) > 1:
        assert currencies_seen["base"] and currencies_seen["foreign"]
    else:
        assert set(currencies_seen) == {"base"}


def test_duplicate_footprint_keeps_exact_sizes(tmp_path):
    for n in (0, 1, 4):
        for rows, docs in ((200, 30), (2000, 200)):
            config = {"n_ledger_rows": rows, "n_documents": docs, "currencies": ["KZT", "USD"],
                      "errors": {"duplicate_invoice": n}}
            folder = gen(tmp_path, 7, config, f"n{n}_{rows}")
            books, truth, _, _ = load(folder)
            assert len(books.ledger) == rows and len(books.invoices) + len(books.payments) == docs
            assert len(truth["issues"]) == n


def test_duplicate_is_deterministic(tmp_path):
    config = {"currencies": ["KZT", "EUR"], "errors": {"duplicate_invoice": 3},
              "distractors": {"recurring_amount": 2}}
    a, b = gen(tmp_path, 11, config, "a"), gen(tmp_path, 11, config, "b")
    for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv",
                 "policy.json", "truth.json", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    c = gen(tmp_path, 12, config, "c")
    assert (a / "invoices.jsonl").read_bytes() != (c / "invoices.jsonl").read_bytes()


def test_many_duplicates_with_every_lookalike(tmp_path):
    """Four duplicates next to every distractor kind: exactly four C5 findings, look-alikes untouched."""
    config = {"currencies": ["KZT", "USD", "JPY"], "n_ledger_rows": 1200, "n_documents": 160,
              "errors": {"duplicate_invoice": 4},
              "distractors": {"split_payment": 2, "refund": 1, "recurring_amount": 3, "boundary": 3, "advance": 2}}
    for seed in range(8):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, _ = load(folder)
        assert len(truth["issues"]) == 4
        affected = {d for issue in truth["issues"] for d in issue["affected_ids"]}
        assert len(affected) == 8
        distractor_docs = {d for item in provenance["distractors"] for d in item["ids"]}
        assert not affected & distractor_docs
        violations = check_books(books)
        assert len(violations) == 4 and all(v.startswith("C5") for v in violations), violations
        # recurring look-alikes share po_ref, lines and total but never the service period
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for item in provenance["distractors"]:
            if item["kind"] == "recurring_amount":
                invoices = [by_doc[d] for d in item["ids"] if d in by_doc]
                assert len({i["service_period"] for i in invoices}) == len(invoices)
        # no key is shared by three or more invoices
        counts = Counter(key_of(i) for i in books.invoices if i["po_ref"] and i["service_period"])
        assert max(counts.values()) == 2


def test_duplicate_with_unallocated_payments_do_not_interfere(tmp_path):
    config = {"currencies": ["KZT", "USD"], "errors": {
        "duplicate_invoice": 3, "unallocated_payment": {"missing": 2, "partial": 2}}}
    for seed in range(6):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, tables = load(folder)
        assert [i["type"] for i in truth["issues"]] == ["duplicate_invoice"] * 3 + ["unallocated_payment"] * 4
        docs = [d for issue in truth["issues"] for d in issue["affected_ids"]]
        assert len(docs) == len(set(docs))
        violations = check_books(books)
        assert Counter(v[:2] for v in violations) == {"C5": 3, "C4": 4}, violations
        reverted = revert_tables(tables, provenance)
        assert check_books(Books(books.manifest, books.policy, reverted)) == []


def test_copy_position_is_not_a_tell(tmp_path):
    """The copy lands anywhere in invoices.jsonl, before or after its original."""
    positions = []
    for seed in range(20):
        folder = gen(tmp_path, seed, {"errors": {"duplicate_invoice": 1}})
        books, truth, provenance, _ = load(folder)
        copy_doc = provenance["injections"][0]["changes"][0]["key"]
        order = [i["doc_id"] for i in books.invoices]
        positions.append(order.index(copy_doc) / len(order))
    assert min(positions) < 0.4 and max(positions) > 0.6


def test_copy_uses_the_rate_of_its_own_date(tmp_path):
    """Foreign copies convert at their own issue date, so their base amount usually differs from the original's."""
    differs = foreign = base = 0
    for seed in range(24):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "errors": {"duplicate_invoice": 1}})
        books, truth, _, _ = load(folder)
        a, b = truth["issues"][0]["affected_ids"]
        by_doc = {i["doc_id"]: i for i in books.invoices}
        if by_doc[a]["currency"] == books.base:
            base += 1
            continue
        foreign += 1
        revenue = [next(r for r in books.posting[by_doc[d]["invoice_id"]] if r["account"] == "REVENUE")
                   for d in (a, b)]
        differs += revenue[0]["credit_minor"] != revenue[1]["credit_minor"]
    assert foreign and base  # a single duplicate is not always foreign (seeded draw)
    assert differs >= foreign // 2


def test_duplicates_sit_on_ordinary_customers(tmp_path):
    """The pair's customer is drawn like a regular invoice's: it usually has other invoices too."""
    sizes = Counter()
    for seed in range(20):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "errors": {"duplicate_invoice": 2}})
        books, truth, _, _ = load(folder)
        per_customer = Counter(i["entity_id"] for i in books.invoices)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            sizes[min(per_customer[by_doc[issue["affected_ids"][0]]["entity_id"]], 3)] += 1
    assert sizes[3] > sizes[2]  # most pairs belong to customers with 3+ invoices


def test_originals_are_issued_over_the_regular_range(tmp_path):
    """Originals are not kept away from the end of the period."""
    late = 0
    for seed in range(30):
        folder = gen(tmp_path, seed, {"errors": {"duplicate_invoice": 2}})
        books, truth, provenance, _ = load(folder)
        copies = {inj["changes"][0]["key"] for inj in provenance["injections"]}
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            original = next(by_doc[d] for d in issue["affected_ids"] if d not in copies)
            late += local_day(original["issued_at"], books.offset) > books.last_day - (books.last_day - books.first_day) / 9
    assert late >= 3


def test_due_dates_follow_ordinary_rule(tmp_path):
    folder = gen(tmp_path, 3, {"errors": {"duplicate_invoice": 4}})
    books, truth, _, _ = load(folder)
    gaps = Counter()
    for inv in books.invoices:
        gaps[(local_day(inv["due_at"], books.offset) - local_day(inv["issued_at"], books.offset)).days] += 1
    assert set(gaps) <= {14, 30, 45}

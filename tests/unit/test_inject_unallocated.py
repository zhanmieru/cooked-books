"""The production unallocated_payment injectors (missing, partial), checked on generated files only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import Books, check_books, instant, pattern_counts, revert_tables

SETUPS = {
    "kzt": {"currencies": ["KZT"]},
    "kzt_usd": {"currencies": ["KZT", "USD"]},
    "jpy_base": {"base_currency": "JPY", "currencies": ["JPY", "USD"], "timezone": "America/Lima"},
    "kwd_even": {"base_currency": "KZT", "currencies": ["KZT", "KWD", "JPY"], "rounding": "half_even",
                 "timezone": "Asia/Kolkata"},
}


def gen(tmp_path: Path, seed: int, config: dict, name: str | None = None) -> Path:
    folder = tmp_path / (name or f"s{seed}")
    generate_episode(seed, config, folder)
    return folder


def load(folder: Path) -> tuple[Books, dict, dict, dict]:
    episode = load_public(folder)
    books = Books(episode.manifest, episode.policy, episode.tables)
    books.payment_by_id = {p["payment_id"]: p for p in books.payments}
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    return books, truth, provenance, episode.tables


def check_one(books: Books, issue: dict, injection: dict) -> tuple[str, str]:
    """The exact truth issue and the DESIGN guarantee; returns (variant, shape)."""
    docs = {d["doc_id"]: d for d in books.invoices + books.payments}
    assert issue["type"] == "unallocated_payment"
    assert len(issue["affected_ids"]) == 2 and issue["evidence_sets"] == [issue["affected_ids"]]
    payment, invoice = (docs[d] for d in issue["affected_ids"])
    assert "payment_id" in payment and "invoice_id" in invoice
    customer = books.payment_customer(payment)
    rest = books.unallocated(payment)
    assert rest > 0 and issue["expected_amount_minor"] == rest and issue["currency"] == payment["currency"]
    assert payment["currency"] == invoice["currency"] and invoice["entity_id"] == customer
    assert instant(invoice["issued_at"]) <= instant(payment["timestamp"])
    # exactly one open invoice of that customer and currency issued no later than the payment
    matching = [i for i in books.invoices if i["entity_id"] == customer and i["currency"] == payment["currency"]
                and instant(i["issued_at"]) <= instant(payment["timestamp"]) and books.open_balance(i) > 0]
    assert matching == [invoice] and books.open_balance(invoice) == rest
    rows = books.pay_rows[payment["payment_id"]]
    advances = [r for r in rows if r["account"] == "CUSTOMER_ADVANCES"]
    assert len(advances) == 1 and advances[0]["credit_minor"] == rest and advances[0]["invoice_id"] is None
    cash = [r for r in rows if r["account"] == "CASH"]
    assert len(cash) == 1 and cash[0]["debit_minor"] == payment["amount_minor"]
    ar = [r for r in rows if r["account"] == "AR"]
    variant = injection["variant"]
    if variant == "missing":
        assert payment["invoice_refs"] is None and ar == [] and rest == payment["amount_minor"]
    else:
        assert payment["invoice_refs"] == [invoice["invoice_id"]]
        assert len(ar) == 1 and 0 < ar[0]["credit_minor"] < payment["amount_minor"]
        assert ar[0]["invoice_id"] == invoice["invoice_id"]
    assert injection["kind"] == "unallocated_payment" and injection["issue_ids"] == [issue["issue_id"]]
    assert injection["effects"] == [{"control": "C4", "currency": payment["currency"], "delta_minor": rest}]
    # the customer's other invoices in this currency issued no later are settled before the payment
    moment = instant(payment["timestamp"])
    for other in books.invoices:
        if other is not invoice and other["entity_id"] == customer and other["currency"] == payment["currency"] \
                and instant(other["issued_at"]) <= moment:
            settling = [books.payment_by_id[pid] for pid, rows in books.pay_rows.items()
                        for r in rows if r["account"] == "AR" and r["invoice_id"] == other["invoice_id"]]
            assert books.open_balance(other) == 0
            assert all(instant(x["timestamp"]) < moment for x in settling)
    # shape: paid by this payment alone, or after a first payment that is allocated in full
    earlier = [books.payment_by_id[pid] for pid, rows in books.pay_rows.items() if pid != payment["payment_id"]
               for r in rows if r["account"] == "AR" and r["invoice_id"] == invoice["invoice_id"]]
    shape = "single" if not earlier else "split"
    if shape == "single":
        assert payment["amount_minor"] == invoice["total_minor"]
    else:
        assert len(earlier) == 1
        first = earlier[0]
        assert instant(first["timestamp"]) < moment
        assert first["invoice_refs"] == [invoice["invoice_id"]] and books.unallocated(first) == 0
        assert first["amount_minor"] + payment["amount_minor"] == invoice["total_minor"]
    keys = {c["key"] for c in injection["changes"]}
    ops = Counter((c["file"], c["op"]) for c in injection["changes"])
    if variant == "missing":
        assert ops == {("payments.jsonl", "modify"): 1, ("ledger.csv", "modify"): 1}
        assert keys == {payment["doc_id"], advances[0]["row_id"]}
    else:
        assert ops == {("ledger.csv", "modify"): 1, ("ledger.csv", "add"): 1}
        assert keys == {ar[0]["row_id"], advances[0]["row_id"]}
    return variant, shape


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("variant", ["missing", "partial"])
def test_variant_truth_changes_and_revert(tmp_path, setup, variant):
    config = {**SETUPS[setup], "errors": {"unallocated_payment": {variant: 2}}}
    shapes, currencies = Counter(), Counter()
    for seed in range(12):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, tables = load(folder)
        assert len(truth["issues"]) == 2
        for issue, injection in zip(truth["issues"], provenance["injections"]):
            got, shape = check_one(books, issue, injection)
            assert got == variant
            shapes[shape] += 1
            currencies["base" if issue["currency"] == books.base else "foreign"] += 1
        violations = check_books(books)
        assert len(violations) == 2 and all(v.startswith("C4") for v in violations), violations
        reverted = revert_tables(tables, provenance)
        assert check_books(Books(books.manifest, books.policy, reverted)) == []
    assert shapes["single"] and shapes["split"]
    if len(SETUPS[setup]["currencies"]) > 1:
        assert currencies["base"] and currencies["foreign"]


def test_bare_number_mixes_variants(tmp_path):
    seen = Counter()
    for seed in range(10):
        folder = gen(tmp_path, seed, {"errors": {"unallocated_payment": 3}})
        books, truth, provenance, _ = load(folder)
        for issue, injection in zip(truth["issues"], provenance["injections"]):
            seen[check_one(books, issue, injection)[0]] += 1
    assert seen["missing"] and seen["partial"] and sum(seen.values()) == 30


def test_footprint_keeps_exact_sizes(tmp_path):
    for errors in ({"missing": 4}, {"partial": 4}, {"missing": 4, "partial": 4}):
        for rows, docs in ((200, 30), (2000, 200)):
            config = {"n_ledger_rows": rows, "n_documents": docs, "currencies": ["KZT", "USD"],
                      "errors": {"unallocated_payment": errors}}
            for seed in range(3):
                folder = gen(tmp_path, seed, config, f"{sorted(errors.items())}_{rows}_{seed}")
                books, truth, _, _ = load(folder)
                assert len(books.ledger) == rows and len(books.invoices) + len(books.payments) == docs
                assert len(truth["issues"]) == sum(errors.values())


def test_unallocated_is_deterministic(tmp_path):
    config = {"currencies": ["KZT", "USD"], "errors": {"unallocated_payment": {"missing": 2, "partial": 2}},
              "distractors": {"split_payment": 2, "advance": 1, "refund": 1}}
    a, b = gen(tmp_path, 5, config, "a"), gen(tmp_path, 5, config, "b")
    for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv",
                 "policy.json", "truth.json", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_lookalikes_stay_correct(tmp_path):
    """Split payments, advances and refunds next to the errors are not turned into errors."""
    config = {"currencies": ["KZT", "USD"], "n_ledger_rows": 1000, "n_documents": 140,
              "errors": {"unallocated_payment": {"missing": 3, "partial": 3}},
              "distractors": {"split_payment": 3, "advance": 3, "refund": 2, "recurring_amount": 1, "boundary": 2}}
    for seed in range(8):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, tables = load(folder)
        affected = [d for issue in truth["issues"] for d in issue["affected_ids"]]
        assert len(affected) == len(set(affected)) == 12
        distractor_docs = {d for item in provenance["distractors"] for d in item["ids"]}
        assert not set(affected) & distractor_docs
        violations = check_books(books)
        assert len(violations) == 6 and all(v.startswith("C4") for v in violations), violations
        # the payments C4 names are exactly the injected ones
        flagged = {v.split()[2].rstrip(":") for v in violations}
        by_doc = {p["doc_id"]: p for p in books.payments}
        assert flagged == {by_doc[i["affected_ids"][0]]["payment_id"] for i in truth["issues"]}
        # every distractor document is untouched by the injections
        changed = {c["key"] for inj in provenance["injections"] for c in inj["changes"]}
        assert not changed & distractor_docs
        reverted = revert_tables(tables, provenance)
        clean = Books(books.manifest, books.policy, reverted)
        assert check_books(clean) == []
        assert pattern_counts(clean)["refund"] == 2


def test_partial_allocation_is_a_proper_part(tmp_path):
    shares = []
    for seed in range(15):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "JPY"], "errors": {"unallocated_payment": {"partial": 2}}})
        books, truth, _, _ = load(folder)
        by_doc = {p["doc_id"]: p for p in books.payments}
        for issue in truth["issues"]:
            payment = by_doc[issue["affected_ids"][0]]
            shares.append(issue["expected_amount_minor"] * 100 // payment["amount_minor"])
    assert min(shares) >= 1 and max(shares) <= 99 and len(set(shares)) > 5


def test_targets_sit_on_ordinary_customers(tmp_path):
    """The target customer is drawn like a regular invoice's: it usually has other (settled) invoices."""
    sizes = Counter()
    for seed in range(20):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "errors": {"unallocated_payment": 3}})
        books, truth, _, _ = load(folder)
        per_customer = Counter(i["entity_id"] for i in books.invoices)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            sizes[min(per_customer[by_doc[issue["affected_ids"][1]]["entity_id"]], 3)] += 1
    assert sizes[3] > sizes[1] + sizes[2]


def advances_share(books: Books, payment: dict) -> int:
    return books.unallocated(payment) * 100 // payment["amount_minor"]


def test_overpayment_lookalike_is_clean_and_shares_the_split_ratio(tmp_path):
    """An overpayment closes its invoice; its advances share comes from the same range as a partial error."""
    config = {"currencies": ["KZT", "USD"], "errors": {"unallocated_payment": {"partial": 3}},
              "distractors": {"overpayment": 3}}
    error_shares, lookalike_shares = [], []
    for seed in range(20):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, tables = load(folder)
        books.payment_by_id = {p["payment_id"]: p for p in books.payments}
        by_doc = {d["doc_id"]: d for d in books.invoices + books.payments}
        flagged = [v for v in check_books(books)]
        assert len(flagged) == 3 and all(v.startswith("C4") for v in flagged), flagged
        overs = [item for item in provenance["distractors"] if item["kind"] == "overpayment"]
        assert len(overs) == 3
        for item in overs:
            invoice, payment = (by_doc[d] for d in item["ids"])
            assert payment["invoice_refs"] == [invoice["invoice_id"]] and books.open_balance(invoice) == 0
            assert payment["amount_minor"] > invoice["total_minor"]
            lookalike_shares.append(advances_share(books, payment))
        for issue in truth["issues"]:
            error_shares.append(advances_share(books, by_doc[issue["affected_ids"][0]]))
        reverted = revert_tables(tables, provenance)
        assert pattern_counts(Books(books.manifest, books.policy, reverted))["overpayment"] == 3
    for shares in (error_shares, lookalike_shares):
        assert 4 <= min(shares) and max(shares) <= 40
    # mean shares within 6 points of each other (integer arithmetic)
    e, k = error_shares, lookalike_shares
    assert abs(sum(e) * len(k) - sum(k) * len(e)) < 6 * len(e) * len(k)

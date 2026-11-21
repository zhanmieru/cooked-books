"""Unit tests of the evidence_solver policy on hand-built books.

Each rule of DESIGN.md ("Точные определения") gets a positive case and its
legitimate look-alikes; the same books relabelled with awkward IDs and
shuffled must give the same findings. The module is loaded from its file so
the test does not depend on the `cookedbooks.policies` package file.
"""

from __future__ import annotations

import ast
import importlib.util
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from cookedbooks.findings import validate_findings

SOLVER_PATH = Path(__file__).resolve().parents[2] / "cookedbooks" / "policies" / "evidence_solver.py"


def load_solver():
    name = "evidence_solver_under_test"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOLVER_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


es = load_solver()


# --- a tiny books builder ---------------------------------------------------------------


class Builder:
    """Books with the postings of DESIGN.md section 3; IDs are plain counters."""

    def __init__(self, base: str = "KZT", offset: str = "+05:00", rounding: str = "half_up") -> None:
        self.base = base
        self.manifest = {
            "episode_id": "ep-test", "base_currency": base,
            "period_start": f"2026-07-01T00:00:00{offset}", "period_end": f"2026-09-30T23:59:59{offset}",
            "utc_offset": offset, "currency_exponents": {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3},
            "rounding": {"mode": rounding, "unit": "minor"},
        }
        self.policy = {
            "fx_rate_date": "document_date", "fx_missing_rate": "previous_available",
            "fx_direction": "source_to_target", "rounding": rounding, "period_bounds": "inclusive",
            "cutoff_basis": "issued_at_local", "allocation_rule": "open_invoices_first",
            "duplicate_key": ["entity_id", "po_ref", "service_period", "currency", "total_minor"],
            "fx_materiality": {"min_relative": "0.005", "min_minor": 100},
            "amount_tolerance_minor": {},
        }
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.ledger: list[dict] = []
        self.fx: list[dict] = []
        self.n = 0

    def _next(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def _row(self, ts: str, account: str, debit: int, credit: int, currency: str,
             entity=None, invoice_id=None, payment_id=None) -> dict:
        row = {"row_id": self._next("row-"), "timestamp": ts, "account": account, "debit_minor": debit,
               "credit_minor": credit, "currency": currency, "entity_id": entity, "invoice_id": invoice_id,
               "payment_id": payment_id}
        self.ledger.append(row)
        return row

    def invoice(self, entity: str, total: int, issued: str, currency: str | None = None, po: str | None = "PO-1",
                period: str | None = "2026-08", posted: str | None = None, revenue: int | None = None) -> dict:
        currency = currency or self.base
        inv = {"doc_id": self._next("doc-"), "invoice_id": self._next("INV-"), "entity_id": entity,
               "issued_at": issued, "due_at": issued, "currency": currency, "total_minor": total,
               "line_items": [{"description": "x", "quantity": 1, "unit_price_minor": total, "amount_minor": total}],
               "po_ref": po, "service_period": period}
        self.invoices.append(inv)
        ts = posted or issued
        ref = {"entity": entity, "invoice_id": inv["invoice_id"]}
        self._row(ts, "AR", total, 0, currency, **ref)
        if currency == self.base:
            self._row(ts, "REVENUE", 0, total, currency, **ref)
        else:
            self._row(ts, "FX_CLEARING", 0, total, currency, **ref)
            self._row(ts, "FX_CLEARING", revenue, 0, self.base, **ref)
            self._row(ts, "REVENUE", 0, revenue, self.base, **ref)
        return inv

    def payment(self, entity: str, amount: int, at: str, currency: str | None = None,
                alloc: list[tuple[dict, int]] = ()) -> dict:
        currency = currency or self.base
        pay = {"doc_id": self._next("doc-"), "payment_id": self._next("PAY-"), "timestamp": at,
               "currency": currency, "amount_minor": amount,
               "invoice_refs": [inv["invoice_id"] for inv, _ in alloc] or None}
        self.payments.append(pay)
        pid = pay["payment_id"]
        if amount < 0:
            self._row(at, "CUSTOMER_ADVANCES", -amount, 0, currency, entity=entity, payment_id=pid)
            self._row(at, "CASH", 0, -amount, currency, entity=entity, payment_id=pid)
            return pay
        self._row(at, "CASH", amount, 0, currency, entity=entity, payment_id=pid)
        for inv, part in alloc:
            self._row(at, "AR", 0, part, currency, entity=entity, invoice_id=inv["invoice_id"], payment_id=pid)
        rest = amount - sum(part for _, part in alloc)
        if rest:
            self._row(at, "CUSTOMER_ADVANCES", 0, rest, currency, entity=entity, payment_id=pid)
        return pay

    def rate(self, day: str, src: str, rate: str, tgt: str | None = None) -> str:
        tgt = tgt or self.base
        self.fx.append({"date": day, "source_currency": src, "target_currency": tgt, "rate": rate,
                        "id": f"{day}:{src}:{tgt}"})
        return f"{day}:{src}:{tgt}"

    def books(self):
        tables = {"invoices": list(self.invoices), "payments": list(self.payments),
                  "ledger": list(self.ledger), "fx": list(self.fx)}
        return es.Books(self.manifest, self.policy, tables, set(tables))


def keyed(findings: list[dict]) -> set[tuple]:
    return {(f["type"], frozenset(f["affected_ids"]), f["claimed_amount_minor"], f["currency"],
             frozenset(f["evidence_ids"])) for f in findings}


def analyse(b: Builder) -> list[dict]:
    findings = es.analyse(b.books())
    assert validate_findings({"episode_id": "x", "findings": findings}) is None
    return findings


def row_of(b: Builder, inv: dict, account: str) -> str:
    return next(r["row_id"] for r in b.ledger if r["invoice_id"] == inv["invoice_id"]
                and r["payment_id"] is None and r["account"] == account)


# --- arithmetic -------------------------------------------------------------------------


def test_rounding_modes_on_a_tie():
    assert es.convert(2, Fraction("470.25"), 2, 2, "half_up") == 941
    assert es.convert(2, Fraction("470.25"), 2, 2, "half_even") == 940
    assert es.convert(-3, Fraction("0.5"), 2, 2, "half_up") == -2
    assert es.convert(1, Fraction(1), 2, 2, "banker") is None


def test_conversion_uses_exponents():
    assert es.convert(100, Fraction("3.1234"), 0, 2, "half_up") == 31234    # JPY -> KZT
    assert es.convert(1000, Fraction("3.2567"), 3, 2, "half_up") == 326     # KWD -> USD
    assert es.convert(100000, Fraction("470.25"), 2, 2, "half_up") == 47025000


def test_instants_and_offsets():
    assert es.parse_instant("2026-09-30T21:00:00+00:00") == es.parse_instant("2026-10-01T02:00:00+05:00")
    assert es.parse_instant("2026-09-30T21:00:00") is None  # naive times are refused
    assert es.parse_offset("-05:00").utcoffset(None).total_seconds() == -5 * 3600


# --- duplicate_invoice ---------------------------------------------------------------------


def test_duplicate_pair_is_found_and_lookalikes_are_not():
    b = Builder()
    a1 = b.invoice("A", 125000, "2026-08-03T10:00:00+05:00")
    a2 = b.invoice("A", 125000, "2026-08-10T10:00:00+05:00")
    b.invoice("B", 70000, "2026-08-03T10:00:00+05:00", period="2026-07")       # monthly twin: other month
    b.invoice("B", 70000, "2026-09-03T10:00:00+05:00", period="2026-08")
    b.invoice("C", 50000, "2026-08-03T10:00:00+05:00", po=None)                 # empty key field
    b.invoice("C", 50000, "2026-08-04T10:00:00+05:00", po=None)
    b.invoice("D", 9000, "2026-08-03T10:00:00+05:00", po="")                    # empty string counts as empty
    b.invoice("D", 9000, "2026-08-04T10:00:00+05:00", po="")
    for day in ("05", "06", "07"):                                              # three never occur: not decidable
        b.invoice("E", 3000, f"2026-08-{day}T10:00:00+05:00")
    b.invoice("F", 4000, "2026-08-05T10:00:00+05:00", currency="USD", revenue=1880000)  # same number, other currency
    b.invoice("F", 4000, "2026-08-06T10:00:00+05:00")
    b.rate("2026-08-05", "USD", "470.00")
    ids = [a1["doc_id"], a2["doc_id"]]
    assert keyed(analyse(b)) == keyed([es._finding("duplicate_invoice", ids, 125000, "KZT", ids, "")])


# --- unallocated_payment ---------------------------------------------------------------------


def test_missing_and_partial_allocation_are_found():
    b = Builder()
    i1 = b.invoice("A", 50000, "2026-08-01T10:00:00+05:00", po="PO-A")
    p1 = b.payment("A", 50000, "2026-08-10T10:00:00+05:00")
    i2 = b.invoice("B", 80000, "2026-08-01T10:00:00+05:00", po="PO-B")
    p2 = b.payment("B", 80000, "2026-08-11T10:00:00+05:00", alloc=[(i2, 60000)])
    expected = [
        es._finding("unallocated_payment", [p1["doc_id"], i1["doc_id"]], 50000, "KZT",
                    [p1["doc_id"], i1["doc_id"]], ""),
        es._finding("unallocated_payment", [p2["doc_id"], i2["doc_id"]], 20000, "KZT",
                    [p2["doc_id"], i2["doc_id"]], ""),
    ]
    assert keyed(analyse(b)) == keyed(expected)


def test_payment_lookalikes_are_not_reported():
    b = Builder()
    # genuine advance: the only invoice is issued after the payment
    b.payment("A", 30000, "2026-08-05T10:00:00+05:00")
    b.invoice("A", 30000, "2026-08-20T10:00:00+05:00", po="PO-A")
    # overpayment: the referenced invoice is closed, the rest is a real advance
    c = b.invoice("C", 40000, "2026-08-01T10:00:00+05:00", po="PO-C")
    b.payment("C", 45000, "2026-08-09T10:00:00+05:00", alloc=[(c, 40000)])
    # refund: paid twice, second payment to advances, money returned
    d = b.invoice("D", 10000, "2026-08-01T10:00:00+05:00", po="PO-D")
    b.payment("D", 10000, "2026-08-04T10:00:00+05:00", alloc=[(d, 10000)])
    b.payment("D", 10000, "2026-08-05T10:00:00+05:00")
    b.payment("D", -10000, "2026-08-12T10:00:00+05:00")
    # split payment: two payments, each fully allocated
    e = b.invoice("E", 90000, "2026-08-01T10:00:00+05:00", po="PO-E")
    b.payment("E", 50000, "2026-08-04T10:00:00+05:00", alloc=[(e, 50000)])
    b.payment("E", 40000, "2026-08-06T10:00:00+05:00", alloc=[(e, 40000)])
    # an open invoice in another currency than the payment
    b.invoice("G", 1000, "2026-08-01T10:00:00+05:00", currency="USD", po="PO-G", revenue=470000)
    b.rate("2026-08-01", "USD", "470.00")
    b.payment("G", 1000, "2026-08-04T10:00:00+05:00")
    # same instant written in another offset: invoice issued after the payment
    b.invoice("H", 7000, "2026-08-04T06:00:00+00:00", po="PO-H")   # 11:00 local
    b.payment("H", 7000, "2026-08-04T10:30:00+05:00")
    assert analyse(b) == []


def test_two_candidate_invoices_are_not_decidable_and_not_reported():
    b = Builder()
    b.invoice("F", 1000, "2026-08-01T10:00:00+05:00", po="PO-1")
    b.invoice("F", 1000, "2026-08-02T10:00:00+05:00", po="PO-2")
    b.payment("F", 1000, "2026-08-10T10:00:00+05:00")
    assert analyse(b) == []


def test_invoice_issued_at_the_same_instant_counts_as_no_later():
    b = Builder()
    inv = b.invoice("A", 1000, "2026-08-04T05:30:00+00:00", po="PO-A")
    pay = b.payment("A", 1000, "2026-08-04T10:30:00+05:00")
    assert [f["affected_ids"] for f in analyse(b)] == [[pay["doc_id"], inv["doc_id"]]]


# --- fx_error ---------------------------------------------------------------------------------


def fx_books() -> Builder:
    b = Builder()
    for day, rate in (("2026-08-13", "465.00"), ("2026-08-14", "470.25"), ("2026-08-17", "475.00")):
        b.rate(day, "USD", rate)
    b.rate("2026-08-14", "EUR", "512.40")
    b.rate("2026-08-14", "USD", "1.0890", tgt="EUR")  # another pair, never the right one
    return b


def test_fx_wrong_date_and_wrong_pair_are_found():
    b = fx_books()
    wrong_date = b.invoice("A", 100000, "2026-08-14T10:00:00+05:00", currency="USD", revenue=46500000)
    wrong_pair = b.invoice("B", 100000, "2026-08-14T11:00:00+05:00", currency="USD", revenue=51240000)
    findings = {f["affected_ids"][0]: f for f in analyse(b)}
    assert set(findings) == {wrong_date["doc_id"], wrong_pair["doc_id"]}
    f = findings[wrong_date["doc_id"]]
    assert (f["type"], f["claimed_amount_minor"], f["currency"]) == ("fx_error", 525000, "KZT")
    assert set(f["evidence_ids"]) == {wrong_date["doc_id"], row_of(b, wrong_date, "REVENUE"), "2026-08-14:USD:KZT"}
    assert findings[wrong_pair["doc_id"]]["claimed_amount_minor"] == 51240000 - 47025000


def test_fx_lookalikes_are_not_reported():
    b = fx_books()
    # 21:00 UTC on the 13th is the 14th locally: the rate of the 14th is right
    b.invoice("A", 100000, "2026-08-13T21:00:00+00:00", currency="USD", revenue=47025000)
    # Saturday the 15th has no rate: the previous available (14th) is right
    b.invoice("B", 100000, "2026-08-15T12:00:00+05:00", currency="USD", revenue=47025000)
    # below materiality (0.5 % of 47 025 000 is 235 125)
    b.invoice("C", 100000, "2026-08-14T12:00:00+05:00", currency="USD", revenue=47025000 + 200000)
    assert analyse(b) == []


def test_fx_previous_available_key_is_the_evidence():
    b = fx_books()
    inv = b.invoice("A", 100000, "2026-08-16T12:00:00+05:00", currency="USD", revenue=47500000)
    (f,) = analyse(b)
    assert "2026-08-14:USD:KZT" in f["evidence_ids"] and f["claimed_amount_minor"] == 475000
    assert f["affected_ids"] == [inv["doc_id"]]


def test_fx_materiality_threshold_is_inclusive():
    b = Builder()
    b.rate("2026-08-14", "USD", "1")
    b.manifest["currency_exponents"]["USD"] = 2
    b.invoice("A", 20000, "2026-08-14T12:00:00+05:00", currency="USD", revenue=20100)  # diff 100 = 0.5 %
    b.invoice("B", 20000, "2026-08-14T12:00:00+05:00", currency="USD", revenue=20099, po="PO-2")  # diff 99
    assert [f["claimed_amount_minor"] for f in analyse(b)] == [100]


def test_fx_half_even_policy_is_followed():
    b = Builder(rounding="half_even")
    b.rate("2026-08-14", "USD", "470.25")
    # 1000.02 USD -> 47 025 940.5 -> 47 025 940 (half even); a book at 47 025 941 differs by 1: not an error
    b.invoice("A", 100002, "2026-08-14T12:00:00+05:00", currency="USD", revenue=47025941)
    assert analyse(b) == []
    b.invoice("B", 100002, "2026-08-14T12:00:00+05:00", currency="USD", revenue=47025940 - 400000, po="PO-2")
    assert [f["claimed_amount_minor"] for f in analyse(b)] == [400000]


# --- cutoff_error ---------------------------------------------------------------------------


def test_cutoff_late_early_and_before_start_are_found():
    b = Builder()
    late = b.invoice("A", 1000, "2026-09-29T10:00:00+05:00", posted="2026-10-01T09:00:00+05:00", po="1")
    early = b.invoice("B", 2000, "2026-10-01T10:00:00+05:00", posted="2026-09-30T15:00:00+05:00", po="2")
    before = b.invoice("C", 3000, "2026-06-30T12:00:00+05:00", posted="2026-07-01T10:00:00+05:00", po="3")
    expected = [es._finding("cutoff_error", [inv["doc_id"]], inv["total_minor"], "KZT",
                            [inv["doc_id"], row_of(b, inv, "AR")], "") for inv in (late, early, before)]
    assert keyed(analyse(b)) == keyed(expected)


def test_cutoff_lookalikes_are_not_reported():
    b = Builder()
    # posted two days later, same side of the bound
    b.invoice("A", 1000, "2026-09-26T10:00:00+05:00", posted="2026-09-28T16:00:00+05:00", po="1")
    # 20:00 UTC on 30 Sep is 1 Oct locally: issued after the period, posted after it too
    b.invoice("B", 1000, "2026-09-30T20:00:00+00:00", posted="2026-10-01T10:00:00+05:00", po="2")
    # local 23:30 on 30 Sep; posting written in UTC is 23:45 locally: both inside
    b.invoice("C", 1000, "2026-09-30T23:30:00+05:00", posted="2026-09-30T18:45:00+00:00", po="3")
    # posted exactly on the inclusive end bound
    b.invoice("D", 1000, "2026-09-30T12:00:00+05:00", posted="2026-09-30T23:59:59+05:00", po="4")
    # issued exactly on the start bound, posted a day later
    b.invoice("E", 1000, "2026-07-01T00:00:00+05:00", posted="2026-07-02T09:00:00+05:00", po="5")
    assert analyse(b) == []


def test_cutoff_and_fx_on_a_foreign_invoice():
    b = fx_books()
    inv = b.invoice("A", 100000, "2026-08-14T10:00:00+05:00", currency="USD", revenue=47025000,
                    posted="2026-10-01T10:00:00+05:00")
    (f,) = analyse(b)
    assert (f["type"], f["claimed_amount_minor"], f["currency"]) == ("cutoff_error", 100000, "USD")
    assert set(f["evidence_ids"]) == {inv["doc_id"], row_of(b, inv, "AR")}


# --- opaque IDs and order ----------------------------------------------------------------


def mixed_books() -> Builder:
    b = fx_books()
    a1 = b.invoice("A", 125000, "2026-08-03T10:00:00+05:00")
    b.invoice("A", 125000, "2026-08-10T10:00:00+05:00")
    b.payment("A", 125000, "2026-08-11T10:00:00+05:00", alloc=[(a1, 125000)])
    i1 = b.invoice("B", 50000, "2026-08-01T10:00:00+05:00", po="PO-B")
    b.payment("B", 50000, "2026-08-10T10:00:00+05:00")
    b.invoice("C", 100000, "2026-08-14T10:00:00+05:00", currency="USD", revenue=46500000)
    b.invoice("D", 1000, "2026-09-29T10:00:00+05:00", posted="2026-10-01T09:00:00+05:00", po="D")
    b.payment("E", 30000, "2026-08-05T10:00:00+05:00")
    b.invoice("E", 30000, "2026-08-20T10:00:00+05:00", po="PO-E")
    assert i1
    return b


AWKWARD = ("1", "01", " 1", "1 ", "a", "A", "é", "é", "x,y", 'q"q', "#7", "a|b", " ", "\t", "ID")


def relabel(b: Builder, seed: int) -> tuple[Builder, dict[str, str]]:
    """Every ID replaced by an awkward string (spaces mapped apart), every table shuffled."""
    rng = random.Random(seed)
    spaces = {"row_id": [], "doc_id": [], "invoice_id": [], "payment_id": [], "entity_id": []}
    for table in (b.ledger, b.invoices, b.payments):
        for rec in table:
            for space in spaces:
                if rec.get(space) and rec[space] not in spaces[space]:
                    spaces[space].append(rec[space])
    mapping: dict[str, dict[str, str]] = {}
    back: dict[str, str] = {}
    counter = 0  # one counter over all spaces keeps new IDs unique, so evidence can be mapped back
    for space, old in sorted(spaces.items()):
        new = [f"{AWKWARD[j % len(AWKWARD)]}{j // len(AWKWARD) or ''}" for j in range(counter, counter + len(old))]
        counter += len(old)
        rng.shuffle(new)
        mapping[space] = dict(zip(old, new))
        if space in ("row_id", "doc_id"):
            back.update({v: o for o, v in mapping[space].items()})

    def m(rec: dict) -> dict:
        out = dict(rec)
        for space, table in mapping.items():
            if out.get(space) in table:
                out[space] = table[out[space]]
        if out.get("invoice_refs"):
            out["invoice_refs"] = [mapping["invoice_id"][r] for r in out["invoice_refs"]]
        return out

    c = Builder()
    c.manifest, c.policy = b.manifest, b.policy
    c.ledger = [m(r) for r in b.ledger]
    c.invoices = [m(r) for r in b.invoices]
    c.payments = [m(r) for r in b.payments]
    c.fx = list(b.fx)
    for table in (c.ledger, c.invoices, c.payments, c.fx):
        rng.shuffle(table)
    return c, back


@pytest.mark.parametrize("seed", range(5))
def test_relabelled_and_shuffled_books_give_the_same_findings(seed):
    b = mixed_books()
    original = analyse(b)
    assert sorted(f["type"] for f in original) == ["cutoff_error", "duplicate_invoice", "fx_error",
                                                   "unallocated_payment"]
    c, back = relabel(b, seed)
    restored = [{**f, "affected_ids": [back.get(i, i) for i in f["affected_ids"]],
                 "evidence_ids": [back.get(i, i) for i in f["evidence_ids"]]} for f in analyse(c)]
    assert keyed(restored) == keyed(original)


# --- the act() protocol --------------------------------------------------------------------


class FakeEnv:
    """Serves the builder's tables page by page with opaque cursors, as the real tools do."""

    def __init__(self, b: Builder, fail_first_fx: bool = False) -> None:
        self.tables = {"ledger": b.ledger, "invoices": b.invoices, "payments": b.payments, "fx": b.fx}
        self.b = b
        self.calls: list[dict] = []
        self.fail_first_fx = fail_first_fx

    def first(self) -> dict:
        return {"task": "t", "manifest": self.b.manifest, "policy": self.b.policy, "remaining_steps": 100}

    def step(self, action: dict) -> dict:
        self.calls.append(action)
        remaining = 100 - len(self.calls)
        args = action["args"]
        if action["name"] != "list_records":
            return {"result": {"accepted": True}, "error": None, "remaining_steps": remaining}
        if self.fail_first_fx and args["table"] == "fx":
            self.fail_first_fx = False
            return {"result": None, "error": {"code": "bad_arguments", "message": "m"}, "remaining_steps": remaining}
        rows = self.tables[args["table"]]
        offset = 0 if args["cursor"] is None else int(args["cursor"].split("@")[1])
        end = offset + args["limit"]
        cursor = f"opaque@{end}" if end < len(rows) else None
        return {"result": {"records": rows[offset:end], "next_cursor": cursor}, "error": None,
                "remaining_steps": remaining}


def run(env: FakeEnv, seed: int = 0) -> list[dict]:
    policy = es.make_policy(seed)
    action = policy.act(env.first())
    while True:
        obs = env.step(action)
        if action["name"] == "submit":
            return env.calls
        action = policy.act(obs)


def big_books(n: int = 130) -> Builder:
    b = mixed_books()
    for k in range(n):
        inv = b.invoice(f"Z{k}", 1000 + k, "2026-08-03T10:00:00+05:00", po=f"Z{k}")
        b.payment(f"Z{k}", 1000 + k, "2026-08-04T10:00:00+05:00", alloc=[(inv, 1000 + k)])
    return b


def test_policy_reads_every_page_once_then_submits_the_analysis():
    b = big_books()
    calls = run(FakeEnv(b))
    pages = calls[:-1]
    assert all(c["name"] == "list_records" and c["args"]["limit"] == 50 for c in pages)
    per_table = {t: sum(1 for c in pages if c["args"]["table"] == t) for t in ("invoices", "payments", "ledger", "fx")}
    assert per_table == {t: -(-len(rows) // 50) for t, rows in
                         (("invoices", b.invoices), ("payments", b.payments), ("ledger", b.ledger), ("fx", b.fx))}
    answer = calls[-1]["args"]["findings"]
    assert validate_findings(answer) is None and answer["episode_id"] == "ep-test"
    assert keyed(answer["findings"]) == keyed(analyse(b))


def test_policy_is_deterministic():
    assert run(FakeEnv(big_books()), seed=1) == run(FakeEnv(big_books()), seed=2)


def test_fx_is_not_read_when_every_invoice_is_in_the_base_currency():
    b = Builder()
    b.invoice("A", 1000, "2026-08-03T10:00:00+05:00")
    b.rate("2026-08-03", "USD", "470")
    calls = run(FakeEnv(b))
    assert [c["args"]["table"] for c in calls[:-1]] == ["invoices", "payments", "ledger"]


def test_an_unreadable_table_disables_only_its_rule():
    b = mixed_books()
    calls = run(FakeEnv(b, fail_first_fx=True))
    kinds = sorted(f["type"] for f in calls[-1]["args"]["findings"]["findings"])
    assert kinds == ["cutoff_error", "duplicate_invoice", "unallocated_payment"]


def test_policy_submits_before_the_budget_runs_out():
    b = big_books(2000)  # more pages than the budget allows
    calls = run(FakeEnv(b))
    assert len(calls) == 100 and calls[-1]["name"] == "submit"


# --- isolation of the module ---------------------------------------------------------------


def test_module_imports_only_the_standard_library_and_uses_no_float():
    tree = ast.parse(SOLVER_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        assert not (isinstance(node, ast.Constant) and isinstance(node.value, float))
        assert not (isinstance(node, ast.Name) and node.id in ("float", "open", "eval", "exec", "__import__"))
    assert imported <= set(sys.stdlib_module_names) and "cookedbooks" not in imported


# --- the task text carries the whole rule book ---------------------------------------------
# The first observation is the only place an outside agent learns the rules, so every
# definition the solver relies on must be stated there. Each phrase below pins one of them.

TASK_DEFINITIONS = {
    "instants": "compared as instants",
    "local date": "calendar date after conversion to manifest.utc_offset",
    "inclusive bounds": "period_start <= time <= period_end (both bounds inclusive)",
    "opaque IDs": "IDs are opaque strings compared exactly",
    "invoice posting": "every ledger row with its invoice_id and an empty payment_id",
    "posting time": "these rows share one timestamp, the posting time",
    "AR line": "Its AR line is the Dr AR row for total_minor in the invoice currency",
    "REVENUE line": "its REVENUE line is the Cr REVENUE row in the base currency",
    "payment rows": "A payment's rows are every ledger row with its payment_id",
    "payment customer": "The payment's customer is the entity_id on these rows",
    "incoming only": "A payment with amount_minor < 0 is a refund to the customer and is never an error",
    "allocated": "Allocated to an invoice = the sum of all Cr AR rows with its invoice_id",
    "open invoice": "An invoice is open if total_minor minus allocated is above 0",
    "remainder": "unallocated remainder of an incoming payment = amount_minor minus the sum of its own Cr AR rows",
    "background rows": "background entries and are never part of an error",
    "duplicate key": "(entity_id, po_ref, service_period, currency, total_minor)",
    "empty key field": "An invoice with any of these fields null or empty takes no part",
    "no later than": "issued_at <= payment timestamp",
    "one matching invoice": "There is never more than one such invoice",
    "fx pair": "source_currency = the invoice currency and target_currency = the base currency",
    "rate date": "on the local date of issued_at",
    "previous available": "the record of the same pair with the latest earlier date",
    "fallback key": "the earlier date's key when the fallback applies",
    "formula": "base_minor = round(src_minor * rate * 10^(exp_base - exp_src))",
    "exponents": "exp from manifest.currency_exponents",
    "rounding": "half_up: halves away from zero; half_even: halves to the even integer",
    "materiality minor": "at least policy.fx_materiality.min_minor AND",
    "materiality relative": "at least policy.fx_materiality.min_relative times the correct amount (both inclusive)",
    "fx evidence": "the invoice doc_id, the row_id of its REVENUE line and the key of the fx record",
    "cutoff sides": "before period_start, inside the period or after period_end",
    "cutoff evidence": "the invoice doc_id and the row_id of its AR line, or the invoice doc_id and the row_id of its REVENUE line",
    "unallocated ids": "the payment doc_id and the invoice doc_id",
    "unallocated amount": "the payment's unallocated remainder, in the payment currency",
    "fx amount": "Amount: the difference, in the base currency",
    "duplicate amount": "Amount: total_minor in the invoice currency",
    "look-alikes": "do not report them",
    "overpayment": "an overpayment (the referenced invoice is fully paid",
    "monthly twin": "the same amount but another service_period",
    "two combinations": "can carry both an fx_error and an unallocated_payment",
    "duplicate plus fx": "one invoice of a duplicate_invoice pair can also carry an fx_error",
    "doc_id only": "identified by doc_id only, never by invoice_id or payment_id",
    "extra evidence": "extra IDs are allowed if they exist",
    "tolerance": "within policy.amount_tolerance_minor[type] of the correct one",
    "page size": "A page holds at most 50 records",
    "size caps": "at most 2000 ledger rows and at most 200 documents",
    "limits": "at most 30 findings, at most 10 evidence_ids per finding (repeats count)",
    "explanation": "explanation at most 1000 characters",
    "budget": "100 actions per episode including submit",
    "time": "Time: 60 seconds from reset",
}


def task_text_of_env() -> str:
    from cookedbooks.env.core import task_text

    return task_text(60.0)


@pytest.mark.parametrize("name", sorted(TASK_DEFINITIONS))
def test_task_text_states_definition(name):
    assert TASK_DEFINITIONS[name] in task_text_of_env(), name


def test_task_text_names_every_manifest_and_policy_field_the_solver_reads():
    text = task_text_of_env()
    for field_name in ("base_currency", "period_start", "period_end", "utc_offset", "currency_exponents",
                       "episode_id", "duplicate_key", "allocation_rule", "fx_direction", "fx_rate_date",
                       "fx_missing_rate", "fx_materiality", "min_minor", "min_relative", "policy.rounding",
                       "cutoff_basis", "amount_tolerance_minor"):
        assert field_name in text, field_name


def test_task_text_reveals_nothing_hidden():
    lowered = task_text_of_env().lower()
    for banned in ("truth", "provenance", "seed", "reward", "issue_id", "evidence_sets", "oracle", "grader",
                   "score", "hidden", "inject", "generator", ".json", ".csv", ".jsonl", "f1", "precision",
                   "recall", "0.002"):
        assert banned not in lowered, banned


def test_task_text_is_plain_compact_text():
    text = task_text_of_env()
    assert "\n|" not in text and "|---" not in text and "```" not in text and "**" not in text  # no Markdown
    assert 3000 < len(text) < 9000
    for number in range(1, 8):
        assert f"\n{number}. " in text

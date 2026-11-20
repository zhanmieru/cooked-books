"""Hand-written, correctly posted books for verifier tests (DESIGN.md section 3).

`Book` produces postings exactly as the design table says, so a test only
states what differs. It computes FX amounts on its own (datetime +
Decimal.quantize), not with the verifier's code, so the two can disagree.

`Injector` records every change it makes to a Book as a provenance change
with full before/after records, grouped into injections.

Row IDs are plain counters ("1", "2", ...) and therefore collide with the
numeric doc IDs most tests use: nothing may tell ID spaces apart by shape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

from tests.helpers.factory import default_policy, write_episode

TABLE_FILE = {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl", "fx": "fx.csv"}
KEY = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id"}


@dataclass
class InvoiceHandle:
    doc_id: str
    invoice_id: str
    entity_id: str
    currency: str
    total: int
    rows: dict[str, str]          # ar, revenue, and for foreign invoices fx_src, fx_base
    fx_key: str | None = None
    base_amount: int | None = None

    @property
    def posting_row_ids(self) -> list[str]:
        return list(self.rows.values())


@dataclass
class PaymentHandle:
    doc_id: str
    payment_id: str
    entity_id: str
    currency: str
    amount: int
    cash_row: str
    ar_rows: dict[str, str] = field(default_factory=dict)   # invoice_id -> row_id
    advances_row: str | None = None


def _offset(text: str) -> timezone:
    if text == "Z":
        return timezone.utc
    sign = -1 if text[0] == "-" else 1
    hours, minutes = int(text[1:3]), int(text[4:6])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


class Book:
    def __init__(self, *, base: str = "KZT", exponents: dict[str, int] | None = None,
                 utc_offset: str = "+05:00", period_start: str = "2026-07-01T00:00:00+05:00",
                 period_end: str = "2026-09-30T23:59:59+05:00", rounding: str = "half_up",
                 episode_id: str = "ep-v") -> None:
        self.base = base
        self.exponents = dict(exponents or {"KZT": 2, "USD": 2})
        self.utc_offset = utc_offset
        self.period_start, self.period_end = period_start, period_end
        self.rounding = rounding
        self.episode_id = episode_id
        self.tables: dict[str, list[dict]] = {"ledger": [], "invoices": [], "payments": [], "fx": []}
        self._row = 0

    # --- helpers -----------------------------------------------------------

    def next_row_id(self) -> str:
        self._row += 1
        return str(self._row)

    def _put(self, table: str, record: dict, inj: "Injector | None") -> None:
        if inj is None:
            self.tables[table].append(record)
        else:
            inj.add(table, record)

    def _row_record(self, ts, account, debit, credit, currency, entity=None, invoice_id=None, payment_id=None):
        return {"row_id": self.next_row_id(), "timestamp": ts, "account": account, "debit_minor": debit,
                "credit_minor": credit, "currency": currency, "entity_id": entity,
                "invoice_id": invoice_id, "payment_id": payment_id}

    def local_date(self, ts: str) -> str:
        return datetime.fromisoformat(ts).astimezone(_offset(self.utc_offset)).date().isoformat()

    def rate_row(self, currency: str, ts: str) -> dict:
        """Policy lookup done independently: exact local date, else the latest earlier one."""
        day = self.local_date(ts)
        rows = [r for r in self.tables["fx"] if r["source_currency"] == currency
                and r["target_currency"] == self.base and r["date"] <= day]
        return max(rows, key=lambda r: r["date"])

    def convert(self, total: int, currency: str, rate: str) -> int:
        mode = ROUND_HALF_UP if self.rounding == "half_up" else ROUND_HALF_EVEN
        scaled = Decimal(total) * Decimal(rate) * Decimal(10) ** (self.exponents[self.base] - self.exponents[currency])
        return int(scaled.quantize(Decimal(1), rounding=mode))

    # --- records -------------------------------------------------------------

    def add_fx(self, date: str, source: str, rate: str, target: str | None = None) -> str:
        target = target or self.base
        self.tables["fx"].append({"date": date, "source_currency": source, "target_currency": target, "rate": rate})
        return f"{date}:{source}:{target}"

    def invoice(self, doc_id: str, invoice_id: str, entity: str, issued_at: str, total: int, *,
                currency: str | None = None, po_ref: str | None = None, service_period: str | None = None,
                posted_at: str | None = None, base_amount: int | None = None,
                inj: "Injector | None" = None) -> InvoiceHandle:
        currency = currency or self.base
        posted_at = posted_at or issued_at
        record = {
            "doc_id": doc_id, "invoice_id": invoice_id, "entity_id": entity, "issued_at": issued_at,
            "due_at": issued_at, "currency": currency, "total_minor": total,
            "line_items": [{"description": "Service", "quantity": 1, "unit_price_minor": total - total // 3,
                            "amount_minor": total - total // 3},
                           {"description": "Support", "quantity": 1, "unit_price_minor": total // 3,
                            "amount_minor": total // 3}],
            "po_ref": po_ref, "service_period": service_period,
        }
        self._put("invoices", record, inj)
        rows: dict[str, str] = {}
        fx_key = None
        if currency == self.base:
            base = total
            ar = self._row_record(posted_at, "AR", total, 0, currency, entity, invoice_id)
            rev = self._row_record(posted_at, "REVENUE", 0, total, self.base, entity, invoice_id)
            records = [("ar", ar), ("revenue", rev)]
        else:
            rate = self.rate_row(currency, issued_at)
            fx_key = f"{rate['date']}:{currency}:{self.base}"
            base = self.convert(total, currency, rate["rate"]) if base_amount is None else base_amount
            records = [
                ("ar", self._row_record(posted_at, "AR", total, 0, currency, entity, invoice_id)),
                ("fx_src", self._row_record(posted_at, "FX_CLEARING", 0, total, currency, entity, invoice_id)),
                ("fx_base", self._row_record(posted_at, "FX_CLEARING", base, 0, self.base, entity, invoice_id)),
                ("revenue", self._row_record(posted_at, "REVENUE", 0, base, self.base, entity, invoice_id)),
            ]
        for name, row in records:
            self._put("ledger", row, inj)
            rows[name] = row["row_id"]
        return InvoiceHandle(doc_id, invoice_id, entity, currency, total, rows, fx_key, base)

    def payment(self, doc_id: str, payment_id: str, entity: str, ts: str, amount: int, *,
                currency: str | None = None, allocations: dict[str, int] | None = None,
                inj: "Injector | None" = None) -> PaymentHandle:
        """Incoming payment; allocations maps invoice_id -> amount (None = no refs)."""
        currency = currency or self.base
        refs = list(allocations) if allocations else None
        self._put("payments", {"doc_id": doc_id, "payment_id": payment_id, "timestamp": ts, "currency": currency,
                               "amount_minor": amount, "invoice_refs": refs}, inj)
        cash = self._row_record(ts, "CASH", amount, 0, currency, entity, None, payment_id)
        self._put("ledger", cash, inj)
        handle = PaymentHandle(doc_id, payment_id, entity, currency, amount, cash["row_id"])
        for iid, part in (allocations or {}).items():
            row = self._row_record(ts, "AR", 0, part, currency, entity, iid, payment_id)
            self._put("ledger", row, inj)
            handle.ar_rows[iid] = row["row_id"]
        rest = amount - sum((allocations or {}).values())
        if rest:
            row = self._row_record(ts, "CUSTOMER_ADVANCES", 0, rest, currency, entity, None, payment_id)
            self._put("ledger", row, inj)
            handle.advances_row = row["row_id"]
        return handle

    def refund(self, doc_id: str, payment_id: str, entity: str, ts: str, amount: int, *,
               currency: str | None = None) -> PaymentHandle:
        """Refund of `amount` (> 0) to the client: a payment with a negative amount."""
        currency = currency or self.base
        self.tables["payments"].append({"doc_id": doc_id, "payment_id": payment_id, "timestamp": ts,
                                        "currency": currency, "amount_minor": -amount, "invoice_refs": None})
        dr = self._row_record(ts, "CUSTOMER_ADVANCES", amount, 0, currency, entity, None, payment_id)
        cr = self._row_record(ts, "CASH", 0, amount, currency, entity, None, payment_id)
        self.tables["ledger"] += [dr, cr]
        return PaymentHandle(doc_id, payment_id, entity, currency, -amount, cr["row_id"])

    def background(self, ts: str, account: str, amount: int) -> None:
        self.tables["ledger"].append(self._row_record(ts, account, amount, 0, self.base))
        self.tables["ledger"].append(self._row_record(ts, "CASH", 0, amount, self.base))

    def record(self, table: str, key: str) -> dict:
        return next(r for r in self.tables[table] if r[KEY[table]] == key)

    # --- output --------------------------------------------------------------

    def manifest(self) -> dict:
        return {
            "schema_version": "1.0", "episode_id": self.episode_id, "base_currency": self.base,
            "period_start": self.period_start, "period_end": self.period_end, "timezone": "Test/Fixed",
            "utc_offset": self.utc_offset, "currency_exponents": dict(self.exponents),
            "rounding": {"mode": self.rounding, "unit": "minor"},
        }

    def policy(self) -> dict:
        policy = default_policy()
        policy["rounding"] = self.rounding
        return policy

    def write(self, directory, *, issues: list[dict] | None = None, injector: "Injector | None" = None,
              provenance: dict | None = None, manifest: dict | None = None, policy: dict | None = None) -> Path:
        if provenance is None:
            provenance = (injector or Injector(self)).provenance()
        return write_episode(directory, manifest=manifest or self.manifest(), policy=policy or self.policy(),
                             issues=[] if issues is None else issues, provenance=provenance,
                             **copy.deepcopy(self.tables))


class Injector:
    """Make changes to a Book and record them as provenance injections."""

    def __init__(self, book: Book) -> None:
        self.book = book
        self.injections: list[dict] = []
        self._changes: list[dict] = []

    def _change(self, table, op, key, before, after) -> None:
        self._changes.append({"file": TABLE_FILE[table], "op": op, "key": key,
                              "before": copy.deepcopy(before), "after": copy.deepcopy(after)})

    def add(self, table: str, record: dict) -> None:
        self.book.tables[table].append(record)
        self._change(table, "add", record[KEY[table]], None, record)

    def modify(self, table: str, key: str, **fields) -> None:
        record = self.book.record(table, key)
        before = copy.deepcopy(record)
        record.update(fields)
        self._change(table, "modify", key, before, record)

    def remove(self, table: str, key: str) -> None:
        record = self.book.record(table, key)
        self.book.tables[table].remove(record)
        self._change(table, "remove", key, record, None)

    def commit(self, issue_ids: list[str], kind: str, effects: list[dict], *, scenario: str | None = None) -> None:
        self.injections.append({"issue_ids": list(issue_ids), "kind": kind, "scenario": scenario,
                                "description": kind, "changes": self._changes, "effects": effects})
        self._changes = []

    def provenance(self) -> dict:
        return {"schema_version": "1.0", "episode_id": self.book.episode_id, "generator_version": "test",
                "seed": 0, "attempt": 0, "team_config": {}, "injections": self.injections, "distractors": []}


# --- injection recipes ---------------------------------------------------------


def inject_missing(inj: Injector, pay: PaymentHandle) -> None:
    """unallocated_payment/missing: refs -> null, Cr AR rows gone, all of it in advances."""
    book = inj.book
    inj.modify("payments", pay.doc_id, invoice_refs=None)
    for row_id in pay.ar_rows.values():
        inj.remove("ledger", row_id)
    if pay.advances_row is None:
        row = book._row_record(book.record("ledger", pay.cash_row)["timestamp"], "CUSTOMER_ADVANCES", 0,
                               pay.amount, pay.currency, pay.entity_id, None, pay.payment_id)
        inj.add("ledger", row)
    else:
        inj.modify("ledger", pay.advances_row, credit_minor=pay.amount)


def inject_partial(inj: Injector, pay: PaymentHandle, invoice_id: str, keep: int) -> int:
    """unallocated_payment/partial: Cr AR on `invoice_id` reduced to `keep`. Returns the moved amount."""
    book = inj.book
    row = book.record("ledger", pay.ar_rows[invoice_id])
    moved = row["credit_minor"] - keep
    inj.modify("ledger", row["row_id"], credit_minor=keep)
    if pay.advances_row is None:
        adv = book._row_record(row["timestamp"], "CUSTOMER_ADVANCES", 0, moved, pay.currency, pay.entity_id,
                               None, pay.payment_id)
        inj.add("ledger", adv)
    else:
        old = book.record("ledger", pay.advances_row)["credit_minor"]
        inj.modify("ledger", pay.advances_row, credit_minor=old + moved)
    return moved


def inject_fx(inj: Injector, inv: InvoiceHandle, new_base: int) -> int:
    """fx_error: base amount of the posting changed to `new_base`. Returns |difference|."""
    inj.modify("ledger", inv.rows["fx_base"], debit_minor=new_base)
    inj.modify("ledger", inv.rows["revenue"], credit_minor=new_base)
    return abs(new_base - inv.base_amount)


def inject_cutoff(inj: Injector, inv: InvoiceHandle, new_ts: str) -> None:
    """cutoff_error: every posting row moved to `new_ts`; issued_at unchanged."""
    for row_id in inv.posting_row_ids:
        inj.modify("ledger", row_id, timestamp=new_ts)

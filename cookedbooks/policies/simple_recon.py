"""simple_recon: the naive reconciliation an inexperienced checker would write.

It pages through invoices, payments and the ledger (every page, 50 records
at a time), then reports:

- duplicate_invoice: every pair of invoices of one customer with equal
  currency and `total_minor`. It ignores `po_ref` and `service_period`, so a
  legitimate monthly invoice for the same amount (look-alike
  `recurring_amount`) is reported too.
- unallocated_payment: an incoming payment without an invoice reference is
  reported against an invoice of the same currency and amount (same
  customer first), for the full amount; a payment with a reference whose
  `Cr AR` lines leave a remainder is reported against the referenced invoice
  for that remainder. It never checks whether the invoice is still open or
  was issued before the payment, so a refunded double payment or an honest
  overpayment fool it.
- cutoff_error: an invoice whose `issued_at` and posting fall on different
  sides of a period bound when the timestamps are read as plain date
  strings. Offsets are ignored, so a record written in UTC near the bound
  fools it, and a real crossing that shows only in local time is missed.

It makes no use of time zones, the FX table or the period policy and never
reports fx_error. Deterministic; `seed` is unused.
"""

from __future__ import annotations

from typing import Any

from cookedbooks.policies.base import (
    MAX_FINDINGS, Action, Observation, Pager, Record, episode_id_of, finding, is_int, is_reset, submit,
)

TABLES = ("invoices", "payments", "ledger")


class SimpleRecon:
    def __init__(self) -> None:
        self._pager = Pager(TABLES)
        self._episode_id = ""
        self._period = ("", "")

    def act(self, observation: Observation) -> Action:
        if is_reset(observation):
            self._pager = Pager(TABLES)
            self._episode_id = episode_id_of(observation)
            manifest = observation.get("manifest") or {}
            start, end = manifest.get("period_start"), manifest.get("period_end")
            self._period = (start[:10] if isinstance(start, str) else "", end[:10] if isinstance(end, str) else "")
        action = self._pager.next_action(observation)
        if action is not None:
            return action
        return submit(self._episode_id, self.findings()[:MAX_FINDINGS])

    def findings(self) -> list[dict]:
        records = self._pager.records
        invoices, payments, ledger = records["invoices"], records["payments"], records["ledger"]
        return (_duplicates(invoices) + _unallocated(invoices, payments, ledger)
                + _cutoffs(invoices, ledger, self._period))


def _s(record: Record, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _n(record: Record, key: str) -> int:
    value = record.get(key)
    return value if is_int(value) else 0


def _duplicates(invoices: list[Record]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[Record]] = {}
    for inv in invoices:
        groups.setdefault((_s(inv, "entity_id"), _s(inv, "currency"), _n(inv, "total_minor")), []).append(inv)
    out = []
    for (_, currency, total), group in groups.items():
        for other in group[1:]:
            ids = [_s(group[0], "doc_id"), _s(other, "doc_id")]
            out.append(finding("duplicate_invoice", ids, total, currency, ids,
                               "Same customer and same total invoiced twice."))
    return out


def _unallocated(invoices: list[Record], payments: list[Record], ledger: list[Record]) -> list[dict]:
    allocated: dict[str, int] = {}
    customer: dict[str, str] = {}
    for row in ledger:
        payment_id = _s(row, "payment_id")
        if not payment_id:
            continue
        if _s(row, "entity_id") and payment_id not in customer:
            customer[payment_id] = _s(row, "entity_id")
        if _s(row, "account") == "AR" and _n(row, "credit_minor") > 0:
            allocated[payment_id] = allocated.get(payment_id, 0) + _n(row, "credit_minor")
    by_invoice_id = {_s(inv, "invoice_id"): inv for inv in invoices}
    out = []
    for pay in payments:
        amount, currency = _n(pay, "amount_minor"), _s(pay, "currency")
        if amount <= 0:
            continue
        refs = pay.get("invoice_refs")
        target: Record | None = None
        claimed = 0
        if not refs:
            same = [inv for inv in invoices if _s(inv, "currency") == currency and _n(inv, "total_minor") == amount]
            mine = [inv for inv in same if _s(inv, "entity_id") == customer.get(_s(pay, "payment_id"))]
            target = (mine or same or [None])[0]
            claimed = amount
        elif isinstance(refs, list):
            remainder = amount - allocated.get(_s(pay, "payment_id"), 0)
            ref = refs[0] if refs and isinstance(refs[0], str) else ""
            if remainder > 0 and ref in by_invoice_id:
                target, claimed = by_invoice_id[ref], remainder
        if target is not None:
            ids = [_s(pay, "doc_id"), _s(target, "doc_id")]
            out.append(finding("unallocated_payment", ids, claimed, currency, ids,
                               "Payment not (fully) applied to an invoice of the same amount."))
    return out


def _cutoffs(invoices: list[Record], ledger: list[Record], period: tuple[str, str]) -> list[dict]:
    start, end = period
    if not start or not end:
        return []
    posting: dict[str, Record] = {}
    for row in ledger:
        if (_s(row, "account") == "AR" and _n(row, "debit_minor") > 0 and _s(row, "invoice_id")
                and not _s(row, "payment_id")):
            posting.setdefault(_s(row, "invoice_id"), row)

    def inside(text: str) -> bool:
        day = text[:10]  # the date as written, offset ignored
        return start <= day <= end

    out = []
    for inv in invoices:
        row = posting.get(_s(inv, "invoice_id"))
        if row is None:
            continue
        if inside(_s(inv, "issued_at")) != inside(_s(row, "timestamp")):
            out.append(finding("cutoff_error", [_s(inv, "doc_id")], _n(inv, "total_minor"), _s(inv, "currency"),
                               [_s(inv, "doc_id"), _s(row, "row_id")], "Invoice and posting in different periods."))
    return out


def make_policy(seed: int, **options: Any) -> SimpleRecon:
    return SimpleRecon()

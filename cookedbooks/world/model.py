"""The in-memory world: documents and ledger rows with internal references.

Internal refs (`I12`, `P3`, `L407`, `C5`) are stable keys used only inside the
generator. Public IDs are assigned at export time (`ids.py`), after every
record, injected ones included, exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from cookedbooks.world.fx import FxTable, convert
from cookedbooks.world.timeline import Timeline

TABLES = ("ledger", "invoices", "payments")


@dataclass(frozen=True)
class LineItem:
    description: str
    quantity: int
    unit_price_minor: int
    amount_minor: int


@dataclass(frozen=True)
class Invoice:
    ref: str
    customer: str
    issued_utc: int
    issued_offset: int  # minutes; the offset `issued_at` is written with
    due_utc: int
    due_offset: int
    currency: str
    line_items: tuple[LineItem, ...]
    total_minor: int
    po_ref: str | None
    service_period: str | None


@dataclass(frozen=True)
class Payment:
    ref: str
    customer: str  # internal only: payments.jsonl has no entity field
    ts_utc: int
    ts_offset: int
    currency: str
    amount_minor: int  # > 0 incoming, < 0 refund to the customer
    invoice_refs: tuple[str, ...] | None  # internal invoice refs


@dataclass(frozen=True)
class LedgerRow:
    ref: str
    posting: str  # rows of one posting share it (used for ordering only)
    ts_utc: int
    ts_offset: int
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    customer: str | None
    invoice: str | None  # internal invoice ref
    payment: str | None  # internal payment ref


@dataclass(frozen=True)
class Distractor:
    kind: str
    docs: tuple[tuple[str, str], ...]  # ("invoice" | "payment", internal ref)


@dataclass(frozen=True)
class ReservedTarget:
    """A clean invoice (and its payments, if paid) built for one injector.

    `payment` is the last payment of the invoice (the only one for paid "full",
    the second for paid "split"); `payments` lists all of them in time order.
    """

    customer: str
    invoice: str
    payment: str | None
    payments: tuple[str, ...] = ()


@dataclass
class World:
    timeline: Timeline
    fx: FxTable
    base_currency: str
    foreign_currencies: tuple[str, ...]
    reference_currency: str | None  # extra fx pair, never used by documents
    rounding: str
    invoices: dict[str, Invoice] = field(default_factory=dict)
    payments: dict[str, Payment] = field(default_factory=dict)
    ledger: dict[str, LedgerRow] = field(default_factory=dict)
    distractors: list[Distractor] = field(default_factory=list)
    reserved: dict[int, tuple[ReservedTarget, ...]] = field(default_factory=dict)
    used_po_refs: set[str] = field(default_factory=set)
    used_totals: set[tuple[str, str, int]] = field(default_factory=set)
    _counters: dict[str, int] = field(default_factory=dict)

    # --- bookkeeping -----------------------------------------------------------

    def new_ref(self, prefix: str) -> str:
        """A fresh internal ref: 'I' invoice, 'P' payment, 'L' ledger row, 'T' posting, 'C' customer."""
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    def table(self, name: str) -> dict:
        if name not in TABLES:
            raise KeyError(name)
        return getattr(self, name)

    @property
    def currencies(self) -> tuple[str, ...]:
        return (self.base_currency, *self.foreign_currencies)

    def document_count(self) -> int:
        return len(self.invoices) + len(self.payments)

    def row_count(self) -> int:
        return len(self.ledger)

    # --- queries ----------------------------------------------------------------

    def invoice_rows(self, invoice_ref: str) -> list[LedgerRow]:
        """The invoice's posting: rows with its invoice ref and no payment ref."""
        return [r for r in self.ledger.values() if r.invoice == invoice_ref and r.payment is None]

    def payment_rows(self, payment_ref: str) -> list[LedgerRow]:
        return [r for r in self.ledger.values() if r.payment == payment_ref]

    def rate_date(self, invoice: Invoice) -> date:
        """Local date of `issued_at` (policy fx_rate_date: document_date)."""
        return self.timeline.local_date(invoice.issued_utc)

    def correct_rate(self, invoice: Invoice) -> tuple[date, Decimal]:
        """(date of the fx row used, rate) by the policy rule, for a foreign-currency invoice."""
        return self.fx.rate_on_or_before(self.rate_date(invoice), invoice.currency, self.base_currency)

    def correct_base_amount(self, invoice: Invoice) -> int:
        _, rate = self.correct_rate(invoice)
        return convert(invoice.total_minor, rate, invoice.currency, self.base_currency, self.rounding)

    def protected_docs(self) -> set[str]:
        """Internal refs of documents in distractors or reserved for an injector: other injectors keep off."""
        refs = {ref for d in self.distractors for _, ref in d.docs}
        for targets in self.reserved.values():
            for t in targets:
                refs.add(t.invoice)
                refs.update(t.payments)
                if t.payment is not None:
                    refs.add(t.payment)
        return refs

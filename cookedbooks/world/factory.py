"""Build clean documents with their postings, under a document/row budget.

Every helper stores the records directly in the world (this is the clean
simulation; injectors use the change recorder instead).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta

from cookedbooks.world.model import Invoice, LineItem, Payment, World
from cookedbooks.world.postings import invoice_posting, payment_posting, refund_posting
from cookedbooks.world.rng import chance
from cookedbooks.world.timeutil import DAY

# Unit price ranges in minor units, and the step prices are rounded to.
_PRICE = {
    "KZT": (5_000_00, 600_000_00, 100),
    "USD": (10_00, 1_500_00, 5),
    "EUR": (10_00, 1_400_00, 5),
    "JPY": (1_500, 220_000, 10),
    "KWD": (3_000, 450_000, 5),
}
_SERVICES = (
    "Consulting hours", "Software support", "Cloud hosting", "Training session",
    "Data migration", "Maintenance contract", "Implementation work", "Security audit",
    "Licence subscription", "Report preparation", "Network setup", "Translation services",
)
_DUE_DAYS = (14, 30, 30, 45)
_NULL_PO_PERCENT = 10
_PREVIOUS_MONTH_SERVICE_PERCENT = 25


ADVANCE_SHARE_PERCENT = (5, 40)  # share of a payment left in CUSTOMER_ADVANCES next to a Cr AR row


def advance_share(rng: random.Random) -> int:
    """Percent of a payment that goes to advances when the payment also credits AR.

    One distribution for both the `overpayment` look-alike (legitimate excess
    over a closed invoice) and the `unallocated_payment.partial` error, so the
    split ratio does not tell them apart.
    """
    return rng.randint(*ADVANCE_SHARE_PERCENT)


def advance_part(rng: random.Random, amount: int) -> int:
    """The advances part of a payment of `amount` that also credits AR: in [1, amount - 1]."""
    if amount < 2:
        raise ValueError("an amount below 2 minor units cannot be split")
    return min(max(amount * advance_share(rng) // 100, 1), amount - 1)


def overpaid_amount(rng: random.Random, open_amount: int) -> int:
    """A payment that closes `open_amount` and leaves an advances share drawn like `advance_part`."""
    share = advance_share(rng)
    return max(open_amount * 100 // (100 - share), open_amount + 1)


class BudgetError(ValueError):
    """The requested sizes cannot hold the requested distractors and injections."""


@dataclass
class Budget:
    """Documents and ledger rows still to be generated. Every remaining document needs >= 2 rows."""

    documents: int
    rows: int

    def can(self, documents: int, rows: int) -> bool:
        left_docs, left_rows = self.documents - documents, self.rows - rows
        return left_docs >= 0 and left_rows >= 2 * left_docs

    def take(self, documents: int, rows: int) -> None:
        if not self.can(documents, rows):
            raise BudgetError(
                "n_documents / n_ledger_rows are too small for the requested distractors and errors "
                f"(short of {documents} document(s) and {rows} row(s))"
            )
        self.documents -= documents
        self.rows -= rows


def invoice_rows(world: World, currency: str) -> int:
    return 2 if currency == world.base_currency else 4


def pick_currency(world: World, rng: random.Random, budget: Budget, *, invoices: int, other_docs: int,
                  other_rows: int, foreign_percent: int = 50, foreign_only: bool = False) -> str | None:
    """A currency for a group of `invoices` invoices plus other documents, if the budget allows foreign."""
    docs = invoices + other_docs
    foreign_ok = bool(world.foreign_currencies) and budget.can(docs, 4 * invoices + other_rows)
    if foreign_only:
        return rng.choice(world.foreign_currencies) if foreign_ok else None
    if foreign_ok and chance(rng, foreign_percent):
        return rng.choice(world.foreign_currencies)
    return world.base_currency


def draw_lines(rng: random.Random, currency: str) -> tuple[LineItem, ...]:
    low, high, step = _PRICE[currency]
    lines = []
    for description in rng.sample(_SERVICES, rng.randint(1, 3)):
        quantity = rng.randint(1, 12)
        unit = rng.randrange(low, high + 1, step)
        lines.append(LineItem(description, quantity, unit, quantity * unit))
    return tuple(lines)


def draw_amount(rng: random.Random, currency: str) -> int:
    """An amount shaped like an invoice total."""
    return sum(item.amount_minor for item in draw_lines(rng, currency))


def new_po_ref(world: World, rng: random.Random, *, allow_null: bool = True) -> str | None:
    if allow_null and chance(rng, _NULL_PO_PERCENT):
        return None
    while True:
        ref = f"PO-{rng.randrange(100_000, 1_000_000)}"
        if ref not in world.used_po_refs:
            world.used_po_refs.add(ref)
            return ref


def service_period_for(world: World, rng: random.Random, issued: int) -> str:
    day = world.timeline.local_date(issued)
    if chance(rng, _PREVIOUS_MONTH_SERVICE_PERCENT):
        day = day.replace(day=1) - timedelta(days=1)
    return f"{day.year:04d}-{day.month:02d}"


def unique_lines(world: World, rng: random.Random, customer: str, currency: str) -> tuple[LineItem, ...]:
    """Line items whose total no other invoice of this customer and currency has."""
    while True:
        lines = draw_lines(rng, currency)
        key = (customer, currency, sum(i.amount_minor for i in lines))
        if key not in world.used_totals:
            world.used_totals.add(key)
            return lines


def draw_due_utc(world: World, rng: random.Random, issued: int) -> int:
    """The due instant of an invoice issued at `issued`: end of a local day 14-45 days later."""
    tl = world.timeline
    due_day = tl.local_date(issued) + timedelta(days=rng.choice(_DUE_DAYS))
    return tl.at(due_day, DAY - 1)


def issue_invoice(world: World, rng: random.Random, budget: Budget, *, customer: str, currency: str,
                  issued: int, written_offset: int, posting_ts: int,
                  lines: tuple[LineItem, ...] | None = None, po_ref: str | None | bool = True,
                  service_period: str | None = None) -> Invoice:
    """Create an invoice and its posting. po_ref=True draws a fresh one (maybe null)."""
    budget.take(1, invoice_rows(world, currency))
    tl = world.timeline
    if lines is None:
        lines = unique_lines(world, rng, customer, currency)
    if po_ref is True:
        po_ref = new_po_ref(world, rng)
    due = draw_due_utc(world, rng, issued)
    invoice = Invoice(
        ref=world.new_ref("I"), customer=customer, issued_utc=issued, issued_offset=written_offset,
        due_utc=due, due_offset=tl.offset, currency=currency,
        line_items=lines, total_minor=sum(i.amount_minor for i in lines), po_ref=po_ref or None,
        service_period=service_period or service_period_for(world, rng, issued),
    )
    world.invoices[invoice.ref] = invoice
    rows = invoice_posting(world, invoice, posting_ts)
    if any(r.debit_minor + r.credit_minor <= 0 for r in rows):
        raise AssertionError("conversion produced a zero amount")
    for row in rows:
        world.ledger[row.ref] = row
    return invoice


def receive_payment(world: World, rng: random.Random, budget: Budget, *, customer: str, currency: str,
                    ts: int, amount: int, allocations: list[tuple[str, int]],
                    refs: bool = True) -> Payment:
    """An incoming payment. `refs=False` writes `invoice_refs: null` (only valid with no allocations)."""
    rest = amount - sum(a for _, a in allocations)
    budget.take(1, 1 + len(allocations) + (1 if rest > 0 else 0))
    payment = Payment(
        ref=world.new_ref("P"), customer=customer, ts_utc=ts,
        ts_offset=world.timeline.written_offset(rng, ts), currency=currency, amount_minor=amount,
        invoice_refs=tuple(ref for ref, _ in allocations) if refs else None,
    )
    world.payments[payment.ref] = payment
    for row in payment_posting(world, payment, allocations):
        world.ledger[row.ref] = row
    return payment


def refund(world: World, rng: random.Random, budget: Budget, *, customer: str, currency: str,
           ts: int, amount: int) -> Payment:
    budget.take(1, 2)
    payment = Payment(
        ref=world.new_ref("P"), customer=customer, ts_utc=ts,
        ts_offset=world.timeline.written_offset(rng, ts), currency=currency, amount_minor=-amount,
        invoice_refs=None,
    )
    world.payments[payment.ref] = payment
    for row in refund_posting(world, payment):
        world.ledger[row.ref] = row
    return payment


def posting_time_of(world: World, invoice: Invoice) -> int:
    return world.invoice_rows(invoice.ref)[0].ts_utc


def pay_in_full(world: World, rng: random.Random, budget: Budget, invoice: Invoice, *,
                max_days: int = 35, before: int | None = None, not_before: int = 0) -> Payment | None:
    """One payment allocated fully to `invoice`, after its issue and posting (and `not_before`),
    strictly before `before` if given; None if the window has no room."""
    after = max(invoice.issued_utc, posting_time_of(world, invoice), not_before)
    ts = world.timeline.payment_time(rng, after, max_days=max_days, before=before)
    if ts is None:
        return None
    return receive_payment(world, rng, budget, customer=invoice.customer, currency=invoice.currency,
                           ts=ts, amount=invoice.total_minor, allocations=[(invoice.ref, invoice.total_minor)])

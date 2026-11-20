"""Legitimate look-alikes in clean books (DESIGN.md section 5, "Корректные двойники").

Each builder creates one instance and returns a `Distractor` naming its
documents. With a count of 0 a pattern never occurs: regular documents are
built so that none of them matches any of these patterns.

- split_payment: one invoice paid by two payments, or one payment for two
  invoices; everything fully allocated.
- advance: a payment with `invoice_refs: null` in CUSTOMER_ADVANCES from a
  customer with no open invoice in that currency issued at or before it
  (sometimes followed by a later invoice).
- refund: an invoice paid twice; the second payment sits in advances and is
  returned later by a negative payment.
- overpayment: a payment that references its invoice, closes it in full and
  leaves the excess in CUSTOMER_ADVANCES; the customer has no other open
  invoice in that currency issued at or before it, so the excess is a true
  advance. The advances share is drawn like the one of a partial allocation.

advance, refund and overpayment use an ordinary pool customer and put a guard
on (customer, currency) at their payment with a remainder (see simulate.py):
the customer's other invoices of that currency issued no later are paid
before it.
- recurring_amount: monthly invoices to one customer with the same lines,
  amount and po_ref, different service_period.
- boundary: one invoice near a period bound or a date line, all correct by
  policy (see BOUNDARY_SUBTYPES).
"""

from __future__ import annotations

import random
from datetime import timedelta

from typing import Protocol

from cookedbooks.world.factory import (
    Budget, invoice_rows, issue_invoice, new_po_ref, overpaid_amount, pay_in_full, pick_currency,
    posting_time_of, receive_payment, refund, unique_lines, draw_amount,
)
from cookedbooks.world.model import Distractor, Invoice, World
from cookedbooks.world.rng import chance


class GuardedCustomers(Protocol):
    """What a guarded look-alike needs from the simulator's customer pool."""

    def target(self, currency: str) -> str: ...

    def guard(self, customer: str, currency: str, instant: int) -> None: ...

BOUNDARY_SUBTYPES = ("midnight_offset", "delayed_posting", "next_period", "weekend_rate")
_CHAIN_ROOM_DAYS = 30


def _docs(*items: tuple[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple((kind, obj.ref) for kind, obj in items)


def _regular_invoice(world: World, rng: random.Random, budget: Budget, customer: str, currency: str,
                     *, latest=None, month=None, **kwargs) -> Invoice:
    tl = world.timeline
    issued = tl.regular_invoice_time(rng, latest=latest, month=month)
    return issue_invoice(world, rng, budget, customer=customer, currency=currency, issued=issued,
                         written_offset=tl.written_offset(rng, issued),
                         posting_ts=tl.regular_posting_time(rng, issued), **kwargs)


def build_split_payment(world: World, rng: random.Random, budget: Budget, customer: str) -> Distractor:
    tl = world.timeline
    latest = tl.window_last - timedelta(days=_CHAIN_ROOM_DAYS)
    if chance(rng, 50):  # one invoice, two payments
        currency = pick_currency(world, rng, budget, invoices=1, other_docs=2, other_rows=4)
        invoice = _regular_invoice(world, rng, budget, customer, currency, latest=latest)
        first_part = invoice.total_minor * rng.randint(30, 70) // 100 or 1
        t1 = tl.payment_time(rng, max(invoice.issued_utc, posting_time_of(world, invoice)), max_days=14)
        p1 = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=t1,
                             amount=first_part, allocations=[(invoice.ref, first_part)])
        rest = invoice.total_minor - first_part
        t2 = tl.payment_time(rng, t1, max_days=14)
        p2 = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=t2,
                             amount=rest, allocations=[(invoice.ref, rest)])
        return Distractor("split_payment", _docs(("invoice", invoice), ("payment", p1), ("payment", p2)))
    # one payment, two invoices
    currency = pick_currency(world, rng, budget, invoices=2, other_docs=1, other_rows=3)
    a = _regular_invoice(world, rng, budget, customer, currency, latest=latest)
    b = _regular_invoice(world, rng, budget, customer, currency, latest=latest)
    a, b = sorted((a, b), key=lambda inv: inv.issued_utc)
    after = max(posting_time_of(world, a), posting_time_of(world, b), b.issued_utc)
    ts = tl.payment_time(rng, after, max_days=14)
    payment = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=ts,
                              amount=a.total_minor + b.total_minor,
                              allocations=[(a.ref, a.total_minor), (b.ref, b.total_minor)])
    return Distractor("split_payment", _docs(("invoice", a), ("invoice", b), ("payment", payment)))


def build_advance(world: World, rng: random.Random, budget: Budget, customers: GuardedCustomers) -> Distractor:
    tl = world.timeline
    currency = pick_currency(world, rng, budget, invoices=0, other_docs=1, other_rows=2)
    customer = customers.target(currency)
    ts = tl.time_on(rng, tl.regular_dates)
    amount = draw_amount(rng, currency)
    payment = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=ts,
                              amount=amount, allocations=[], refs=False)
    customers.guard(customer, currency, ts)
    docs = [("payment", payment)]
    later = [d for d in tl.regular_dates if tl.at(d, 0) > ts + 86_400]
    if later and chance(rng, 50) and budget.can(1, invoice_rows(world, currency)):
        issued = tl.time_on(rng, later)
        invoice = issue_invoice(world, rng, budget, customer=customer, currency=currency, issued=issued,
                                written_offset=tl.written_offset(rng, issued),
                                posting_ts=tl.regular_posting_time(rng, issued))
        docs.append(("invoice", invoice))
    return Distractor("advance", _docs(*docs))


def build_refund(world: World, rng: random.Random, budget: Budget, customers: GuardedCustomers) -> Distractor:
    tl = world.timeline
    currency = pick_currency(world, rng, budget, invoices=1, other_docs=3, other_rows=6)
    customer = customers.target(currency)
    latest = tl.window_last - timedelta(days=_CHAIN_ROOM_DAYS)
    invoice = _regular_invoice(world, rng, budget, customer, currency, latest=latest)
    total = invoice.total_minor
    t1 = tl.payment_time(rng, max(invoice.issued_utc, posting_time_of(world, invoice)), max_days=10)
    first = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=t1,
                            amount=total, allocations=[(invoice.ref, total)])
    t2 = tl.payment_time(rng, t1, max_days=8)
    second = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=t2,
                             amount=total, allocations=[], refs=False)
    customers.guard(customer, currency, t2)
    t3 = tl.payment_time(rng, t2, max_days=8)
    back = refund(world, rng, budget, customer=customer, currency=currency, ts=t3, amount=total)
    return Distractor("refund", _docs(("invoice", invoice), ("payment", first), ("payment", second),
                                      ("payment", back)))


def build_overpayment(world: World, rng: random.Random, budget: Budget,
                      customers: GuardedCustomers) -> Distractor:
    """An invoice closed by one payment larger than its total; the excess is a genuine advance."""
    tl = world.timeline
    currency = pick_currency(world, rng, budget, invoices=1, other_docs=1, other_rows=3)
    customer = customers.target(currency)
    invoice = _regular_invoice(world, rng, budget, customer, currency)
    ts = tl.payment_time(rng, max(invoice.issued_utc, posting_time_of(world, invoice)))
    if ts is None:
        raise AssertionError("no room in the window for an overpayment")
    amount = overpaid_amount(rng, invoice.total_minor)
    payment = receive_payment(world, rng, budget, customer=customer, currency=currency, ts=ts,
                              amount=amount, allocations=[(invoice.ref, invoice.total_minor)])
    customers.guard(customer, currency, ts)
    return Distractor("overpayment", _docs(("invoice", invoice), ("payment", payment)))


def build_recurring(world: World, rng: random.Random, budget: Budget, customer: str) -> Distractor:
    tl = world.timeline
    months = sorted({(d.year, d.month) for d in tl.regular_dates})
    run = 3 if len(months) >= 3 and chance(rng, 50) else 2
    run = min(run, len(months))
    currency = pick_currency(world, rng, budget, invoices=run, other_docs=0, other_rows=0)
    while run > 2 and not budget.can(run, run * invoice_rows(world, currency)):
        run -= 1
    start = rng.randrange(len(months) - run + 1)
    lines = unique_lines(world, rng, customer, currency)
    po_ref = new_po_ref(world, rng, allow_null=False)
    docs = []
    for year, month in months[start:start + run]:
        invoice = _regular_invoice(world, rng, budget, customer, currency, month=(year, month),
                                   lines=lines, po_ref=po_ref, service_period=f"{year:04d}-{month:02d}")
        docs.append(("invoice", invoice))
    for _, invoice in list(docs):
        if chance(rng, 50) and budget.can(1, 2):
            payment = pay_in_full(world, rng, budget, invoice)
            if payment is not None:
                docs.append(("payment", payment))
    return Distractor("recurring_amount", _docs(*docs))


def build_boundary(world: World, rng: random.Random, budget: Budget, customer: str, subtype: str) -> Distractor:
    """One invoice near a bound or a date line, posted and converted correctly by policy."""
    tl = world.timeline
    if subtype == "weekend_rate":
        currency = pick_currency(world, rng, budget, invoices=1, other_docs=0, other_rows=0, foreign_only=True)
        if currency is None:
            subtype = "midnight_offset"
    if subtype != "weekend_rate":
        currency = pick_currency(world, rng, budget, invoices=1, other_docs=0, other_rows=0, foreign_percent=70)
    written = None
    if subtype == "midnight_offset":
        day = rng.choice([tl.last, tl.last + timedelta(days=1)] * 2
                         + [tl.first - timedelta(days=1), tl.first])
        written = tl.alternative_offset(rng)
        issued = tl.midnight_time(rng, day, written)
        posting = tl.same_day_later(rng, issued)
    elif subtype == "weekend_rate":
        issued = tl.time_on(rng, tl.weekend_dates)
        posting = tl.regular_posting_time(rng, issued)
    elif subtype == "delayed_posting":
        day, delays = rng.choice([
            (tl.last - timedelta(days=2), (1, 2)), (tl.last - timedelta(days=1), (1,)),
            (tl.last - timedelta(days=2), (1, 2)), (tl.first, (1, 2)), (tl.first + timedelta(days=1), (1,)),
        ])
        issued = tl.at(day, tl.business_second(rng))
        posting = tl.at(day + timedelta(days=rng.choice(delays)), tl.business_second(rng))
    elif subtype == "next_period":
        day = tl.last + timedelta(days=rng.choice([1, 2]))
        issued = tl.at(day, tl.business_second(rng))
        posting = issued if chance(rng, 50) else tl.at(day + timedelta(days=1), tl.business_second(rng))
    else:
        raise ValueError(f"unknown boundary subtype {subtype!r}")
    if written is None:
        written = tl.written_offset(rng, issued)
    invoice = issue_invoice(world, rng, budget, customer=customer, currency=currency, issued=issued,
                            written_offset=written, posting_ts=posting)
    docs = [("invoice", invoice)]
    if chance(rng, 50) and budget.can(1, 2):
        payment = pay_in_full(world, rng, budget, invoice)
        if payment is not None:
            docs.append(("payment", payment))
    return Distractor("boundary", _docs(*docs))

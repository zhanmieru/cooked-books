"""duplicate_invoice: the same obligation invoiced a second time (DESIGN.md section 5).

The simulator builds a clean invoice for an ordinary pool customer (drawn
like the customer of any regular invoice, so it has ordinary other
invoices), issued over the regular date range and either left open or paid
in full by one payment. The injector adds a copy: same customer, po_ref,
service_period, currency, line items and total; new doc_id and invoice_id;
its own correct posting. The copy is drawn like an ordinary invoice: a
regular business-day instant in the month of the service period or the month
after (the months ordinary invoices of that service period are issued in),
the usual written offset, due date and 0-2 day posting delay, and a
conversion at the rate of its own issue date. Regular dates are at least
three days from a period bound, so the posting never crosses one (no cut-off
error) and the conversion follows policy (no fx error).

Currency: with foreign currencies in the config, a seeded draw picks a
foreign invoice half of the time.

Payments: the original and the copy are each paid in full (by one payment
of their own) half of the time, independently, the same odds the invoices of
a recurring_amount look-alike have. So "neither paid / one paid / both paid"
does not separate a duplicate pair from a recurring one. Every payment is
allocated in full, so no other control changes. Exactly two invoices share
the key: the original's total is unique for its customer and currency.
"""

from __future__ import annotations

import dataclasses
from datetime import date

from cookedbooks.world.factory import draw_due_utc, invoice_rows
from cookedbooks.world.inject.base import (
    ApplyContext, Footprint, InjectionOutcome, InjectionPlan, IssueDraft, PlanContext, Ref, ReserveSpec,
)
from cookedbooks.world.inject.registry import register
from cookedbooks.world.model import Invoice, Payment, World
from cookedbooks.world.postings import invoice_posting, payment_posting
from cookedbooks.world.rng import chance

KIND = "duplicate_invoice"
FOREIGN_PERCENT = 50  # share of foreign-currency duplicates, when the config has one
PAID_PERCENT = 50  # each of the two invoices is paid in full with this chance
DESCRIPTION = "Счёт за то же обязательство выставлен второй раз под новым номером"


def _month_after(year: int, month: int) -> tuple[int, int]:
    return (year + month // 12, month % 12 + 1)


def copy_dates(world: World, original: Invoice) -> list[date]:
    """Regular issue dates for a copy: the service month or the month after, not the original's day."""
    year, month = (int(part) for part in original.service_period.split("-"))
    months = {(year, month), _month_after(year, month)}
    candidates = [d for d in world.timeline.regular_dates if (d.year, d.month) in months]
    others = [d for d in candidates if d != world.timeline.local_date(original.issued_utc)]
    return others or candidates or [world.timeline.local_date(original.issued_utc)]


@register(KIND)
class DuplicateInvoice:
    """Add a second invoice for an existing obligation, with its own correct posting."""

    @staticmethod
    def _pay(ctx: ApplyContext, invoice: Invoice, after: int) -> None:
        """One payment allocated in full to `invoice`, timed like a regular payment."""
        world, rng, tl = ctx.world, ctx.rng, ctx.world.timeline
        ts = tl.payment_time(rng, after)
        if ts is None:
            raise AssertionError("no room in the window for the copy's payment")
        payment = Payment(ref=world.new_ref("P"), customer=invoice.customer, ts_utc=ts,
                          ts_offset=tl.written_offset(rng, ts), currency=invoice.currency,
                          amount_minor=invoice.total_minor, invoice_refs=(invoice.ref,))
        ctx.recorder.add("payments", payment)
        ctx.recorder.add_all("ledger", payment_posting(world, payment, [(invoice.ref, invoice.total_minor)]))

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        currency = ctx.base_currency
        if ctx.foreign_currencies and chance(ctx.rng, FOREIGN_PERCENT):
            currency = ctx.rng.choice(ctx.foreign_currencies)
        paid = "full" if chance(ctx.rng, PAID_PERCENT) else "none"
        copy_paid = chance(ctx.rng, PAID_PERCENT)
        rows = (2 if currency == ctx.base_currency else 4) + (2 if copy_paid else 0)
        return InjectionPlan(footprint=Footprint(documents=2 if copy_paid else 1, ledger_rows=rows),
                             reserve=(ReserveSpec(currency=currency, paid=paid, timing="regular",
                                                  customer="pool"),),
                             params={"currency": currency, "paid": paid, "copy_paid": copy_paid})

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        world, rng, tl = ctx.world, ctx.rng, ctx.world.timeline
        original = world.invoices[ctx.targets[0].invoice]
        if original.po_ref is None or original.service_period is None:
            raise AssertionError("a reserved invoice always has a po_ref and a service_period")
        issued = tl.time_on(rng, copy_dates(world, original))
        copy = dataclasses.replace(
            original, ref=world.new_ref("I"), issued_utc=issued,
            issued_offset=tl.written_offset(rng, issued), due_utc=draw_due_utc(world, rng, issued),
            due_offset=tl.offset,
        )
        posted = tl.regular_posting_time(rng, issued)
        rows = invoice_posting(world, copy, posted)
        if len(rows) != invoice_rows(world, copy.currency) or any(r.debit_minor + r.credit_minor <= 0 for r in rows):
            raise AssertionError("the copy's posting does not have the planned shape")
        ctx.recorder.add("invoices", copy)
        ctx.recorder.add_all("ledger", rows)
        if ctx.plan.params["copy_paid"]:
            self._pay(ctx, copy, max(issued, posted))
        pair: list[Ref] = [("invoice", original.ref), ("invoice", copy.ref)]
        issue = IssueDraft(type=KIND, affected=list(pair), expected_amount_minor=copy.total_minor,
                           currency=copy.currency, evidence_sets=[list(pair)])
        return InjectionOutcome(
            issues=[issue],
            effects=[{"control": "C5", "currency": copy.currency, "delta_minor": copy.total_minor}],
            description=DESCRIPTION, kind=KIND,
        )

"""unallocated_payment: a payment left (partly) in CUSTOMER_ADVANCES while its invoice stays open.

Variants (DESIGN.md section 5):
- missing: the payment has `invoice_refs: null`, its whole amount sits in
  CUSTOMER_ADVANCES, the invoice stays open.
- partial: the payment references the invoice, only part of it is credited
  to AR, the rest sits in CUSTOMER_ADVANCES. The advances share is drawn by
  `factory.advance_part`, the same distribution the `overpayment` look-alike
  uses, so the split ratio does not tell an error from a true advance.

The target is a clean invoice of an ordinary pool customer, issued over the
regular date range and paid in full, either by one payment ("single") or by
two (like the split_payment look-alike, "split"). The injector rewrites the
last payment. The reservation carries a guard: every other invoice of that
customer in that currency issued no later than that payment is paid in full
before it, and no look-alike or other target uses the pair. So the DESIGN
guarantee holds: exactly one open invoice of that customer in that currency
issued no later than the payment, and its open balance equals the payment's
unallocated remainder. Reverting the change gives the fully paid invoice.
"""

from __future__ import annotations

from cookedbooks.world.factory import advance_part
from cookedbooks.world.inject.base import (
    ApplyContext, Footprint, InjectionOutcome, InjectionPlan, IssueDraft, PlanContext, Ref, ReserveSpec,
)
from cookedbooks.world.inject.registry import register
from cookedbooks.world.model import LedgerRow, Payment
from cookedbooks.world.rng import chance

KIND = "unallocated_payment"
FOREIGN_PERCENT = 40
SPLIT_PERCENT = 35
DESCRIPTIONS = {
    "missing": "Платёж не распределён: вся сумма ушла в авансы, счёт остался открытым",
    "partial": "Платёж распределён частично: остаток ушёл в авансы, счёт остался открытым",
}


class _UnallocatedPayment:
    """Shared logic; subclasses fix the variant."""

    variant: str = ""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        currency = ctx.base_currency
        if ctx.foreign_currencies and chance(ctx.rng, FOREIGN_PERCENT):
            currency = ctx.rng.choice(ctx.foreign_currencies)
        paid = "split" if chance(ctx.rng, SPLIT_PERCENT) else "full"
        rows = 1 if self.variant == "partial" else 0  # partial adds the Cr CUSTOMER_ADVANCES row
        return InjectionPlan(
            footprint=Footprint(documents=0, ledger_rows=rows),
            reserve=(ReserveSpec(currency=currency, paid=paid, timing="regular", customer="pool", guard=True,
                                 needs_po_ref=False),),
            params={"currency": currency, "paid": paid},
        )

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        target = ctx.targets[0]
        payment, remainder = self._rewrite_payment(ctx, ctx.world.payments[target.payment])
        pair: list[Ref] = [("payment", payment.ref), ("invoice", target.invoice)]
        issue = IssueDraft(type=KIND, affected=list(pair), expected_amount_minor=remainder,
                           currency=payment.currency, evidence_sets=[list(pair)])
        return InjectionOutcome(
            issues=[issue],
            effects=[{"control": "C4", "currency": payment.currency, "delta_minor": remainder}],
            description=DESCRIPTIONS[self.variant], kind=KIND,
        )

    def _rewrite_payment(self, ctx: ApplyContext, payment: Payment) -> tuple[Payment, int]:
        """Turn a clean fully allocated payment into a missing or partial one; returns (payment, remainder)."""
        world, recorder = ctx.world, ctx.recorder
        ar_rows = [r for r in world.payment_rows(payment.ref) if r.account == "AR"]
        if len(ar_rows) != 1 or ar_rows[0].credit_minor != payment.amount_minor:
            raise AssertionError("a reserved payment is one Cr AR row on its full amount")
        ar = ar_rows[0]
        if self.variant == "missing":
            payment = recorder.update("payments", payment.ref, invoice_refs=None)
            recorder.update("ledger", ar.ref, account="CUSTOMER_ADVANCES", invoice=None)
            return payment, payment.amount_minor
        rest = advance_part(ctx.rng, payment.amount_minor)
        recorder.update("ledger", ar.ref, credit_minor=payment.amount_minor - rest)
        recorder.add("ledger", LedgerRow(
            ref=world.new_ref("L"), posting=ar.posting, ts_utc=ar.ts_utc, ts_offset=ar.ts_offset,
            account="CUSTOMER_ADVANCES", debit_minor=0, credit_minor=rest, currency=ar.currency,
            customer=ar.customer, invoice=None, payment=payment.ref,
        ))
        return payment, rest


@register(KIND, "missing")
class MissingAllocation(_UnallocatedPayment):
    """unallocated_payment.missing"""

    variant = "missing"


@register(KIND, "partial")
class PartialAllocation(_UnallocatedPayment):
    """unallocated_payment.partial"""

    variant = "partial"

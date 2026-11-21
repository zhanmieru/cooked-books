"""The two interaction scenarios of DESIGN.md section 6, built from the injectors' building blocks.

S1 - a foreign invoice near the period end, issued where the UTC date (the
written date, on another clock) and the local date differ, converted at the
written date's rate, and its payment allocated only partially. Two issues on
the one invoice: fx_error (base currency) and unallocated_payment (the
remainder, in the invoice currency). The target is reserved like an
unallocated_payment target (pool customer, paid in full by one or two
payments, with the guard that leaves it the only open invoice under the
remainder) and like a "written_date" fx_error target (ReserveSpec.rate_shift,
period end only), so the written date's rate is another fx row and material.

S2 - a duplicate of a foreign invoice whose copy is converted at another
date's rate (a neighbouring business day, or the copy's payment date), never
one that gives the original's base amount: the two copies differ in the base
currency. Two issues: duplicate_invoice on both invoices and fx_error on the
copy. The copy is added with its correct posting first (the duplicate), then
its two base rows are rewritten (the fx error), so provenance shows both.

Each scenario is one provenance injection with `scenario` set, both issue ids
and both effects.
"""

from __future__ import annotations

from cookedbooks.world.fx import FX_TARGET_MIN_BASE_MINOR
from cookedbooks.world.inject import duplicate_invoice as dup
from cookedbooks.world.inject import fx_error as fx
from cookedbooks.world.inject.base import (
    ApplyContext, Footprint, InjectionOutcome, InjectionPlan, PlanContext,
)
from cookedbooks.world.inject.registry import register
from cookedbooks.world.inject.unallocated_payment import SPLIT_PERCENT, rewrite_payment, unallocated_issue
from cookedbooks.world.rng import chance

S2_FLAVOURS = {"neighbour_day": 70, "payment_date": 30}
DESCRIPTIONS = {
    "S1": "Валютный счёт у конца периода пересчитан по курсу даты UTC, его платёж распределён частично",
    "S2": "Счёт выставлен второй раз, и копия пересчитана по курсу другой даты",
}


@register("S1")
class S1:
    """fx_error (written date's rate) + unallocated_payment.partial on one invoice near the period end."""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        fx.require_foreign(ctx, "scenario S1")
        currency = ctx.rng.choice(ctx.foreign_currencies)
        paid = "split" if chance(ctx.rng, SPLIT_PERCENT) else "full"
        spec = fx.target_spec(currency, "midnight_offset", paid=paid, rate_shift=True, bound="end", guard=True)
        # partial allocation adds the Cr CUSTOMER_ADVANCES row
        return InjectionPlan(footprint=Footprint(documents=0, ledger_rows=1), reserve=(spec,),
                             params={"currency": currency, "paid": paid})

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        target = ctx.targets[0]
        invoice = ctx.world.invoices[target.invoice]
        _, posted = fx.wrong_date_amount(ctx.world, ctx.rng, invoice, "written_date")
        fx_issue, fx_effect = fx.misconvert(ctx, invoice, posted)
        payment, rest = rewrite_payment(ctx, ctx.world.payments[target.payment], "partial")
        pay_issue, pay_effect = unallocated_issue(payment, invoice.ref, rest)
        return InjectionOutcome(issues=[fx_issue, pay_issue], effects=[fx_effect, pay_effect],
                                description=DESCRIPTIONS["S1"], kind="S1")


@register("S2")
class S2:
    """duplicate_invoice of a foreign invoice + fx_error on the copy (another date's rate)."""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        fx.require_foreign(ctx, "scenario S2")
        currency = ctx.rng.choice(ctx.foreign_currencies)
        flavour = fx.draw_weighted(ctx.rng, S2_FLAVOURS)
        plan = dup.duplicate_plan(ctx, currency, min_base_minor=FX_TARGET_MIN_BASE_MINOR)
        plan.params["flavour"] = flavour
        return plan

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        world = ctx.world
        original, copy = dup.add_copy(ctx)
        dup_issue, dup_effect = dup.duplicate_issue(original, copy)
        paid = [p.ts_utc for p in world.payments.values() if p.invoice_refs == (copy.ref,)]
        original_base = next(r.credit_minor for r in world.invoice_rows(original.ref) if r.account == "REVENUE")
        _, posted = fx.wrong_date_amount(world, ctx.rng, copy, ctx.plan.params["flavour"],
                                         payment_ts=paid[0] if paid else None, avoid=frozenset({original_base}))
        fx_issue, fx_effect = fx.misconvert(ctx, copy, posted)
        return InjectionOutcome(issues=[dup_issue, fx_issue], effects=[dup_effect, fx_effect],
                                description=DESCRIPTIONS["S2"], kind="S2")

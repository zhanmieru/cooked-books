"""cutoff_error: an invoice posted on the other side of the period end (DESIGN.md section 5).

Only the timestamps of the rows of the invoice's posting change; `issued_at`,
the amounts (so the conversion, which follows `issued_at`) and every payment
stay as in the clean books. The error is therefore never an fx_error too.

Variants:
- late: issued in the last two days of the period, posted after its end.
- early: issued in the first two days after the period end, posted inside it.

The moved posting looks like an ordinary one: the delay is one of the
ordinary posting delays of `timeline.POSTING_DELAY_WEIGHTS` (1-2 days after,
or 1-2 days before, the issue day; ordinary clean invoices are pre-posted now
and then too, so a posting before its invoice is no tell), drawn with the same
weights restricted to the delays that cross the bound, at business hours.

The clean target is posted at its issue instant. A paid target's payment is
reserved at least `PAY_AFTER_DAYS` days after the issue, so it still follows
the moved posting. The customer is an ordinary pool customer; the currency is
the base or a foreign one.
"""

from __future__ import annotations

from cookedbooks.world.inject.base import (
    ApplyContext, Footprint, InjectionOutcome, InjectionPlan, IssueDraft, PlanContext, Ref, ReserveSpec,
)
from cookedbooks.world.inject.registry import register
from cookedbooks.world.rng import chance, weighted_index
from cookedbooks.world.timeline import POSTING_DELAY_WEIGHTS

KIND = "cutoff_error"
FOREIGN_PERCENT = 40
PAID_PERCENT = 65
PAY_AFTER_DAYS = 3  # a moved posting is at most 2 local days + business hours after the issue
INSIDE_DAYS = (-1, 0)  # issue day relative to the period's last day, for "late"
OUTSIDE_DAYS = (1, 2)  # for "early"
DESCRIPTIONS = {
    "late": "Счёт выставлен в периоде, а проведён после его конца",
    "early": "Счёт выставлен после конца периода, а проведён внутри него",
}


def crossing_moves(variant: str) -> list[tuple[int, int, int]]:
    """(issue day relative to the last day, delay in days, weight) for every ordinary delay that crosses the end."""
    moves = []
    for day in (INSIDE_DAYS if variant == "late" else OUTSIDE_DAYS):
        for delay, weight in sorted(POSTING_DELAY_WEIGHTS.items()):
            posted_day = day + delay
            if (variant == "late" and posted_day > 0) or (variant == "early" and posted_day <= 0 < day):
                moves.append((day, delay, weight))
    return moves


class _CutoffError:
    """Shared logic; subclasses fix the variant."""

    variant: str = ""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        moves = crossing_moves(self.variant)
        day, delay, _ = moves[weighted_index(ctx.rng, [w for _, _, w in moves])]
        currency = ctx.base_currency
        if ctx.foreign_currencies and chance(ctx.rng, FOREIGN_PERCENT):
            currency = ctx.rng.choice(ctx.foreign_currencies)
        paid = "full" if chance(ctx.rng, PAID_PERCENT) else "none"
        timing = "near_end_inside" if self.variant == "late" else "near_end_outside"
        spec = ReserveSpec(currency=currency, paid=paid, timing=timing, customer="pool", needs_po_ref=False,
                           end_day=day, pay_after_days=PAY_AFTER_DAYS)
        return InjectionPlan(footprint=Footprint(), reserve=(spec,),
                             params={"currency": currency, "paid": paid, "end_day": day, "delay": delay})

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        world, tl = ctx.world, ctx.world.timeline
        target = ctx.targets[0]
        invoice = world.invoices[target.invoice]
        rows = world.invoice_rows(invoice.ref)
        if not rows or any(r.ts_utc != invoice.issued_utc for r in rows):
            raise AssertionError("a cutoff target is posted at its issue instant")
        posted = tl.posting_at(ctx.rng, invoice.issued_utc, ctx.plan.params["delay"])
        if tl.in_period(posted) == tl.in_period(invoice.issued_utc):
            raise AssertionError("the moved posting does not cross the period end")
        if any(world.payments[p].ts_utc <= posted for p in target.payments):
            raise AssertionError("a payment would precede the moved posting")
        for row in rows:
            ctx.recorder.update("ledger", row.ref, ts_utc=posted)
        ar = [r for r in rows if r.account == "AR" and r.debit_minor > 0]
        revenue = [r for r in rows if r.account == "REVENUE" and r.credit_minor > 0]
        if len(ar) != 1 or len(revenue) != 1:
            raise AssertionError("an invoice posting has one Dr AR and one Cr REVENUE row")
        doc: Ref = ("invoice", invoice.ref)
        issue = IssueDraft(type=KIND, affected=[doc], expected_amount_minor=invoice.total_minor,
                           currency=invoice.currency,
                           evidence_sets=[[doc, ("row", ar[0].ref)], [doc, ("row", revenue[0].ref)]])
        return InjectionOutcome(
            issues=[issue],
            effects=[{"control": "C6", "currency": invoice.currency, "delta_minor": invoice.total_minor}],
            description=DESCRIPTIONS[self.variant], kind=KIND,
        )


@register(KIND, "late")
class LatePosting(_CutoffError):
    """cutoff_error.late"""

    variant = "late"


@register(KIND, "early")
class EarlyPosting(_CutoffError):
    """cutoff_error.early"""

    variant = "early"

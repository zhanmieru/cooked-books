"""fx_error: a foreign-currency invoice converted at a wrong rate (DESIGN.md section 5).

Only the two base-currency rows of the invoice's posting change (Dr
FX_CLEARING and Cr REVENUE, both in the base currency); the invoice, its AR
row, its FX_CLEARING row in the invoice currency and its payments stay as in
the clean books. The posted amount is a real conversion, `convert(total,
wrong rate)`, so the implied rate is a rate an accountant could have taken.

Variants:
- wrong_date: the rate of the right pair on another date. One of four
  flavours, drawn in `plan`:
  - "written_date": a near-midnight invoice written on another clock (UTC for
    a non-UTC company), converted by the written calendar date instead of the
    local one; next to a period bound or on an ordinary mid-period weekday
    (the `boundary` look-alike "midnight_offset" sits in both places too).
    The target is reserved so that the two dates resolve to different fx
    rows (ReserveSpec.rate_shift).
  - "neighbour_day": an ordinary mid-period invoice converted at the previous
    or the next business day's rate.
  - "payment_date": an ordinary paid invoice converted at the rate of its
    payment's date (falls back to a neighbouring day when that rate is the
    same row or not material).
  - "weekend": a Saturday/Sunday invoice converted at Monday's rate instead of
    the previous available (Friday's).
- wrong_pair: the rate of another pair in fx.csv on the correct date; of the
  pairs whose result is material, the one closest to the correct amount (the
  most plausible mix-up). Targets are ordinary, near-midnight or weekend
  invoices.

Materiality by construction: targets convert to at least
`FX_TARGET_MIN_BASE_MINOR` base minor units, and consecutive published rates
of a pair differ by at least `fx.MIN_MOVE_BP`, so a neighbouring business
day's rate is always material. Every choice is still checked exactly with
`fx.is_material`, trying the candidates in order of preference; a candidate
that is not material is never used, so no difference above zero but below the
threshold can appear.

Targets: an ordinary pool customer (drawn like a regular invoice's), a
foreign currency of the config, paid in full with the odds of a regular
invoice. A reserved target never carries a duplicate key (its total is unique
for the customer) and is posted on the same side of the period bounds as its
issue date, so it is neither a duplicate nor a cut-off error.
"""

from __future__ import annotations

import random
from datetime import date

from cookedbooks.world.fx import FX_TARGET_MIN_BASE_MINOR, FxTable, convert, is_material
from cookedbooks.world.inject.base import (
    ApplyContext, InjectionOutcome, InjectionPlan, IssueDraft, PlanContext, Ref, ReserveSpec,
)
from cookedbooks.world.inject.registry import register
from cookedbooks.world.model import Invoice, World
from cookedbooks.world.rng import chance, weighted_index
from cookedbooks.world.timeutil import local_date

KIND = "fx_error"
PAID_PERCENT = 60  # with "payment_date" always paid, about 2/3 of targets are paid, as regular invoices
WRONG_DATE_FLAVOURS = {"written_date": 30, "neighbour_day": 35, "payment_date": 20, "weekend": 15}
WRONG_PAIR_TIMINGS = {"regular": 65, "midnight_offset": 20, "weekend": 15}
MIDNIGHT_MID_PERIOD_PERCENT = 50  # near-midnight targets on an ordinary mid-period day instead of near a bound
FLAVOUR_TIMING = {"written_date": "midnight_offset", "neighbour_day": "regular", "payment_date": "regular",
                  "weekend": "weekend"}
DESCRIPTIONS = {
    "wrong_date": "Валютный счёт пересчитан по курсу другой даты",
    "wrong_pair": "Валютный счёт пересчитан по курсу другой валютной пары",
}


def draw_weighted(rng, weights: dict[str, int]) -> str:
    names = list(weights)
    return names[weighted_index(rng, [weights[n] for n in names])]


def require_foreign(ctx: PlanContext, what: str) -> None:
    if not ctx.foreign_currencies:
        raise ValueError(f"{what} needs a foreign currency: add one to team_config.currencies")


def target_spec(currency: str, timing: str, *, paid: str, rate_shift: bool = False, bound: str = "any",
                guard: bool = False) -> ReserveSpec:
    """A reserved fx_error target: pool customer, big enough for materiality, po_ref drawn as usual."""
    return ReserveSpec(currency=currency, paid=paid, timing=timing, customer="pool", needs_po_ref=False,
                       min_base_minor=FX_TARGET_MIN_BASE_MINOR, rate_shift=rate_shift, bound=bound, guard=guard)


# --- conversions ---------------------------------------------------------------------


def pair_dates(world: World, currency: str) -> list[date]:
    """The dates fx.csv has a rate for currency -> base, in order."""
    base = world.base_currency
    return sorted(d for (d, s, t) in world.fx.text if s == currency and t == base)


def at_date(world: World, invoice: Invoice, day: date) -> tuple[date, int]:
    """(fx row date, base amount) when `invoice` is converted by `day` with the policy's fallback."""
    row_day, rate = world.fx.rate_on_or_before(day, invoice.currency, world.base_currency)
    return row_day, convert(invoice.total_minor, rate, invoice.currency, world.base_currency, world.rounding)


def neighbours(world: World, invoice: Invoice, order: tuple[int, ...]) -> list[date]:
    """The published dates just before (-1) and after (+1) the invoice's correct fx row, in `order`."""
    dates = pair_dates(world, invoice.currency)
    position = dates.index(world.correct_rate(invoice)[0])
    found = []
    for step in order:
        if 0 <= position + step < len(dates):
            found.append(dates[position + step])
    return found


def by_distance(world: World, invoice: Invoice) -> list[date]:
    """Every other published date of the pair, nearest to the correct row first (last-resort fallback)."""
    correct = world.correct_rate(invoice)[0]
    others = [d for d in pair_dates(world, invoice.currency) if d != correct]
    return sorted(others, key=lambda d: (abs((d - correct).days), d))


def first_material(world: World, invoice: Invoice, days: list[date], *,
                   avoid: frozenset[int] = frozenset()) -> tuple[date, int]:
    """The first of `days` whose rate gives a material, different amount (not in `avoid`)."""
    correct = world.correct_base_amount(invoice)
    for day in [*days, *by_distance(world, invoice)]:
        row_day, posted = at_date(world, invoice, day)
        if row_day != world.correct_rate(invoice)[0] and posted not in avoid and is_material(posted, correct):
            return row_day, posted
    raise AssertionError("no material wrong date for this invoice")


def wrong_date_amount(world: World, rng: random.Random, invoice: Invoice, flavour: str, *,
                      payment_ts: int | None = None, avoid: frozenset[int] = frozenset()) -> tuple[str, int]:
    """(what was used, posted base amount) for a wrong_date flavour.

    `payment_ts`: the invoice's (last) payment, for "payment_date". `avoid`: base amounts not to post.
    """
    tl = world.timeline
    if flavour == "written_date":
        written = local_date(invoice.issued_utc, invoice.issued_offset)  # the calendar date on the document
        row_day, posted = at_date(world, invoice, written)
        if row_day == world.correct_rate(invoice)[0] or not is_material(posted, world.correct_base_amount(invoice)) \
                or posted in avoid:
            raise AssertionError("a rate_shift target's written date must give a material other rate")
        return f"written date {written.isoformat()}", posted
    if flavour == "weekend":
        days = neighbours(world, invoice, (1, -1))
    else:
        days = neighbours(world, invoice, (-1, 1) if rng.randrange(2) == 0 else (1, -1))
    if flavour == "payment_date" and payment_ts is not None:
        days = [tl.local_date(payment_ts), *days]
    row_day, posted = first_material(world, invoice, days, avoid=avoid)
    return f"rate of {row_day.isoformat()}", posted


def wrong_pair_amount(world: World, invoice: Invoice) -> tuple[str, int]:
    """(pair used, posted base amount): the closest material result among the other pairs on the right date."""
    row_day, _ = world.correct_rate(invoice)
    correct = world.correct_base_amount(invoice)
    base = world.base_currency
    sources = sorted({*world.foreign_currencies, *([world.reference_currency] if world.reference_currency else [])}
                     - {invoice.currency, base})
    options = []
    for source in sources:
        rate = world.fx.rate(row_day, source, base)
        if rate is None:
            continue
        posted = convert(invoice.total_minor, rate, invoice.currency, base, world.rounding)
        if posted > 0 and is_material(posted, correct):
            options.append((abs(posted - correct), source, posted))
    if not options:
        raise AssertionError("no other pair gives a material amount")
    _, source, posted = min(options)
    return f"pair {source}:{base}", posted


# --- the change and the truth -------------------------------------------------------------


def misconvert(ctx: ApplyContext, invoice: Invoice, posted: int) -> tuple[IssueDraft, dict]:
    """Rewrite the two base-currency rows of the invoice's posting to `posted`; the issue and C3 effect."""
    world, base = ctx.world, ctx.world.base_currency
    correct = world.correct_base_amount(invoice)
    if not is_material(posted, correct):
        raise AssertionError("an fx_error must be material")
    rows = world.invoice_rows(invoice.ref)
    clearing = [r for r in rows if r.account == "FX_CLEARING" and r.currency == base and r.debit_minor > 0]
    revenue = [r for r in rows if r.account == "REVENUE" and r.currency == base and r.credit_minor > 0]
    if len(rows) != 4 or len(clearing) != 1 or len(revenue) != 1 or revenue[0].credit_minor != correct:
        raise AssertionError("an fx_error target is a correctly converted four-row posting")
    ctx.recorder.update("ledger", clearing[0].ref, debit_minor=posted)
    ctx.recorder.update("ledger", revenue[0].ref, credit_minor=posted)
    amount = abs(posted - correct)
    key = FxTable.key(world.correct_rate(invoice)[0], invoice.currency, base)
    evidence: list[Ref] = [("invoice", invoice.ref), ("row", revenue[0].ref), ("fx", key)]
    issue = IssueDraft(type=KIND, affected=[("invoice", invoice.ref)], expected_amount_minor=amount,
                       currency=base, evidence_sets=[evidence])
    return issue, {"control": "C3", "currency": base, "delta_minor": amount}


class _FxError:
    """Shared logic; subclasses fix the variant."""

    variant: str = ""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        require_foreign(ctx, f"{KIND}.{self.variant}")
        currency = ctx.rng.choice(ctx.foreign_currencies)
        if self.variant == "wrong_date":
            flavour = draw_weighted(ctx.rng, WRONG_DATE_FLAVOURS)
            timing = FLAVOUR_TIMING[flavour]
        else:
            flavour, timing = "wrong_pair", draw_weighted(ctx.rng, WRONG_PAIR_TIMINGS)
        paid = "full" if flavour == "payment_date" or chance(ctx.rng, PAID_PERCENT) else "none"
        bound = "any"
        if timing == "midnight_offset" and chance(ctx.rng, MIDNIGHT_MID_PERIOD_PERCENT):
            bound = "mid"
        spec = target_spec(currency, timing, paid=paid, rate_shift=flavour == "written_date", bound=bound)
        return InjectionPlan(reserve=(spec,), params={"currency": currency, "flavour": flavour, "paid": paid})

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        invoice = ctx.world.invoices[ctx.targets[0].invoice]
        flavour = ctx.plan.params["flavour"]
        if self.variant == "wrong_date":
            target = ctx.targets[0]
            payment_ts = ctx.world.payments[target.payment].ts_utc if target.payment else None
            _, posted = wrong_date_amount(ctx.world, ctx.rng, invoice, flavour, payment_ts=payment_ts)
        else:
            _, posted = wrong_pair_amount(ctx.world, invoice)
        issue, effect = misconvert(ctx, invoice, posted)
        return InjectionOutcome(issues=[issue], effects=[effect], description=DESCRIPTIONS[self.variant],
                                kind=KIND)


@register(KIND, "wrong_date")
class WrongDate(_FxError):
    """fx_error.wrong_date"""

    variant = "wrong_date"


@register(KIND, "wrong_pair")
class WrongPair(_FxError):
    """fx_error.wrong_pair"""

    variant = "wrong_pair"

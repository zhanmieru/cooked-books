"""Simulate the clean company world for one period, at exactly the configured size minus a footprint.

Order: calendar and fx -> customers -> injector targets (reserved) ->
distractors -> regular invoices and payments -> background postings that
pad the ledger to the exact row count.

Customers. Regular invoices draw their customer from a weighted pool. Injector
targets that ask for a "pool" customer and the look-alikes draw theirs the
same way, so the customers of injected records have the same number of
invoices as anyone else's. A payment that legitimately leaves a remainder in
CUSTOMER_ADVANCES (advance, refund, overpayment), and a target that will get
one (unallocated_payment), puts a *guard* on its (customer, currency): every
other invoice of that pair issued no later than the payment is paid in full
before the payment, and no other look-alike or target uses the pair. So C4
holds for the look-alikes, and an injected remainder has exactly one open
invoice to go to.
"""

from __future__ import annotations

import random
from datetime import timedelta

from cookedbooks.world import distractors as dz
from cookedbooks.world.config import WorldConfig
from cookedbooks.world.factory import (
    Budget, BudgetError, issue_invoice, new_po_ref, pay_in_full, posting_time_of, receive_payment, unique_lines,
)
from cookedbooks.world.fx import EXPONENTS, LOOKBACK_DAYS, build_fx, convert
from cookedbooks.world.inject.base import InjectionPlan, ReserveSpec
from cookedbooks.world.model import Invoice, LineItem, Payment, ReservedTarget, World
from cookedbooks.world.postings import background_posting
from cookedbooks.world.rng import chance, config_salt, stream, weighted_index
from cookedbooks.world.timeline import NEAR_MAX_DISTANCE, Timeline
from cookedbooks.world.timeutil import DAY, HOUR, days

_REGULAR_PAID_PERCENT = 40  # share of regular documents that are payments (each pays one invoice)
_REGULAR_FOREIGN_PERCENT = 40
_DISTRACTOR_ORDER = ("refund", "advance", "overpayment", "split_payment", "recurring_amount", "boundary")
_GUARDED_KINDS = frozenset({"refund", "advance", "overpayment"})
_CUSTOMER_TRIES = 50
_SPLIT_GAP_DAYS = 14


class Customers:
    """The weighted customer pool, plus the guards that keep advances-carrying payments valid."""

    def __init__(self, world: World, rng: random.Random, size: int) -> None:
        self.world = world
        self.rng = rng
        self.pool = [world.new_ref("C") for _ in range(size)]
        self.weights = [12 // (rank + 1) + 1 for rank in range(size)]
        self.guards: dict[tuple[str, str], int] = {}  # (customer, currency) -> instant of the payment
        self.taken: set[tuple[str, str]] = set()  # pairs used by a reserved target or a guard
        self.guarded_customers: set[str] = set()

    def draw(self) -> str:
        """A customer as a regular invoice gets one."""
        return self.pool[weighted_index(self.rng, self.weights)]

    def ordinary(self) -> str:
        """For a look-alike without a guard: a pool customer that carries no guard."""
        for _ in range(_CUSTOMER_TRIES):
            customer = self.draw()
            if customer not in self.guarded_customers:
                return customer
        return self.world.new_ref("C")

    def target(self, currency: str) -> str:
        """For a reserved target or a guarded look-alike: a pool customer whose pair is free."""
        for _ in range(_CUSTOMER_TRIES):
            customer = self.draw()
            if (customer, currency) not in self.taken:
                return customer
        return self.world.new_ref("C")

    def take(self, customer: str, currency: str) -> None:
        self.taken.add((customer, currency))

    def guard(self, customer: str, currency: str, instant: int) -> None:
        """Settle every other invoice of (customer, currency) issued at or before `instant`, before it."""
        if (customer, currency) in self.guards:
            raise AssertionError("one guard per customer and currency")
        self.guards[(customer, currency)] = instant
        self.taken.add((customer, currency))
        self.guarded_customers.add(customer)

    def settle_before(self, customer: str, currency: str, issued: int) -> int | None:
        """The instant an invoice of this pair issued at `issued` must be paid before, or None."""
        guard = self.guards.get((customer, currency))
        return guard if guard is not None and issued <= guard else None


def simulate(config: WorldConfig, seed: int, attempt: int, plans: list[InjectionPlan]) -> World:
    """The clean world; `plans` give the footprint to leave free and the targets to build."""
    salt = config_salt(config.generator_config)

    def rng_for(label: str) -> random.Random:
        return stream(seed, attempt, f"{salt}:{label}")

    timeline = Timeline(config.period_first, config.period_last, config.utc_offset)
    foreign = config.foreign_currencies
    reference = None
    if foreign:
        spare = sorted(set(EXPONENTS) - set(config.currencies))
        reference = rng_for("fx:reference").choice(spare)
    sources = tuple(sorted({*foreign, *([reference] if reference else [])}))
    fx = build_fx(rng_for("fx"), timeline.window_first - timedelta(days=LOOKBACK_DAYS),
                  timeline.window_last, config.base_currency, sources)
    world = World(timeline=timeline, fx=fx, base_currency=config.base_currency,
                  foreign_currencies=foreign, reference_currency=reference, rounding=config.rounding)

    footprint_docs = sum(p.footprint.documents for p in plans)
    footprint_rows = sum(p.footprint.ledger_rows for p in plans)
    budget = Budget(config.n_documents - footprint_docs, config.n_ledger_rows - footprint_rows)
    if budget.documents < 0 or budget.rows < 2 * budget.documents:
        raise BudgetError("the requested errors leave no room in n_documents / n_ledger_rows")

    customers = Customers(world, rng_for("customers"), max(2, config.n_documents // 6))

    for index, plan in enumerate(plans):
        rng = rng_for(f"reserve:{index}")
        world.reserved[index] = tuple(_build_reserved(world, rng, budget, spec, config, customers)
                                      for spec in plan.reserve)

    subtypes = [s for s in dz.BOUNDARY_SUBTYPES
                if (foreign or s != "weekend_rate") and (timeline.offset != 0 or s != "date_line")]
    rng_for("distractor:boundary:subtypes").shuffle(subtypes)
    for kind in _DISTRACTOR_ORDER:
        for i in range(config.distractors[kind]):
            rng = rng_for(f"distractor:{kind}:{i}")
            if kind in _GUARDED_KINDS:
                build = {"advance": dz.build_advance, "refund": dz.build_refund,
                         "overpayment": dz.build_overpayment}[kind]
                built = build(world, rng, budget, customers)
            else:
                customer = customers.ordinary()
                if kind == "split_payment":
                    built = dz.build_split_payment(world, rng, budget, customer)
                elif kind == "recurring_amount":
                    built = dz.build_recurring(world, rng, budget, customer)
                else:
                    built = dz.build_boundary(world, rng, budget, customer, subtypes[i % len(subtypes)])
            world.distractors.append(built)

    background_rows = _regular_documents(world, rng_for("regular"), budget, customers)
    _background(world, rng_for("background"), background_rows)
    budget.take(0, background_rows)
    assert budget.documents == 0 and budget.rows == 0, budget
    return world


def _build_reserved(world: World, rng: random.Random, budget: Budget, spec: ReserveSpec,
                    config: WorldConfig, customers: Customers) -> ReservedTarget:
    if spec.currency not in config.currencies:
        raise ValueError(f"an injector reserved an invoice in {spec.currency}, which is not in currencies")
    tl = world.timeline
    customer = world.new_ref("C") if spec.customer == "own" else customers.target(spec.currency)
    written = None
    if spec.timing == "regular":  # the full regular range, as ordinary invoices
        issued = tl.regular_invoice_time(rng)
        posting = tl.regular_posting_time(rng, issued)
    elif spec.timing == "weekend":  # like the weekend look-alike: regular posting delay
        issued = tl.time_on(rng, tl.weekend_dates)
        posting = tl.regular_posting_time(rng, issued)
    else:
        if spec.timing == "midnight_offset":
            issued, written = _midnight_target(world, rng, spec)
        else:
            near = [-1, 0] if spec.timing == "near_end_inside" else [1, 2]
            offsets = near if spec.end_day is None else [spec.end_day]
            issued = tl.time_on(rng, [tl.last + timedelta(days=d) for d in offsets])
        posting = issued
    if written is None:
        written = tl.written_offset(rng, issued)
    invoice = issue_invoice(world, rng, budget, customer=customer, currency=spec.currency, issued=issued,
                            written_offset=written, posting_ts=posting,
                            lines=_target_lines(world, rng, spec, customer, issued),
                            po_ref=new_po_ref(world, rng, allow_null=not spec.needs_po_ref))
    payments: list[Payment] = []
    not_before = issued + spec.pay_after_days * DAY
    if spec.paid == "full":
        payment = pay_in_full(world, rng, budget, invoice, not_before=not_before)
        if payment is None:
            raise AssertionError("no room in the window for a reserved payment")
        payments.append(payment)
    elif spec.paid == "split":
        payments.extend(_pay_in_two(world, rng, budget, invoice, not_before=not_before))
    if spec.guard:
        customers.guard(customer, spec.currency, payments[-1].ts_utc)
    else:
        customers.take(customer, spec.currency)
    return ReservedTarget(customer=customer, invoice=invoice.ref,
                          payment=payments[-1].ref if payments else None,
                          payments=tuple(p.ref for p in payments))


_LINE_TRIES = 200


def _target_lines(world: World, rng: random.Random, spec: ReserveSpec, customer: str,
                  issued: int) -> tuple[LineItem, ...]:
    """Unique line items; for a foreign target with `min_base_minor`, redrawn until big enough."""
    for _ in range(_LINE_TRIES):
        lines = unique_lines(world, rng, customer, spec.currency)
        if spec.min_base_minor == 0 or spec.currency == world.base_currency:
            return lines
        _, rate = world.fx.rate_on_or_before(world.timeline.local_date(issued), spec.currency,
                                             world.base_currency)
        total = sum(item.amount_minor for item in lines)
        if convert(total, rate, spec.currency, world.base_currency, world.rounding) >= spec.min_base_minor:
            return lines
    raise AssertionError(f"no invoice of at least {spec.min_base_minor} base minor units in {_LINE_TRIES} draws")


def _midnight_target(world: World, rng: random.Random, spec: ReserveSpec) -> tuple[int, int]:
    """(issue instant, written offset) of a "midnight_offset" target (see ReserveSpec).

    Local day near the bound(s), written on another clock so the written calendar
    date is the day before or after. With `rate_shift`, only days whose written
    date resolves to another fx row than the local date; if none of the days next
    to the bound qualify, the days within NEAR_MAX_DISTANCE of it are used.
    """
    tl = world.timeline
    one = timedelta(days=1)
    if spec.bound == "mid":
        near = list(tl.regular_dates)
    else:
        near = [tl.last, tl.last + one] + ([] if spec.bound == "end" else [tl.first - one, tl.first])
    if not spec.rate_shift:
        day = rng.choice(near)
        written = tl.alternative_offset(rng)
        return tl.midnight_time(rng, day, written), written
    clocks = [0] if tl.offset != 0 else [360, -300]

    def shifts(day, written: int) -> bool:
        written_day = day + one if written > tl.offset else day - one
        rate_day = world.fx.rate_on_or_before(day, spec.currency, world.base_currency)[0]
        return world.fx.rate_on_or_before(written_day, spec.currency, world.base_currency)[0] != rate_day

    candidates = [(d, w) for d in near for w in clocks if shifts(d, w)]
    if not candidates and spec.bound != "mid":
        span = NEAR_MAX_DISTANCE * one
        wider = days(tl.last - span, tl.last + span + one)
        if spec.bound == "any":
            wider += days(tl.first - span - one, tl.first + span)
        candidates = [(d, w) for d in wider for w in clocks if shifts(d, w)]
    if not candidates:
        raise AssertionError("no day near the bound where the written date changes the fx row")
    day, written = rng.choice(candidates)
    return tl.midnight_time(rng, day, written), written


def _pay_in_two(world: World, rng: random.Random, budget: Budget, invoice: Invoice, *,
                not_before: int = 0) -> list[Payment]:
    """Two payments, each allocated in full, together the invoice total (as the split_payment look-alike)."""
    tl = world.timeline
    first_part = invoice.total_minor * rng.randint(30, 70) // 100 or 1
    after = max(invoice.issued_utc, posting_time_of(world, invoice), not_before)
    t1 = tl.payment_time(rng, after, max_days=_SPLIT_GAP_DAYS, before=tl.window_end_utc - HOUR + 1)
    if t1 is None:
        raise AssertionError("no room in the window for a reserved split payment")
    p1 = receive_payment(world, rng, budget, customer=invoice.customer, currency=invoice.currency, ts=t1,
                         amount=first_part, allocations=[(invoice.ref, first_part)])
    rest = invoice.total_minor - first_part
    t2 = tl.payment_time(rng, t1, max_days=_SPLIT_GAP_DAYS)
    p2 = receive_payment(world, rng, budget, customer=invoice.customer, currency=invoice.currency, ts=t2,
                         amount=rest, allocations=[(invoice.ref, rest)])
    return [p1, p2]


def _regular_documents(world: World, rng: random.Random, budget: Budget, customers: Customers) -> int:
    """Fill the remaining documents with regular invoices and payments; return the background row count."""
    documents, rows = budget.documents, budget.rows
    n_payments = min(documents // 2, documents * _REGULAR_PAID_PERCENT // 100)
    n_invoices = documents - n_payments
    spare_rows = rows - 2 * documents
    flags = [bool(world.foreign_currencies) and chance(rng, _REGULAR_FOREIGN_PERCENT) for _ in range(n_invoices)]
    n_foreign = min(sum(flags), spare_rows // 2)
    if spare_rows - 2 * n_foreign == 1 and n_foreign > 0:
        n_foreign -= 1
    background = spare_rows - 2 * n_foreign
    if background == 1:
        raise BudgetError(
            "cannot fill the ledger exactly: one row would be left over; "
            "change n_ledger_rows or n_documents by one"
        )
    kept = 0
    tl = world.timeline
    invoices = []
    settle: dict[int, int] = {}  # position -> instant its payment must come before (a guard)
    for flag in flags:
        foreign = flag and kept < n_foreign
        kept += foreign
        currency = rng.choice(world.foreign_currencies) if foreign else world.base_currency
        customer = customers.draw()
        issued = tl.regular_invoice_time(rng)
        written = tl.written_offset(rng, issued)
        posting = tl.regular_posting_time(rng, issued)
        customer, before = _settleable_customer(customers, customer, currency, issued, posting,
                                                room=len(settle) < n_payments)
        if before is not None:
            settle[len(invoices)] = before
        invoices.append(issue_invoice(world, rng, budget, customer=customer, currency=currency, issued=issued,
                                      written_offset=written, posting_ts=posting))
    if settle:
        others = [i for i in range(n_invoices) if i not in settle]
        positions = sorted([*settle, *rng.sample(others, n_payments - len(settle))])
    else:
        positions = sorted(rng.sample(range(n_invoices), n_payments))
    for position in positions:
        if pay_in_full(world, rng, budget, invoices[position], before=settle.get(position)) is None:
            raise AssertionError("no room in the window for a regular payment")
    return background


def _settleable_customer(customers: Customers, customer: str, currency: str, issued: int, posting: int,
                         *, room: bool) -> tuple[str, int | None]:
    """Keep `customer` unless its guard cannot be honoured; returns (customer, settle-before instant).

    An invoice on a guarded pair issued at or before the guard must be paid before it. That needs
    an hour between posting and guard and a free regular payment; otherwise another customer is drawn.
    """
    for _ in range(_CUSTOMER_TRIES):
        before = customers.settle_before(customer, currency, issued)
        if before is None:
            return customer, None
        if room and max(issued, posting) + HOUR <= before - 1:
            return customer, before
        customer = customers.draw()
    return customers.world.new_ref("C"), None


def _background_amount(rng: random.Random, base: str, account: str) -> int:
    scale = {"RENT_EXPENSE": (40, 120), "PAYROLL_EXPENSE": (20, 200), "UTILITIES_EXPENSE": (2, 30)}[account]
    unit = 10 ** EXPONENTS[base]
    major = {"KZT": 10_000, "USD": 20, "EUR": 20, "JPY": 3_000, "KWD": 7}[base]
    return rng.randint(*scale) * major * unit + rng.randrange(unit)


def _background(world: World, rng: random.Random, rows: int) -> None:
    """Exactly `rows` rows of expense postings without documents (2 rows, or 3 when parity needs it)."""
    if rows == 1:
        raise BudgetError("one background row cannot form a balanced posting")
    three_row = rows % 2
    while rows >= 3 * (three_row + 2) + 2 and chance(rng, 30) and three_row < 6:
        three_row += 2
    two_row = (rows - 3 * three_row) // 2
    shapes = [3] * three_row + [2] * two_row
    rng.shuffle(shapes)
    tl = world.timeline
    dates = days(tl.window_first, tl.window_last)
    base = world.base_currency
    for shape in shapes:
        ts = tl.time_on(rng, dates)
        if shape == 2:
            account = ("RENT_EXPENSE", "PAYROLL_EXPENSE", "UTILITIES_EXPENSE")[weighted_index(rng, [10, 30, 60])]
            lines = [(account, _background_amount(rng, base, account))]
        elif chance(rng, 50):
            lines = [("RENT_EXPENSE", _background_amount(rng, base, "RENT_EXPENSE")),
                     ("UTILITIES_EXPENSE", _background_amount(rng, base, "UTILITIES_EXPENSE"))]
        else:
            lines = [("PAYROLL_EXPENSE", _background_amount(rng, base, "PAYROLL_EXPENSE")),
                     ("PAYROLL_EXPENSE", _background_amount(rng, base, "PAYROLL_EXPENSE"))]
        for row in background_posting(world, ts, lines):
            world.ledger[row.ref] = row

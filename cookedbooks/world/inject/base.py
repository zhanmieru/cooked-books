"""The injector interface.

An injector is any object with two methods:

    plan(ctx: PlanContext) -> InjectionPlan
        Called before the clean world exists. Decide everything that changes
        record counts: the footprint (documents and ledger rows the injection
        will add, net; negative if it removes) and the clean targets the
        simulator must build for it (`reserve`). Keep free-form decisions in
        `params`. Use only `ctx.rng` for randomness.

    apply(ctx: ApplyContext) -> InjectionOutcome
        Called on the finished clean world. Read anything from `ctx.world`;
        change it ONLY through `ctx.recorder` (add / update / remove). Return
        the truth issues (with internal refs) and the control effects.

The pipeline checks that the net change in documents and rows equals the
declared footprint, so the final files have exactly the configured sizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from cookedbooks.world.model import ReservedTarget, World
from cookedbooks.world.timeline import Timeline

# A reference to something that gets a public ID at export time:
#   ("invoice", ref) / ("payment", ref) -> that document's doc_id
#   ("row", ref)                        -> row_id
#   ("invoice_id", ref) / ("payment_id", ref) -> the business key
#   ("fx", "2026-09-30:USD:KZT")        -> the fx key, unchanged
Ref = tuple[str, str]

TIMINGS = ("regular", "midnight_offset", "weekend", "near_end_inside", "near_end_outside")
PAID = ("none", "full", "split")
CUSTOMERS = ("own", "pool")


@dataclass(frozen=True)
class InjectionRequest:
    """One requested error: kind is an error type or a scenario name ("S1", "S2")."""

    kind: str
    variant: str | None = None
    scenario: str | None = None
    index: int = 0  # position in the pipeline; also the key of `World.reserved`


@dataclass(frozen=True)
class Footprint:
    documents: int = 0
    ledger_rows: int = 0


@dataclass(frozen=True)
class ReserveSpec:
    """A clean invoice the simulator builds for an injector.

    currency: an exact code from the config's currencies.
    paid: "none" (left open), "full" (one payment allocated in full) or
        "split" (two payments, each allocated in full, together the total).
    customer: "own" (a customer created for this target, with no other
        documents) or "pool" (an ordinary customer drawn like the customer of
        a regular invoice, so it has ordinary other invoices). No other
        reserved target uses the same customer and currency.
    guard: only with paid "full"/"split". The customer's other invoices in
        this currency issued at or before the target's last payment are paid
        in full before that payment, and no look-alike uses this customer, so
        the target invoice is the customer's only open invoice in that
        currency issued no later than the payment (DESIGN.md, C4).
    needs_po_ref: True (default) gives the invoice a po_ref and a service_period
        so it can carry a duplicate; False draws po_ref like a regular
        invoice (null now and then).
    timing: "regular" (the regular distribution and posting delay),
        "midnight_offset" (near a period bound, written on another clock so the
        written date differs from the local date), "weekend" (local Sat/Sun),
        "near_end_inside" (last 2 days of the period), "near_end_outside"
        (first 2 days after it). Non-regular timings post at the issue instant.
    """

    currency: str
    paid: str = "none"
    timing: str = "regular"
    customer: str = "own"
    guard: bool = False
    needs_po_ref: bool = True

    def __post_init__(self) -> None:
        if self.paid not in PAID or self.timing not in TIMINGS or self.customer not in CUSTOMERS \
                or (self.guard and self.paid == "none"):
            raise ValueError(f"bad ReserveSpec {self!r}")


@dataclass
class InjectionPlan:
    footprint: Footprint = field(default_factory=Footprint)
    reserve: tuple[ReserveSpec, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanContext:
    request: InjectionRequest
    base_currency: str
    foreign_currencies: tuple[str, ...]
    timeline: Timeline
    rng: random.Random


@dataclass
class IssueDraft:
    """A truth issue in internal refs; the exporter turns it into truth.json form with final IDs."""

    type: str
    affected: list[Ref]
    expected_amount_minor: int
    currency: str
    evidence_sets: list[list[Ref]]


@dataclass
class InjectionOutcome:
    issues: list[IssueDraft]
    effects: list[dict[str, Any]] = field(default_factory=list)  # {"control", "currency", "delta_minor"}
    description: str = ""
    kind: str | None = None  # provenance "kind"; defaults to the request kind


@dataclass
class ApplyContext:
    world: World
    request: InjectionRequest
    plan: InjectionPlan
    targets: tuple[ReservedTarget, ...]  # built from plan.reserve, in the same order
    recorder: Any  # ChangeRecorder (typed loosely to avoid an import cycle)
    rng: random.Random


class Injector(Protocol):
    def plan(self, ctx: PlanContext) -> InjectionPlan: ...

    def apply(self, ctx: ApplyContext) -> InjectionOutcome: ...

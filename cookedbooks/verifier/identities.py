"""Control checks C1-C6 (DESIGN.md, section 4 and "Точные определения").

Each check returns a list of `Violation`s; an empty list means it holds.
`check_controls` runs all six plus the structural rules collected by
`Books` (control `STRUCT`). The helpers `c4_pairs`, `c5_groups`,
`fx_diffs` and `cutoff_mismatches` are shared with the issue finder, so a
control and the issue type it guards can never disagree.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import timeutil
from .books import Books, FxRow, InvoiceView, PaymentView, Violation, _side, _sorted_ids

CONTROLS = ("C1", "C2", "C3", "C4", "C5", "C6")


def check_structure(books: Books) -> list[Violation]:
    """Structural rules of DESIGN.md section 3 (posting shapes, links, references)."""
    return list(books.structure)


def check_c1(books: Books) -> list[Violation]:
    """Per currency, total debits equal total credits."""
    debit: dict[str, int] = defaultdict(int)
    credit: dict[str, int] = defaultdict(int)
    for row in books.ledger:
        if _side(row) is None:
            continue
        currency = row.get("currency")
        debit[currency] += row["debit_minor"]
        credit[currency] += row["credit_minor"]
    out = []
    for currency in sorted(set(debit) | set(credit), key=str):
        delta = debit[currency] - credit[currency]
        if delta:
            out.append(Violation("C1", "unbalanced_currency", (), f"debits minus credits = {delta}",
                                 delta, currency))
    return out


def check_c2(books: Books) -> list[Violation]:
    """Each invoice has one Dr AR row of total_minor; its Cr AR rows sum to at most total_minor."""
    out = []
    for view in books.invoices.values():
        rows = books.rows_by_iid.get(view.invoice_id, [])
        debits = [r for r in rows if r.get("account") == "AR" and _side(r) == "D"]
        ids = (view.doc_id,)
        if len(debits) != 1:
            out.append(Violation("C2", "dr_ar_count", ids, f"{len(debits)} Dr AR rows instead of one",
                                 None, view.currency))
        elif debits[0]["debit_minor"] != view.total or debits[0].get("currency") != view.currency:
            out.append(Violation("C2", "dr_ar_amount", ids, "Dr AR is not total_minor in the invoice currency",
                                 debits[0]["debit_minor"] - view.total, view.currency))
        credited = sum(r["credit_minor"] for r in rows if r.get("account") == "AR" and _side(r) == "C")
        if credited > view.total:
            out.append(Violation("C2", "over_credited", ids, "Cr AR rows exceed total_minor",
                                 credited - view.total, view.currency))
    return out


@dataclass(frozen=True)
class FxDiff:
    invoice: InvoiceView
    rate: FxRow | None
    correct: int | None
    posted: int
    reason: str | None = None

    @property
    def diff(self) -> int | None:
        return None if self.correct is None else abs(self.posted - self.correct)


def fx_diffs(books: Books) -> list[FxDiff]:
    """Every well-formed foreign-currency invoice with its posted and correct base amounts."""
    out = []
    for view in books.invoices.values():
        if not view.ok or not books.is_foreign(view):
            continue
        rate, reason = books.correct_rate(view)
        correct = books.correct_base(view, rate) if rate is not None else None
        out.append(FxDiff(view, rate, correct, view.posted_base, reason))
    return out


def check_c3(books: Books) -> list[Violation]:
    """Each foreign invoice's REVENUE credit equals the policy conversion of total_minor."""
    out = []
    for item in fx_diffs(books):
        ids = (item.invoice.doc_id,)
        if item.correct is None:
            out.append(Violation("C3", "rate_unavailable", ids, item.reason, None, books.config.base))
        elif item.diff:
            out.append(Violation("C3", "fx_mismatch", ids,
                                 f"posted {item.posted}, correct {item.correct} at {item.rate.key}",
                                 item.diff, books.config.base))
    return out


def open_candidates(books: Books, payment: PaymentView) -> list[InvoiceView]:
    """Open invoices of the payment's client, in its currency, issued no later than the payment."""
    return [
        inv for inv in books.invoices.values()
        if inv.ok and inv.entity_id == payment.entity_id and inv.currency == payment.currency
        and inv.remainder > 0 and inv.issued <= payment.time
    ]


def c4_pairs(books: Books) -> list[tuple[PaymentView, list[InvoiceView]]]:
    """Incoming payments with a remainder, each with its non-empty list of open candidates."""
    out = []
    for payment in books.payments.values():
        if not payment.ok or payment.amount <= 0 or payment.remainder <= 0:
            continue
        candidates = open_candidates(books, payment)
        if candidates:
            out.append((payment, candidates))
    return out


def check_c4(books: Books) -> list[Violation]:
    """No client has an unallocated payment remainder and an open invoice issued no later."""
    out = []
    for payment, candidates in c4_pairs(books):
        for invoice in candidates:
            out.append(Violation("C4", "unallocated_with_open_invoice", _sorted_ids([payment.doc_id, invoice.doc_id]),
                                 f"payment remainder {payment.remainder}, invoice remainder {invoice.remainder}",
                                 payment.remainder, payment.currency))
    return out


def duplicate_key_of(books: Books, view: InvoiceView) -> tuple | None:
    """The obligation key, or None if any of its fields is empty (then no duplicate is possible)."""
    values = []
    for name in books.config.duplicate_key:
        value = view.record.get(name)
        if value is None:
            return None
        # Type-tagged so that 1, True and "1" never collide.
        values.append((type(value).__name__, repr(value)))
    return tuple(values)


def c5_groups(books: Books) -> list[list[InvoiceView]]:
    """Groups of two or more invoices sharing one obligation key, in file order."""
    groups: dict[tuple, list[InvoiceView]] = defaultdict(list)
    for view in books.invoices.values():
        key = duplicate_key_of(books, view)
        if key is not None:
            groups[key].append(view)
    return [group for group in groups.values() if len(group) > 1]


def check_c5(books: Books) -> list[Violation]:
    """No two invoices share an obligation key (policy.duplicate_key)."""
    return [
        Violation("C5", "duplicate_key", _sorted_ids(v.doc_id for v in group),
                  f"{len(group)} invoices with one obligation key", group[0].total, group[0].currency)
        for group in c5_groups(books)
    ]


def cutoff_mismatches(books: Books) -> list[InvoiceView]:
    """Well-formed invoices whose issued_at and posting lie on different sides of the period."""
    cfg = books.config
    return [
        view for view in books.invoices.values()
        if view.ok and view.posting_time is not None
        and timeutil.period_side(view.issued, cfg.period_start, cfg.period_end)
        != timeutil.period_side(view.posting_time, cfg.period_start, cfg.period_end)
    ]


def check_c6(books: Books) -> list[Violation]:
    """Each invoice's posting lies on the same side of the period bounds as its issued_at."""
    return [
        Violation("C6", "cutoff", (view.doc_id,), "issued_at and posting on different sides of the period",
                  view.total, view.currency)
        for view in cutoff_mismatches(books)
    ]


CHECKS = {"C1": check_c1, "C2": check_c2, "C3": check_c3, "C4": check_c4, "C5": check_c5, "C6": check_c6}


def check_controls(books: Books) -> list[Violation]:
    """STRUCT violations followed by C1..C6 violations."""
    out = check_structure(books)
    for name in CONTROLS:
        out.extend(CHECKS[name](books))
    return out

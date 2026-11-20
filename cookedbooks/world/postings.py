"""Ledger rows for each kind of event, exactly as in DESIGN.md section 3.

The functions build rows; they do not store them. The simulator stores them
directly, injectors store them through the change recorder.
"""

from __future__ import annotations

from cookedbooks.world.model import Invoice, LedgerRow, Payment, World

BACKGROUND_ACCOUNTS = ("RENT_EXPENSE", "PAYROLL_EXPENSE", "UTILITIES_EXPENSE")


def _row(world: World, posting: str, ts: int, account: str, debit: int, credit: int, currency: str,
         customer: str | None, invoice: str | None, payment: str | None) -> LedgerRow:
    if (debit > 0) == (credit > 0) or min(debit, credit) != 0:
        raise ValueError(f"a ledger row needs exactly one positive side, got {debit}/{credit}")
    return LedgerRow(
        ref=world.new_ref("L"), posting=posting, ts_utc=ts, ts_offset=world.timeline.offset,
        account=account, debit_minor=debit, credit_minor=credit, currency=currency,
        customer=customer, invoice=invoice, payment=payment,
    )


def invoice_posting(world: World, invoice: Invoice, ts: int, *, base_minor: int | None = None) -> list[LedgerRow]:
    """Dr AR / Cr REVENUE; for a foreign invoice the four-row FX_CLEARING form.

    `base_minor` overrides the converted amount (an fx_error injector uses it);
    by default it is the policy conversion.
    """
    posting = world.new_ref("T")
    c, i = invoice.customer, invoice.ref
    total, cur, base = invoice.total_minor, invoice.currency, world.base_currency
    if cur == base:
        return [
            _row(world, posting, ts, "AR", total, 0, cur, c, i, None),
            _row(world, posting, ts, "REVENUE", 0, total, base, c, i, None),
        ]
    converted = world.correct_base_amount(invoice) if base_minor is None else base_minor
    return [
        _row(world, posting, ts, "AR", total, 0, cur, c, i, None),
        _row(world, posting, ts, "FX_CLEARING", 0, total, cur, c, i, None),
        _row(world, posting, ts, "FX_CLEARING", converted, 0, base, c, i, None),
        _row(world, posting, ts, "REVENUE", 0, converted, base, c, i, None),
    ]


def payment_posting(world: World, payment: Payment, allocations: list[tuple[str, int]]) -> list[LedgerRow]:
    """Incoming payment: Dr CASH in full; Cr AR per allocated invoice; the rest Cr CUSTOMER_ADVANCES."""
    if payment.amount_minor <= 0:
        raise ValueError("payment_posting needs a positive amount; use refund_posting")
    posting = world.new_ref("T")
    c, p, cur, ts = payment.customer, payment.ref, payment.currency, payment.ts_utc
    rows = [_row(world, posting, ts, "CASH", payment.amount_minor, 0, cur, c, None, p)]
    allocated = 0
    for invoice_ref, amount in allocations:
        rows.append(_row(world, posting, ts, "AR", 0, amount, cur, c, invoice_ref, p))
        allocated += amount
    rest = payment.amount_minor - allocated
    if rest < 0:
        raise ValueError("allocations exceed the payment amount")
    if rest > 0:
        rows.append(_row(world, posting, ts, "CUSTOMER_ADVANCES", 0, rest, cur, c, None, p))
    return rows


def refund_posting(world: World, payment: Payment) -> list[LedgerRow]:
    """Refund (negative amount): Dr CUSTOMER_ADVANCES / Cr CASH on the absolute amount."""
    if payment.amount_minor >= 0:
        raise ValueError("refund_posting needs a negative amount")
    posting = world.new_ref("T")
    amount = -payment.amount_minor
    c, p, cur, ts = payment.customer, payment.ref, payment.currency, payment.ts_utc
    return [
        _row(world, posting, ts, "CUSTOMER_ADVANCES", amount, 0, cur, c, None, p),
        _row(world, posting, ts, "CASH", 0, amount, cur, c, None, p),
    ]


def background_posting(world: World, ts: int, lines: list[tuple[str, int]]) -> list[LedgerRow]:
    """Expense lines debited, one CASH credit for their sum, in the base currency, no references."""
    posting = world.new_ref("T")
    base = world.base_currency
    rows = [_row(world, posting, ts, account, amount, 0, base, None, None, None) for account, amount in lines]
    rows.append(_row(world, posting, ts, "CASH", 0, sum(a for _, a in lines), base, None, None, None))
    return rows

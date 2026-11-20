"""Ready-made books for verifier tests, built with tests.helpers.books_builder."""

from __future__ import annotations

from tests.helpers.books_builder import Book

FX_USD = {
    "2026-07-01": "468.00", "2026-08-13": "470.25", "2026-08-14": "475.50", "2026-08-17": "480.10",
    "2026-09-29": "481.00", "2026-09-30": "482.00", "2026-10-01": "490.00", "2026-10-02": "491.00",
}
FX_EUR = {"2026-08-14": "520.00", "2026-09-30": "530.00"}


def with_fx(book: Book) -> Book:
    for day, rate in FX_USD.items():
        book.add_fx(day, "USD", rate)
    for day, rate in FX_EUR.items():
        book.add_fx(day, "EUR", rate)
    return book


def lookalike_book(**kwargs) -> tuple[Book, dict]:
    """A correct book containing every legitimate look-alike of DESIGN.md section 5."""
    book = with_fx(Book(**kwargs))
    h: dict = {}
    # Recurring monthly invoices: same client, PO and amount, different service_period.
    h["rec1"] = book.invoice("1", "100", "c1", "2026-07-10T10:00:00+05:00", 150000, po_ref="PO-1",
                             service_period="2026-07")
    h["rec2"] = book.invoice("2", "101", "c1", "2026-08-10T10:00:00+05:00", 150000, po_ref="PO-1",
                             service_period="2026-08")
    book.payment("31", "500", "c1", "2026-07-20T10:00:00+05:00", 150000, allocations={"100": 150000})
    book.payment("32", "501", "c1", "2026-08-20T10:00:00+05:00", 150000, allocations={"101": 150000})
    # Same client, amount and month but no PO: not an obligation key, not a duplicate.
    book.invoice("3", "102", "c1", "2026-08-11T10:00:00+05:00", 77700, service_period="2026-08")
    book.invoice("4", "103", "c1", "2026-08-12T10:00:00+05:00", 77700, service_period="2026-08")
    # Split payment: one invoice paid by two payments, each fully allocated.
    book.invoice("5", "104", "c2", "2026-07-15T09:00:00+05:00", 100000, po_ref="PO-2", service_period="2026-07")
    book.payment("33", "502", "c2", "2026-07-25T09:00:00+05:00", 60000, allocations={"104": 60000})
    book.payment("34", "503", "c2", "2026-08-05T09:00:00+05:00", 40000, allocations={"104": 40000})
    # One payment for two invoices.
    book.invoice("6", "105", "c3", "2026-07-16T09:00:00+05:00", 20000)
    book.invoice("7", "106", "c3", "2026-07-17T09:00:00+05:00", 30000)
    book.payment("35", "504", "c3", "2026-07-30T09:00:00+05:00", 50000, allocations={"105": 20000, "106": 30000})
    # Genuine advance: no open invoice at payment time; a later invoice stays open.
    book.payment("36", "505", "c4", "2026-07-05T12:00:00+05:00", 50000)
    book.invoice("8", "107", "c4", "2026-07-20T12:00:00+05:00", 50000)
    # Refunded double payment.
    book.invoice("9", "108", "c5", "2026-08-01T12:00:00+05:00", 30000)
    book.payment("37", "506", "c5", "2026-08-03T12:00:00+05:00", 30000, allocations={"108": 30000})
    book.payment("38", "507", "c5", "2026-08-04T12:00:00+05:00", 30000)
    book.refund("39", "508", "c5", "2026-08-06T12:00:00+05:00", 30000)
    # 21:00 UTC is already the next local day: the next day's rate is right.
    h["utc"] = book.invoice("10", "109", "c6", "2026-08-13T21:00:00+00:00", 100000, currency="USD")
    book.payment("40", "509", "c6", "2026-08-25T10:00:00+05:00", 100000, currency="USD",
                 allocations={"109": 100000})
    # Saturday invoice at Friday's rate (no Saturday row in fx.csv).
    h["weekend"] = book.invoice("11", "110", "c6", "2026-08-15T11:00:00+05:00", 250000, currency="USD",
                                posted_at="2026-08-16T11:00:00+05:00")
    # Delayed posting that stays inside the period.
    h["delayed"] = book.invoice("12", "111", "c7", "2026-08-10T10:00:00+05:00", 90000,
                                posted_at="2026-08-12T10:00:00+05:00")
    # Next-period invoice posted in the next period.
    book.invoice("13", "112", "c7", "2026-10-02T10:00:00+05:00", 91000, posted_at="2026-10-03T10:00:00+05:00")
    # UTC date in the period, local date after it; posted after it too.
    book.invoice("14", "113", "c7", "2026-09-30T19:30:00+00:00", 92000, posted_at="2026-10-01T09:00:00+05:00")
    # Before the period, posted before it.
    book.invoice("15", "114", "c7", "2026-06-25T10:00:00+05:00", 93000, posted_at="2026-06-27T10:00:00+05:00")
    # Background postings (no documents).
    book.background("2026-07-01T09:00:00+05:00", "RENT_EXPENSE", 400000)
    book.background("2026-07-31T09:00:00+05:00", "PAYROLL_EXPENSE", 900000)
    book.background("2026-08-31T09:00:00+05:00", "UTILITIES_EXPENSE", 12345)
    return book, h

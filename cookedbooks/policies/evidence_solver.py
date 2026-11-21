"""evidence_solver: a scripted auditor that finds errors by honest analysis through the tools.

It reads every table page by page with `list_records` (that is its whole
inspection cost), then applies the four rules of DESIGN.md ("Точные
определения") to what the first observation and the pages returned:

- duplicate_invoice: two invoices with equal, non-empty values of every field
  in `policy.duplicate_key`;
- unallocated_payment: an incoming payment with a positive unallocated
  remainder while exactly one open invoice of the same customer and currency,
  issued no later than the payment, exists;
- fx_error: the posted base amount of a foreign-currency invoice differs from
  the conversion at the rate of the local date of `issued_at` (or the previous
  available date) by at least `policy.fx_materiality`;
- cutoff_error: `issued_at` and the invoice posting lie on different sides of
  a period bound.

Exact arithmetic only (`int`, `Fraction`). IDs are opaque strings: they are
only compared and used as dictionary keys, never parsed or sorted, and row
order is never relied on. Anything that does not fit a rule exactly is not
reported. Standard library only; imports nothing from `cookedbooks`; no I/O.

Interface: `make_policy(seed, **options)` returns an object whose
`act(observation)` returns the next action.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from typing import Any

PAGE_LIMIT = 50
DEFAULT_DUPLICATE_KEY = ("entity_id", "po_ref", "service_period", "currency", "total_minor")
DEFAULT_MATERIALITY = {"min_relative": "0.005", "min_minor": 100}

Record = dict[str, Any]
Finding = dict[str, Any]


# --- small parsing helpers ------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str(value: Any) -> bool:
    return isinstance(value, str)


def _empty(value: Any) -> bool:
    return value is None or value == ""


def parse_offset(text: Any) -> timezone | None:
    """'+05:00' / '-05:00' / 'Z' / '+00:00' -> a fixed timezone, or None."""
    if not _is_str(text):
        return None
    if text in ("Z", "z"):
        return timezone.utc
    if len(text) != 6 or text[0] not in "+-" or text[3] != ":":
        return None
    try:
        hours, minutes = int(text[1:3]), int(text[4:6])
    except ValueError:
        return None
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(delta if text[0] == "+" else -delta)


def parse_instant(text: Any) -> datetime | None:
    """An ISO-8601 time with an offset as an aware datetime; None if unreadable or naive."""
    if not _is_str(text):
        return None
    value = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def parse_fraction(value: Any) -> Fraction | None:
    """A decimal string or an int as an exact Fraction (floats are refused)."""
    if _is_int(value):
        return Fraction(value)
    if not _is_str(value):
        return None
    try:
        return Fraction(value.strip())
    except (ValueError, ZeroDivisionError):
        return None


def round_fraction(value: Fraction, mode: str) -> int | None:
    """Round to an integer: 'half_up' (ties away from zero) or 'half_even'; None for other modes."""
    if mode == "half_even":
        return round(value)
    if mode == "half_up":
        sign = -1 if value < 0 else 1
        magnitude = abs(value)
        floor = magnitude.numerator // magnitude.denominator
        if magnitude - floor >= Fraction(1, 2):
            floor += 1
        return sign * floor
    return None


def convert(src_minor: int, rate: Fraction, exp_src: int, exp_base: int, mode: str) -> int | None:
    """base_minor = round(src_minor * rate * 10^(exp_base - exp_src)) (DESIGN.md section 3)."""
    return round_fraction(Fraction(src_minor) * rate * Fraction(10) ** (exp_base - exp_src), mode)


# --- the episode as the solver sees it -------------------------------------------------


@dataclass
class Books:
    """Everything the solver read: the first observation plus the pages of each table."""

    manifest: dict
    policy: dict
    tables: dict[str, list[Record]] = field(default_factory=dict)
    complete: set[str] = field(default_factory=set)


@dataclass
class Context:
    """Parsed manifest and policy values the rules need."""

    base: str
    exponents: dict[str, int]
    company_tz: timezone | None
    period_start: datetime | None
    period_end: datetime | None
    inclusive: bool
    rounding: str | None
    duplicate_key: tuple[str, ...]
    min_minor: int | None
    min_relative: Fraction | None
    rate_date_rule: str | None
    missing_rate_rule: str | None


def make_context(manifest: dict, policy: dict) -> Context:
    manifest = manifest if isinstance(manifest, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    exponents = {
        code: exp for code, exp in (manifest.get("currency_exponents") or {}).items()
        if _is_str(code) and _is_int(exp)
    } if isinstance(manifest.get("currency_exponents"), dict) else {}
    rounding = policy.get("rounding")
    if not _is_str(rounding):
        manifest_rounding = manifest.get("rounding")
        rounding = manifest_rounding.get("mode") if isinstance(manifest_rounding, dict) else None
    key = policy.get("duplicate_key")
    if not (isinstance(key, list) and key and all(_is_str(k) for k in key)):
        key = list(DEFAULT_DUPLICATE_KEY)
    materiality = policy.get("fx_materiality")
    if not isinstance(materiality, dict):
        materiality = DEFAULT_MATERIALITY
    min_minor = materiality.get("min_minor", DEFAULT_MATERIALITY["min_minor"])
    min_relative = parse_fraction(materiality.get("min_relative", DEFAULT_MATERIALITY["min_relative"]))
    return Context(
        base=manifest.get("base_currency") if _is_str(manifest.get("base_currency")) else "",
        exponents=exponents,
        company_tz=parse_offset(manifest.get("utc_offset")),
        period_start=parse_instant(manifest.get("period_start")),
        period_end=parse_instant(manifest.get("period_end")),
        inclusive=policy.get("period_bounds", "inclusive") != "exclusive",
        rounding=rounding if _is_str(rounding) else None,
        duplicate_key=tuple(key),
        min_minor=min_minor if _is_int(min_minor) else None,
        min_relative=min_relative,
        rate_date_rule=policy.get("fx_rate_date", "document_date"),
        missing_rate_rule=policy.get("fx_missing_rate", "previous_available"),
    )


def _side(moment: datetime, ctx: Context) -> int | None:
    """-1 before the period, 0 inside, 1 after; None if the bounds are unknown."""
    if ctx.period_start is None or ctx.period_end is None:
        return None
    if ctx.inclusive:
        if moment < ctx.period_start:
            return -1
        return 1 if moment > ctx.period_end else 0
    if moment <= ctx.period_start:
        return -1
    return 1 if moment >= ctx.period_end else 0


@dataclass
class Index:
    """Lookups over the ledger, built once."""

    invoice_posting: dict[str, list[Record]]   # invoice_id -> rows with that invoice_id and no payment_id
    payment_rows: dict[str, list[Record]]      # payment_id -> rows with that payment_id
    ar_credits: dict[str, int]                 # invoice_id -> sum of its Cr AR rows


def build_index(ledger: list[Record]) -> Index:
    invoice_posting: dict[str, list[Record]] = defaultdict(list)
    payment_rows: dict[str, list[Record]] = defaultdict(list)
    ar_credits: dict[str, int] = defaultdict(int)
    for row in ledger:
        invoice_id, payment_id = row.get("invoice_id"), row.get("payment_id")
        if _is_str(payment_id) and payment_id != "":
            payment_rows[payment_id].append(row)
        elif _is_str(invoice_id) and invoice_id != "":
            invoice_posting[invoice_id].append(row)
        credit = row.get("credit_minor")
        if row.get("account") == "AR" and _is_str(invoice_id) and invoice_id != "" and _is_int(credit) and credit > 0:
            ar_credits[invoice_id] += credit
    return Index(dict(invoice_posting), dict(payment_rows), dict(ar_credits))


def _single(rows: list[Record], account: str, side: str, currency: str) -> Record | None:
    """The only row of `rows` on `account` with a positive `side` amount in `currency`, else None."""
    found = [r for r in rows if r.get("account") == account and r.get("currency") == currency
             and _is_int(r.get(side)) and r[side] > 0]
    return found[0] if len(found) == 1 else None


def _posting_time(rows: list[Record]) -> datetime | None:
    """The common time of an invoice posting; None if the rows disagree or are unreadable."""
    moments = {parse_instant(r.get("timestamp")) for r in rows}
    if len(moments) != 1 or None in moments:
        return None
    return next(iter(moments))


# --- the four rules ---------------------------------------------------------------------


def find_duplicates(invoices: list[Record], ctx: Context) -> list[Finding]:
    """Pairs of invoices with equal, non-empty values of every duplicate_key field."""
    groups: dict[tuple, list[Record]] = defaultdict(list)
    order: list[tuple] = []
    for inv in invoices:
        values = tuple(inv.get(name) for name in ctx.duplicate_key)
        if any(_empty(v) for v in values) or not _is_str(inv.get("doc_id")):
            continue
        try:
            key = tuple((type(v).__name__, v) for v in values)
            hash(key)
        except TypeError:
            continue
        if key not in groups:
            order.append(key)
        groups[key].append(inv)
    findings = []
    for key in order:
        group = groups[key]
        if len(group) != 2:  # three or more never occur by the rules; one is not a duplicate
            continue
        first, second = group
        if first["doc_id"] == second["doc_id"] or not _is_int(first.get("total_minor")) \
                or not _is_str(first.get("currency")):
            continue
        ids = [first["doc_id"], second["doc_id"]]
        findings.append(_finding(
            "duplicate_invoice", ids, first["total_minor"], first["currency"], ids,
            f"Invoices {ids[0]} and {ids[1]} have the same "
            + ", ".join(ctx.duplicate_key) + ": one obligation invoiced twice.",
        ))
    return findings


def find_unallocated(invoices: list[Record], payments: list[Record], index: Index) -> list[Finding]:
    """Incoming payments with a remainder in advances while exactly one matching invoice is open."""
    open_by_customer: dict[tuple[str, str], list[tuple[datetime, Record, int]]] = defaultdict(list)
    for inv in invoices:
        invoice_id, total = inv.get("invoice_id"), inv.get("total_minor")
        issued = parse_instant(inv.get("issued_at"))
        if not _is_str(invoice_id) or not _is_int(total) or issued is None or not _is_str(inv.get("doc_id")):
            continue
        remaining = total - index.ar_credits.get(invoice_id, 0)
        if remaining > 0:
            open_by_customer[(inv.get("entity_id"), inv.get("currency"))].append((issued, inv, remaining))

    findings = []
    for pay in payments:
        payment_id, amount = pay.get("payment_id"), pay.get("amount_minor")
        paid_at = parse_instant(pay.get("timestamp"))
        if not _is_str(payment_id) or not _is_int(amount) or amount <= 0 or paid_at is None \
                or not _is_str(pay.get("doc_id")):
            continue
        rows = index.payment_rows.get(payment_id, [])
        customers = {r.get("entity_id") for r in rows}
        if len(customers) != 1:
            continue
        customer = next(iter(customers))
        if not _is_str(customer) or customer == "":
            continue
        allocated = sum(r["credit_minor"] for r in rows if r.get("account") == "AR"
                        and _is_int(r.get("credit_minor")) and r["credit_minor"] > 0)
        remainder = amount - allocated
        if remainder <= 0:
            continue
        candidates = [(inv, rest) for issued, inv, rest in open_by_customer.get((customer, pay.get("currency")), [])
                      if issued <= paid_at]
        if len(candidates) != 1:  # none: a genuine advance; several: not decidable by the rule
            continue
        inv, rest = candidates[0]
        ids = [pay["doc_id"], inv["doc_id"]]
        findings.append(_finding(
            "unallocated_payment", ids, remainder, pay.get("currency"), ids,
            f"Payment {ids[0]} leaves {remainder} minor units in CUSTOMER_ADVANCES while invoice "
            f"{ids[1]} of the same customer and currency, issued before it, is still open "
            f"({rest} minor units unpaid).",
        ))
    return findings


@dataclass
class RateTable:
    """fx rows per (source, target) pair, sorted by date (a natural key, not an ID)."""

    dates: dict[tuple[str, str], list[str]]
    rows: dict[tuple[str, str], dict[str, list[Record]]]

    @classmethod
    def build(cls, fx: list[Record]) -> "RateTable":
        rows: dict[tuple[str, str], dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
        for rec in fx:
            day, src, tgt = rec.get("date"), rec.get("source_currency"), rec.get("target_currency")
            if _is_str(day) and _is_str(src) and _is_str(tgt):
                rows[(src, tgt)][day].append(rec)
        return cls({pair: sorted(days) for pair, days in rows.items()}, rows)

    def lookup(self, pair: tuple[str, str], day: str, previous: bool) -> Record | None:
        """The rate record for `day`, or the latest earlier one when `previous`; None if absent or ambiguous."""
        days = self.dates.get(pair)
        if not days:
            return None
        pos = bisect_right(days, day)
        if pos == 0:
            return None
        chosen = days[pos - 1]
        if chosen != day and not previous:
            return None
        records = self.rows[pair][chosen]
        return records[0] if len(records) == 1 else None


def _fx_key(rec: Record) -> str:
    key = rec.get("id")
    if _is_str(key):
        return key
    return f"{rec['date']}:{rec['source_currency']}:{rec['target_currency']}"


def find_fx_errors(invoices: list[Record], fx: list[Record], index: Index, ctx: Context) -> list[Finding]:
    """Foreign-currency invoices whose posted base amount misses the correct one materially."""
    if ctx.rate_date_rule != "document_date" or ctx.company_tz is None or ctx.rounding is None \
            or ctx.min_minor is None or ctx.min_relative is None or ctx.base not in ctx.exponents:
        return []
    previous = ctx.missing_rate_rule == "previous_available"
    rates = RateTable.build(fx)
    findings = []
    for inv in invoices:
        currency, total, invoice_id = inv.get("currency"), inv.get("total_minor"), inv.get("invoice_id")
        issued = parse_instant(inv.get("issued_at"))
        if not _is_str(currency) or currency == ctx.base or currency not in ctx.exponents \
                or not _is_int(total) or not _is_str(invoice_id) or issued is None or not _is_str(inv.get("doc_id")):
            continue
        revenue = _single(index.invoice_posting.get(invoice_id, []), "REVENUE", "credit_minor", ctx.base)
        if revenue is None or not _is_str(revenue.get("row_id")):
            continue
        local_day = issued.astimezone(ctx.company_tz).date().isoformat()
        rate_rec = rates.lookup((currency, ctx.base), local_day, previous)
        rate = parse_fraction(rate_rec.get("rate")) if rate_rec is not None else None
        if rate is None:
            continue
        correct = convert(total, rate, ctx.exponents[currency], ctx.exponents[ctx.base], ctx.rounding)
        if correct is None:
            continue
        posted = revenue["credit_minor"]
        diff = abs(posted - correct)
        if diff == 0 or diff < ctx.min_minor or Fraction(diff) < ctx.min_relative * abs(correct):
            continue
        fx_key = _fx_key(rate_rec)
        findings.append(_finding(
            "fx_error", [inv["doc_id"]], diff, ctx.base, [inv["doc_id"], revenue["row_id"], fx_key],
            f"Invoice {inv['doc_id']} ({total} {currency}, local date {local_day}) should convert at "
            f"{fx_key} = {rate_rec.get('rate')} to {correct} {ctx.base}; REVENUE row {revenue['row_id']} "
            f"posts {posted}. Difference {diff}.",
        ))
    return findings


def find_cutoff_errors(invoices: list[Record], index: Index, ctx: Context) -> list[Finding]:
    """Invoices posted on the other side of a period bound than their issued_at."""
    findings = []
    for inv in invoices:
        invoice_id, currency, total = inv.get("invoice_id"), inv.get("currency"), inv.get("total_minor")
        issued = parse_instant(inv.get("issued_at"))
        if not _is_str(invoice_id) or not _is_str(currency) or not _is_int(total) or issued is None \
                or not _is_str(inv.get("doc_id")):
            continue
        rows = index.invoice_posting.get(invoice_id, [])
        posted_at = _posting_time(rows) if rows else None
        if posted_at is None:
            continue
        issued_side, posted_side = _side(issued, ctx), _side(posted_at, ctx)
        if issued_side is None or issued_side == posted_side:
            continue
        line = _single(rows, "AR", "debit_minor", currency)
        if line is None or line.get("debit_minor") != total:
            line = _single(rows, "REVENUE", "credit_minor", ctx.base)
        if line is None or not _is_str(line.get("row_id")):
            continue
        where = {-1: "before the period", 0: "inside the period", 1: "after the period"}
        findings.append(_finding(
            "cutoff_error", [inv["doc_id"]], total, currency, [inv["doc_id"], line["row_id"]],
            f"Invoice {inv['doc_id']} is issued {where[issued_side]} ({inv.get('issued_at')}) but posted "
            f"{where[posted_side]} ({rows[0].get('timestamp')}), ledger row {line['row_id']}.",
        ))
    return findings


def _finding(kind: str, affected: list[str], amount: int, currency: str, evidence: list[str],
             explanation: str) -> Finding:
    return {
        "type": kind,
        "affected_ids": list(affected),
        "claimed_amount_minor": int(amount),
        "currency": currency,
        "evidence_ids": list(dict.fromkeys(evidence))[:10],
        "explanation": explanation[:1000],
    }


def analyse(books: Books) -> list[Finding]:
    """All findings the rules support on what was read. Rules whose tables are incomplete are skipped."""
    ctx = make_context(books.manifest, books.policy)
    done = books.complete
    invoices = books.tables.get("invoices", [])
    payments = books.tables.get("payments", [])
    ledger = books.tables.get("ledger", [])
    fx = books.tables.get("fx", [])
    if "invoices" not in done:
        return []
    findings = find_duplicates(invoices, ctx)
    if "ledger" in done:
        index = build_index(ledger)
        if "payments" in done:
            findings += find_unallocated(invoices, payments, index)
        if "fx" in done:
            findings += find_fx_errors(invoices, fx, index, ctx)
        findings += find_cutoff_errors(invoices, index, ctx)
    return findings[:30]


# --- the policy ---------------------------------------------------------------------------


class EvidenceSolver:
    """Pages through invoices, payments, ledger and (when needed) fx, then submits the analysis."""

    TABLE_ORDER = ("invoices", "payments", "ledger", "fx")

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.books: Books | None = None
        self.queue: list[str] = []
        self.current: str | None = None
        self.findings: list[Finding] | None = None
        self.pages_read = 0

    def act(self, observation: dict) -> dict:
        if self.books is None:
            return self._start(observation)
        self._absorb(observation)
        remaining = observation.get("remaining_steps") if isinstance(observation, dict) else None
        if self.current is None or (_is_int(remaining) and remaining <= 1):
            return self._submit()
        return self._page(self._cursor)

    def _start(self, observation: dict) -> dict:
        manifest = observation.get("manifest") if isinstance(observation, dict) else None
        policy = observation.get("policy") if isinstance(observation, dict) else None
        self.books = Books(manifest if isinstance(manifest, dict) else {}, policy if isinstance(policy, dict) else {})
        self.queue = list(self.TABLE_ORDER)
        self._next_table()
        return self._page(None)

    def _next_table(self) -> None:
        self.current = None
        self._cursor = None
        while self.queue:
            table = self.queue.pop(0)
            if table == "fx" and not self._needs_fx():
                self.books.complete.add("fx")
                continue
            self.current = table
            self.books.tables[table] = []
            return

    def _needs_fx(self) -> bool:
        """fx is read only when some invoice is in a foreign currency."""
        base = self.books.manifest.get("base_currency")
        return any(inv.get("currency") != base for inv in self.books.tables.get("invoices", []))

    def _absorb(self, observation: dict) -> None:
        result = observation.get("result") if isinstance(observation, dict) else None
        if self.current is None:
            return
        if not isinstance(result, dict) or not isinstance(result.get("records"), list):
            self._next_table()  # an unexpected error: leave this table incomplete
            return
        self.pages_read += 1
        self.books.tables[self.current].extend(r for r in result["records"] if isinstance(r, dict))
        cursor = result.get("next_cursor")
        if _is_str(cursor):
            self._cursor = cursor
            return
        self.books.complete.add(self.current)
        self._next_table()

    _cursor: str | None = None

    def _page(self, cursor: str | None) -> dict:
        return {"name": "list_records", "args": {"table": self.current, "cursor": cursor, "limit": PAGE_LIMIT}}

    def _submit(self) -> dict:
        self.findings = analyse(self.books)
        episode_id = self.books.manifest.get("episode_id")
        return {"name": "submit", "args": {"findings": {
            "episode_id": episode_id if _is_str(episode_id) else "",
            "findings": self.findings,
        }}}


def make_policy(seed: int, **options: Any) -> EvidenceSolver:
    """A fresh solver. It is deterministic and ignores `seed` and `options`."""
    return EvidenceSolver(seed)

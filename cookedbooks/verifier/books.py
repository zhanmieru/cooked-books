"""Indexed view of one set of books, plus the structural rules of DESIGN.md section 3.

`Books` takes the manifest/policy-derived `Config` and four tables (lists of
records in the loader's shape) and builds per-document views:

- `InvoiceView`: the invoice, its posting (all rows with its `invoice_id`
  and an empty `payment_id`) and the `Cr AR` rows paid against it;
- `PaymentView`: the payment and all rows with its `payment_id`.

Every deviation from the posting shapes of DESIGN.md section 3 and
"Строки книги и документы" becomes a `Violation` with control `STRUCT`, and
the affected view is marked `ok = False` so that predicates skip it.

IDs are opaque: they are only used as exact dictionary keys.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from typing import Any

from . import money, timeutil

EXPENSE_ACCOUNTS = frozenset({"RENT_EXPENSE", "PAYROLL_EXPENSE", "UTILITIES_EXPENSE"})
ACCOUNTS = frozenset({"AR", "REVENUE", "CASH", "CUSTOMER_ADVANCES", "FX_CLEARING"}) | EXPENSE_ACCOUNTS

POLICY_DEFAULTS: dict[str, Any] = {
    "fx_rate_date": "document_date",
    "fx_missing_rate": "previous_available",
    "fx_direction": "source_to_target",
    "period_bounds": "inclusive",
    "cutoff_basis": "issued_at_local",
    "allocation_rule": "open_invoices_first",
    "duplicate_key": ["entity_id", "po_ref", "service_period", "currency", "total_minor"],
    "fx_materiality": {"min_relative": "0.005", "min_minor": 100},
}
# The only value of each rule that DESIGN.md defines. Anything else is undecidable for us.
SUPPORTED_POLICY = {
    key: value for key, value in POLICY_DEFAULTS.items()
    if key not in ("duplicate_key", "fx_materiality")
}


@dataclass(frozen=True)
class Violation:
    """One failed control check (C1..C6) or structural rule (STRUCT)."""

    control: str
    code: str
    ids: tuple[str, ...]
    message: str
    amount_minor: int | None = None
    currency: str | None = None

    def signature(self) -> tuple:
        """What identifies the violation independently of its amount."""
        return (self.control, self.code, self.ids, self.currency)

    def to_dict(self) -> dict:
        return {
            "control": self.control, "code": self.code, "ids": list(self.ids),
            "amount_minor": self.amount_minor, "currency": self.currency, "message": self.message,
        }


def _sorted_ids(ids) -> tuple[str, ...]:
    """Sort for stable output only; never for meaning."""
    return tuple(sorted(set(ids)))


@dataclass(frozen=True)
class Config:
    """Everything the rules need from manifest.json and policy.json, parsed exactly."""

    base: str
    exponents: dict[str, int]
    offset: int
    period_start: int
    period_end: int
    rounding: str
    min_minor: int
    min_relative: Fraction
    duplicate_key: tuple[str, ...]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def config_from(manifest: dict, policy: dict) -> tuple[Config | None, list[str]]:
    """Parse the parts of manifest and policy the rules depend on.

    Returns (config, problems). config is None when the rules cannot be
    applied at all (missing base currency, bad period, unsupported policy).
    """
    problems: list[str] = []
    base = manifest.get("base_currency")
    if not isinstance(base, str):
        problems.append("manifest.base_currency is not a string")
    exponents = manifest.get("currency_exponents")
    if not isinstance(exponents, dict) or not all(
        isinstance(k, str) and _is_int(v) and v >= 0 for k, v in exponents.items()
    ):
        problems.append("manifest.currency_exponents is not an object of non-negative integers")
        exponents = None
    elif isinstance(base, str) and base not in exponents:
        problems.append("manifest.currency_exponents has no exponent for the base currency")
    try:
        offset = timeutil.parse_offset(manifest.get("utc_offset"))
    except ValueError as exc:
        problems.append(f"manifest.utc_offset: {exc}")
        offset = None
    bounds = []
    for name in ("period_start", "period_end"):
        try:
            bounds.append(timeutil.parse_instant(manifest.get(name)))
        except ValueError as exc:
            problems.append(f"manifest.{name}: {exc}")
    if len(bounds) == 2 and bounds[0] > bounds[1]:
        problems.append("manifest.period_start is after period_end")

    manifest_rounding = manifest.get("rounding")
    manifest_mode = manifest_rounding.get("mode") if isinstance(manifest_rounding, dict) else None
    policy_mode = policy.get("rounding")
    if policy_mode is not None and manifest_mode is not None and policy_mode != manifest_mode:
        problems.append(f"policy.rounding {policy_mode!r} contradicts manifest.rounding.mode {manifest_mode!r}")
    rounding = policy_mode if policy_mode is not None else (manifest_mode or "half_up")
    if rounding not in money.ROUNDING_MODES:
        problems.append(f"unsupported rounding mode {rounding!r}")

    for key, supported in SUPPORTED_POLICY.items():
        value = policy.get(key, supported)
        if value != supported:
            problems.append(f"policy.{key} = {value!r} is not a rule the verifier knows ({supported!r})")

    duplicate_key = policy.get("duplicate_key", POLICY_DEFAULTS["duplicate_key"])
    if not (isinstance(duplicate_key, list) and duplicate_key and all(isinstance(k, str) for k in duplicate_key)):
        problems.append("policy.duplicate_key is not a non-empty list of field names")
        duplicate_key = None
    materiality = policy.get("fx_materiality", POLICY_DEFAULTS["fx_materiality"])
    min_minor = min_relative = None
    if isinstance(materiality, dict):
        min_minor = materiality.get("min_minor", 0)
        if not _is_int(min_minor) or min_minor < 0:
            problems.append("policy.fx_materiality.min_minor is not a non-negative integer")
            min_minor = None
        try:
            min_relative = money.parse_decimal(materiality.get("min_relative", "0"))
        except ValueError:
            problems.append("policy.fx_materiality.min_relative is not a decimal string")
    else:
        problems.append("policy.fx_materiality is not an object")

    tolerance = policy.get("amount_tolerance_minor", {})
    if not (isinstance(tolerance, dict) and all(_is_int(v) and v >= 0 for v in tolerance.values())):
        problems.append("policy.amount_tolerance_minor is not an object of non-negative integers")

    if problems:
        return None, problems
    return Config(
        base=base, exponents=dict(exponents), offset=offset, period_start=bounds[0],
        period_end=bounds[1], rounding=rounding, min_minor=min_minor,
        min_relative=min_relative, duplicate_key=tuple(duplicate_key),
    ), []


def _side(row: dict) -> str | None:
    """'D' for a debit row, 'C' for a credit row, None if it is neither or both."""
    debit, credit = row.get("debit_minor"), row.get("credit_minor")
    if not (_is_int(debit) and _is_int(credit)) or debit < 0 or credit < 0:
        return None
    if debit > 0 and credit == 0:
        return "D"
    if credit > 0 and debit == 0:
        return "C"
    return None


def row_amount(row: dict) -> int:
    side = _side(row)
    return row["debit_minor"] if side == "D" else row["credit_minor"] if side == "C" else 0


def _instant(value: object) -> int | None:
    try:
        return timeutil.parse_instant(value)
    except ValueError:
        return None


@dataclass
class InvoiceView:
    record: dict
    doc_id: str
    invoice_id: str
    entity_id: str
    currency: str
    total: int
    issued: int
    posting_rows: list[dict] = field(default_factory=list)
    ar_row: dict | None = None
    revenue_row: dict | None = None
    posting_time: int | None = None
    credited: int = 0
    ok: bool = True

    @property
    def remainder(self) -> int:
        return self.total - self.credited

    @property
    def posted_base(self) -> int | None:
        return self.revenue_row["credit_minor"] if self.revenue_row is not None else None


@dataclass
class PaymentView:
    record: dict
    doc_id: str
    payment_id: str
    amount: int
    currency: str
    time: int
    refs: list[str] | None
    rows: list[dict] = field(default_factory=list)
    entity_id: str | None = None
    allocated: int = 0
    ok: bool = True

    @property
    def remainder(self) -> int:
        return self.amount - self.allocated


@dataclass(frozen=True)
class FxRow:
    key: str
    day: date
    rate: Fraction


class Books:
    """Four tables plus a Config, indexed by exact ID.

    partial=True is for evidence checks on a subset of documents: references
    to documents outside the subset are then not structural errors.
    """

    def __init__(self, config: Config, tables: dict[str, list[dict]], *, partial: bool = False) -> None:
        self.config = config
        self.partial = partial
        self.ledger = list(tables.get("ledger", ()))
        self.invoice_records = list(tables.get("invoices", ()))
        self.payment_records = list(tables.get("payments", ()))
        self.fx_records = list(tables.get("fx", ()))
        self.structure: list[Violation] = []
        self.invoices: dict[str, InvoiceView] = {}   # by doc_id, file order
        self.payments: dict[str, PaymentView] = {}   # by doc_id, file order
        self.invoice_by_iid: dict[str, InvoiceView] = {}
        self.rows_by_iid: dict[str, list[dict]] = defaultdict(list)
        self.rows_by_pid: dict[str, list[dict]] = defaultdict(list)
        self.fx: dict[tuple[str, str], list[FxRow]] = defaultdict(list)
        self._index_fx()
        self._index_documents()
        self._index_rows()
        for view in self.invoices.values():
            self._check_invoice(view)
        for view in self.payments.values():
            self._check_payment(view)
        self._compute_credited()

    # --- helpers ---------------------------------------------------------

    def _bad(self, code: str, ids, message: str, *, amount: int | None = None, currency: str | None = None) -> None:
        self.structure.append(Violation("STRUCT", code, _sorted_ids(ids), message, amount, currency))

    def is_foreign(self, view: InvoiceView) -> bool:
        return view.currency != self.config.base

    # --- indexing --------------------------------------------------------

    def _index_fx(self) -> None:
        for record in self.fx_records:
            try:
                day = timeutil.parse_date(record.get("date"))
                rate = money.parse_rate(record.get("rate"))
            except ValueError:
                self._bad("bad_fx_row", [str(record.get("id"))], "fx row has a bad date or rate")
                continue
            pair = (record.get("source_currency"), record.get("target_currency"))
            self.fx[pair].append(FxRow(str(record.get("id")), day, rate))

    def _index_documents(self) -> None:
        seen_doc: set[str] = set()
        by_iid: dict[str, list[InvoiceView]] = defaultdict(list)
        for record in self.invoice_records:
            doc_id, iid = record.get("doc_id"), record.get("invoice_id")
            total, currency, entity = record.get("total_minor"), record.get("currency"), record.get("entity_id")
            issued = _instant(record.get("issued_at"))
            if not (isinstance(doc_id, str) and isinstance(iid, str) and isinstance(entity, str)
                    and isinstance(currency, str) and _is_int(total) and total > 0 and issued is not None):
                self._bad("bad_invoice", [str(doc_id)], "invoice has a missing or malformed field")
                continue
            if currency not in self.config.exponents:
                self._bad("unknown_currency", [doc_id], f"invoice currency {currency!r} has no exponent")
                continue
            if doc_id in seen_doc:
                self._bad("duplicate_doc_id", [doc_id], "doc_id used by more than one document")
                continue
            seen_doc.add(doc_id)
            view = InvoiceView(record, doc_id, iid, entity, currency, total, issued)
            self.invoices[doc_id] = view
            by_iid[iid].append(view)
        for iid, views in by_iid.items():
            if len(views) > 1:
                self._bad("duplicate_invoice_id", [v.doc_id for v in views], "invoice_id used by several invoices")
                for v in views:
                    v.ok = False
            self.invoice_by_iid[iid] = views[0]

        seen_pid: dict[str, PaymentView] = {}
        for record in self.payment_records:
            doc_id, pid = record.get("doc_id"), record.get("payment_id")
            amount, currency, refs = record.get("amount_minor"), record.get("currency"), record.get("invoice_refs")
            time = _instant(record.get("timestamp"))
            refs_ok = refs is None or (isinstance(refs, list) and refs and all(isinstance(r, str) for r in refs))
            if not (isinstance(doc_id, str) and isinstance(pid, str) and isinstance(currency, str)
                    and _is_int(amount) and amount != 0 and time is not None and refs_ok):
                self._bad("bad_payment", [str(doc_id)], "payment has a missing or malformed field")
                continue
            if doc_id in seen_doc:
                self._bad("duplicate_doc_id", [doc_id], "doc_id used by more than one document")
                continue
            seen_doc.add(doc_id)
            view = PaymentView(record, doc_id, pid, amount, currency, time, list(refs) if refs else None)
            if pid in seen_pid:
                self._bad("duplicate_payment_id", [doc_id, seen_pid[pid].doc_id], "payment_id used twice")
                view.ok = seen_pid[pid].ok = False
                continue
            seen_pid[pid] = view
            self.payments[doc_id] = view
        self.payment_by_pid = seen_pid

    def _index_rows(self) -> None:
        for row in self.ledger:
            row_id = str(row.get("row_id"))
            iid, pid, entity = row.get("invoice_id"), row.get("payment_id"), row.get("entity_id")
            if _side(row) is None:
                self._bad("bad_row_amounts", [row_id], "row must have exactly one of debit/credit above zero")
                continue
            if row.get("account") not in ACCOUNTS:
                self._bad("unknown_account", [row_id], f"unknown account {row.get('account')!r}")
            if row.get("currency") not in self.config.exponents:
                self._bad("unknown_currency", [row_id], f"row currency {row.get('currency')!r} has no exponent")
            if iid is not None:
                self.rows_by_iid[iid].append(row)
                if iid not in self.invoice_by_iid and not self.partial:
                    self._bad("orphan_row", [row_id], "row names an invoice_id that no invoice has")
            if pid is not None:
                self.rows_by_pid[pid].append(row)
                if pid not in self.payment_by_pid and not self.partial:
                    self._bad("orphan_row", [row_id], "row names a payment_id that no payment has")
            if iid is None and pid is None:
                self._check_background(row, row_id, entity)

    def _check_background(self, row: dict, row_id: str, entity: object) -> None:
        if entity is not None:
            self._bad("entity_without_document", [row_id], "row has entity_id but no invoice_id or payment_id")
        account, side = row.get("account"), _side(row)
        if not ((account in EXPENSE_ACCOUNTS and side == "D") or (account == "CASH" and side == "C")):
            self._bad("bad_background_row", [row_id],
                      "row without document links must be an expense debit or a CASH credit")

    # --- invoice postings -------------------------------------------------

    def _check_invoice(self, view: InvoiceView) -> None:
        rows = [r for r in self.rows_by_iid.get(view.invoice_id, []) if r.get("payment_id") is None]
        view.posting_rows = rows
        base = self.config.base
        ids = [view.doc_id] + [str(r.get("row_id")) for r in rows]
        problems: list[str] = []
        if self.is_foreign(view):
            expected = Counter({("AR", "D", view.currency): 1, ("FX_CLEARING", "C", view.currency): 1,
                                ("FX_CLEARING", "D", base): 1, ("REVENUE", "C", base): 1})
        else:
            expected = Counter({("AR", "D", base): 1, ("REVENUE", "C", base): 1})
        shape = Counter((r.get("account"), _side(r), r.get("currency")) for r in rows)
        if shape != expected:
            problems.append("posting rows do not have the documented shape")
        else:
            by_shape = {(r.get("account"), _side(r), r.get("currency")): r for r in rows}
            view.ar_row = by_shape[("AR", "D", view.currency)]
            view.revenue_row = by_shape[("REVENUE", "C", base)]
            if view.ar_row["debit_minor"] != view.total:
                problems.append("Dr AR is not total_minor")
            if self.is_foreign(view):
                if by_shape[("FX_CLEARING", "C", view.currency)]["credit_minor"] != view.total:
                    problems.append("Cr FX_CLEARING is not total_minor")
                if by_shape[("FX_CLEARING", "D", base)]["debit_minor"] != view.revenue_row["credit_minor"]:
                    problems.append("posting does not balance in the base currency")
            elif view.revenue_row["credit_minor"] != view.total:
                problems.append("Cr REVENUE is not total_minor")
        if any(r.get("entity_id") != view.entity_id for r in rows):
            problems.append("posting rows do not carry the invoice's entity_id")
        instants = {_instant(r.get("timestamp")) for r in rows}
        if len(instants) != 1 or None in instants:
            problems.append("posting rows do not share one timestamp")
        else:
            view.posting_time = instants.pop()
        if problems:
            view.ok = False
            self._bad("malformed_invoice_posting", ids, "; ".join(problems))

    # --- payments ---------------------------------------------------------

    def _check_payment(self, view: PaymentView) -> None:
        rows = self.rows_by_pid.get(view.payment_id, [])
        view.rows = rows
        ids = [view.doc_id] + [str(r.get("row_id")) for r in rows]
        problems: list[str] = []
        entities = {r.get("entity_id") for r in rows}
        if len(entities) != 1 or None in entities:
            problems.append("payment rows do not share one entity_id")
        else:
            view.entity_id = next(iter(entities))
        if any(r.get("currency") != view.currency for r in rows):
            problems.append("payment rows are not all in the payment currency")
        instants = {_instant(r.get("timestamp")) for r in rows}
        if len(instants) != 1 or None in instants:
            problems.append("payment rows do not share one timestamp")
        if view.amount > 0:
            problems += self._incoming_problems(view, rows)
        else:
            problems += self._refund_problems(view, rows)
        if problems:
            view.ok = False
            self._bad("malformed_payment", ids, "; ".join(problems))

    def _incoming_problems(self, view: PaymentView, rows: list[dict]) -> list[str]:
        problems: list[str] = []
        cash, ar, advances, other = [], [], [], []
        for r in rows:
            kind = (r.get("account"), _side(r), r.get("invoice_id") is None)
            if kind == ("CASH", "D", True):
                cash.append(r)
            elif kind == ("AR", "C", False):
                ar.append(r)
            elif kind == ("CUSTOMER_ADVANCES", "C", True):
                advances.append(r)
            else:
                other.append(r)
        if other:
            problems.append("payment has rows outside Dr CASH / Cr AR / Cr CUSTOMER_ADVANCES")
        if len(cash) != 1 or cash[0]["debit_minor"] != view.amount:
            problems.append("payment needs exactly one Dr CASH row for the whole amount")
        ar_iids = [r.get("invoice_id") for r in ar]
        if len(set(ar_iids)) != len(ar_iids):
            problems.append("several Cr AR rows for one invoice")
        refs = view.refs or []
        if len(set(refs)) != len(refs):
            problems.append("invoice_refs repeats an invoice")
        if set(ar_iids) != set(refs):
            problems.append("Cr AR rows do not match invoice_refs one to one")
        view.allocated = sum(r["credit_minor"] for r in ar)
        remainder = view.amount - view.allocated
        if remainder < 0:
            problems.append("Cr AR rows exceed the payment amount")
        elif remainder == 0 and advances:
            problems.append("Cr CUSTOMER_ADVANCES row without a remainder")
        elif remainder > 0 and (len(advances) != 1 or advances[0]["credit_minor"] != remainder):
            problems.append("remainder is not one Cr CUSTOMER_ADVANCES row of that size")
        for iid in refs:
            invoice = self.invoice_by_iid.get(iid)
            if invoice is None:
                if not self.partial:
                    problems.append(f"invoice_refs names an unknown invoice")
                continue
            if invoice.currency != view.currency:
                problems.append("payment currency differs from its invoice's currency")
            if view.entity_id is not None and invoice.entity_id != view.entity_id:
                problems.append("payment client differs from its invoice's client")
        return problems

    def _refund_problems(self, view: PaymentView, rows: list[dict]) -> list[str]:
        problems: list[str] = []
        if view.refs is not None:
            problems.append("refund with invoice_refs")
        shape = Counter((r.get("account"), _side(r), r.get("invoice_id") is None, row_amount(r)) for r in rows)
        magnitude = -view.amount
        if shape != Counter({("CUSTOMER_ADVANCES", "D", True, magnitude): 1, ("CASH", "C", True, magnitude): 1}):
            problems.append("refund must be Dr CUSTOMER_ADVANCES and Cr CASH on the absolute amount")
        return problems

    def _compute_credited(self) -> None:
        for view in self.invoices.values():
            view.credited = sum(
                r["credit_minor"] for r in self.rows_by_iid.get(view.invoice_id, [])
                if r.get("payment_id") is not None and r.get("account") == "AR" and _side(r) == "C"
            )

    # --- FX ---------------------------------------------------------------

    def correct_rate(self, view: InvoiceView) -> tuple[FxRow | None, str | None]:
        """The rate row the policy prescribes, or (None, reason)."""
        day = timeutil.local_date(view.issued, self.config.offset)
        candidates = [r for r in self.fx.get((view.currency, self.config.base), []) if r.day <= day]
        if not candidates:
            return None, "no rate for the pair on or before the invoice's local date"
        best_day = max(r.day for r in candidates)
        best = [r for r in candidates if r.day == best_day]
        if len(best) > 1:
            return None, "several rates for the pair on the same date"
        return best[0], None

    def correct_base(self, view: InvoiceView, rate: FxRow) -> int:
        exps = self.config.exponents
        return money.convert(view.total, rate.rate, exps[view.currency], exps[self.config.base],
                             self.config.rounding)

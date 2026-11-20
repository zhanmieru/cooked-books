"""Clean-book checks for generated episodes, written from DESIGN.md alone.

This module reads public data (through `load_public`, or tables a test has
already reverted) and never imports `cookedbooks.world`. It implements C1-C6
and the "Точные определения" of DESIGN.md section 5 on its own, plus detectors
for the distractor patterns, so a test can count them in the files.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

from cookedbooks.episode.loader import load_public

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
EXPENSES = {"RENT_EXPENSE", "PAYROLL_EXPENSE", "UTILITIES_EXPENSE"}
KEY_FIELDS = {"ledger.csv": "row_id", "invoices.jsonl": "doc_id", "payments.jsonl": "doc_id"}
TABLE_OF_FILE = {"ledger.csv": "ledger", "invoices.jsonl": "invoices", "payments.jsonl": "payments"}


def parse_offset(text: str) -> int:
    sign = -1 if text[0] == "-" else 1
    return sign * (int(text[1:3]) * 60 + int(text[4:6]))


def instant(text: str) -> int:
    moment = datetime.fromisoformat(text)
    assert moment.tzinfo is not None, f"timestamp without offset: {text}"
    return (moment - _EPOCH) // timedelta(seconds=1)


def local_day(text: str, offset: int) -> date:
    return datetime.fromisoformat(text).astimezone(timezone(timedelta(minutes=offset))).date()


def written_day(text: str) -> date:
    return datetime.fromisoformat(text).date()


def convert(src: int, rate: Decimal, exp_src: int, exp_base: int, mode: str) -> int:
    with localcontext() as ctx:
        ctx.prec = 80
        value = Decimal(src) * rate * Decimal(10) ** (exp_base - exp_src)
        rounding = ROUND_HALF_UP if mode == "half_up" else ROUND_HALF_EVEN
        return int(value.quantize(Decimal(1), rounding=rounding))


class Books:
    """Public tables with the derived quantities of DESIGN.md "Точные определения"."""

    def __init__(self, manifest: dict, policy: dict, tables: dict) -> None:
        self.manifest, self.policy = manifest, policy
        self.ledger = list(tables["ledger"])
        self.invoices = list(tables["invoices"])
        self.payments = list(tables["payments"])
        self.fx = list(tables["fx"])
        self.base = manifest["base_currency"]
        self.offset = parse_offset(manifest["utc_offset"])
        self.start = instant(manifest["period_start"])
        self.end = instant(manifest["period_end"])
        self.first_day = local_day(manifest["period_start"], self.offset)
        self.last_day = local_day(manifest["period_end"], self.offset)
        self.exponents = manifest["currency_exponents"]
        self.rates = {(r["date"], r["source_currency"], r["target_currency"]): Decimal(r["rate"]) for r in self.fx}
        self.invoice_by_id = {i["invoice_id"]: i for i in self.invoices}
        self.posting = defaultdict(list)  # invoice_id -> rows with that invoice and no payment
        self.pay_rows = defaultdict(list)  # payment_id -> rows
        for row in self.ledger:
            if row["payment_id"] is not None:
                self.pay_rows[row["payment_id"]].append(row)
            elif row["invoice_id"] is not None:
                self.posting[row["invoice_id"]].append(row)

    def in_period(self, moment: int) -> bool:
        return self.start <= moment <= self.end

    def applied(self, invoice_id: str) -> int:
        return sum(r["credit_minor"] for rows in self.pay_rows.values() for r in rows
                   if r["account"] == "AR" and r["invoice_id"] == invoice_id)

    def open_balance(self, invoice: dict) -> int:
        return invoice["total_minor"] - self.applied(invoice["invoice_id"])

    def payment_customer(self, payment: dict) -> str | None:
        entities = {r["entity_id"] for r in self.pay_rows[payment["payment_id"]]}
        return entities.pop() if len(entities) == 1 else None

    def unallocated(self, payment: dict) -> int:
        allocated = sum(r["credit_minor"] for r in self.pay_rows[payment["payment_id"]] if r["account"] == "AR")
        return payment["amount_minor"] - allocated

    def correct_base(self, invoice: dict) -> tuple[int, str] | None:
        """(correct base amount, fx key) by policy; None if no rate resolves."""
        day = local_day(invoice["issued_at"], self.offset)
        for back in range(60):
            probe = (day - timedelta(days=back)).isoformat()
            rate = self.rates.get((probe, invoice["currency"], self.base))
            if rate is not None:
                exps = self.exponents
                value = convert(invoice["total_minor"], rate, exps[invoice["currency"]], exps[self.base],
                                self.policy["rounding"])
                return value, f"{probe}:{invoice['currency']}:{self.base}"
        return None

    def distance(self, day: date) -> int:
        to_start = (day - self.first_day).days if day >= self.first_day else (self.first_day - day).days - 1
        to_end = (self.last_day - day).days if day <= self.last_day else (day - self.last_day).days - 1
        return min(to_start, to_end)


def books_from_folder(folder: str | Path) -> Books:
    episode = load_public(folder)
    return Books(episode.manifest, episode.policy, episode.tables)


def check_books(books: Books) -> list[str]:
    """Every violation of C1-C6 and of the posting definitions; [] for clean books."""
    v: list[str] = []
    b = books
    # rows: exactly one positive side, known accounts
    for row in b.ledger:
        d, c = row["debit_minor"], row["credit_minor"]
        if not (isinstance(d, int) and isinstance(c, int)) or (d > 0) == (c > 0) or min(d, c) != 0:
            v.append(f"row {row['row_id']}: bad sides {d}/{c}")
        if row["currency"] not in b.exponents:
            v.append(f"row {row['row_id']}: currency without exponent")
    # C1
    totals = defaultdict(lambda: [0, 0])
    for row in b.ledger:
        totals[row["currency"]][0] += row["debit_minor"]
        totals[row["currency"]][1] += row["credit_minor"]
    for currency, (dr, cr) in sorted(totals.items()):
        if dr != cr:
            v.append(f"C1 {currency}: debits {dr} != credits {cr}")
    # invoices: documents, postings, C2, C3, C6
    for inv in b.invoices:
        iid, cur, total = inv["invoice_id"], inv["currency"], inv["total_minor"]
        if total != sum(i["amount_minor"] for i in inv["line_items"]) or not 1 <= len(inv["line_items"]) <= 3:
            v.append(f"invoice {iid}: total/lines mismatch")
        for item in inv["line_items"]:
            if item["amount_minor"] != item["quantity"] * item["unit_price_minor"]:
                v.append(f"invoice {iid}: line amount mismatch")
        rows = b.posting.get(iid, [])
        if len({r["timestamp"] for r in rows}) != 1 or len({instant(r["timestamp"]) for r in rows}) != 1:
            v.append(f"invoice {iid}: posting rows missing or at different times")
            continue
        if any(r["entity_id"] != inv["entity_id"] for r in rows):
            v.append(f"invoice {iid}: posting entity differs")
        ar = [r for r in rows if r["account"] == "AR" and r["debit_minor"] > 0]
        revenue = [r for r in rows if r["account"] == "REVENUE" and r["credit_minor"] > 0]
        if len(ar) != 1 or ar[0]["debit_minor"] != total or ar[0]["currency"] != cur:
            v.append(f"C2 invoice {iid}: needs one Dr AR of total_minor in {cur}")
        if len(revenue) != 1 or revenue[0]["currency"] != b.base:
            v.append(f"invoice {iid}: needs one Cr REVENUE in base")
            continue
        if cur == b.base:
            if len(rows) != 2 or revenue[0]["credit_minor"] != total:
                v.append(f"invoice {iid}: base posting must be Dr AR / Cr REVENUE on total")
        else:
            shape = sorted((r["account"], r["currency"], r["debit_minor"] > 0) for r in rows)
            expected = sorted([("AR", cur, True), ("FX_CLEARING", cur, False),
                               ("FX_CLEARING", b.base, True), ("REVENUE", b.base, False)])
            fxc_src = [r for r in rows if r["account"] == "FX_CLEARING" and r["currency"] == cur]
            fxc_base = [r for r in rows if r["account"] == "FX_CLEARING" and r["currency"] == b.base]
            if shape != expected or fxc_src[0]["credit_minor"] != total \
                    or fxc_base[0]["debit_minor"] != revenue[0]["credit_minor"]:
                v.append(f"invoice {iid}: foreign posting is not the four-row FX_CLEARING form")
            correct = b.correct_base(inv)
            if correct is None or correct[0] != revenue[0]["credit_minor"]:
                v.append(f"C3 invoice {iid}: posted {revenue[0]['credit_minor']} correct {correct}")
        if b.applied(iid) > total:
            v.append(f"C2 invoice {iid}: applied more than total")
        if b.in_period(instant(rows[0]["timestamp"])) != b.in_period(instant(inv["issued_at"])):
            v.append(f"C6 invoice {iid}: posting on the other side of a period bound")
    # payments
    for pay in b.payments:
        pid, amount, cur, refs = pay["payment_id"], pay["amount_minor"], pay["currency"], pay["invoice_refs"]
        rows = b.pay_rows.get(pid, [])
        customer = b.payment_customer(pay)
        if customer is None or any(r["currency"] != cur for r in rows) \
                or len({instant(r["timestamp"]) for r in rows}) != 1:
            v.append(f"payment {pid}: rows missing, mixed entity, currency or time")
            continue
        if amount < 0:
            ok = refs is None and sorted((r["account"], r["debit_minor"], r["credit_minor"], r["invoice_id"])
                                         for r in rows) == sorted([("CASH", 0, -amount, None),
                                                                   ("CUSTOMER_ADVANCES", -amount, 0, None)])
            if not ok:
                v.append(f"payment {pid}: refund rows wrong")
            continue
        cash = [r for r in rows if r["account"] == "CASH"]
        ar = [r for r in rows if r["account"] == "AR"]
        adv = [r for r in rows if r["account"] == "CUSTOMER_ADVANCES"]
        if len(cash) + len(ar) + len(adv) != len(rows) or len(cash) != 1 \
                or cash[0]["debit_minor"] != amount or cash[0]["invoice_id"] is not None:
            v.append(f"payment {pid}: needs one Dr CASH in full without invoice_id")
        if any(r["credit_minor"] <= 0 or r["invoice_id"] is None for r in ar) \
                or sorted(r["invoice_id"] for r in ar) != sorted(refs or []):
            v.append(f"payment {pid}: Cr AR rows do not match invoice_refs")
        for iid in refs or []:
            inv = b.invoice_by_id.get(iid)
            if inv is None or inv["currency"] != cur or inv["entity_id"] != customer:
                v.append(f"payment {pid}: ref {iid} is not an invoice of the same customer and currency")
            elif instant(inv["issued_at"]) > instant(pay["timestamp"]):
                v.append(f"payment {pid}: pays an invoice issued after it")
        rest = b.unallocated(pay)
        if rest < 0 or (rest > 0) != bool(adv) or len(adv) > 1 or \
                (adv and (adv[0]["credit_minor"] != rest or adv[0]["invoice_id"] is not None)):
            v.append(f"payment {pid}: remainder must be one Cr CUSTOMER_ADVANCES row")
        # C4
        if rest > 0:
            moment = instant(pay["timestamp"])
            for inv in b.invoices:
                if inv["entity_id"] == customer and inv["currency"] == cur \
                        and instant(inv["issued_at"]) <= moment and b.open_balance(inv) > 0:
                    v.append(f"C4 payment {pid}: unallocated {rest} while {inv['invoice_id']} is open")
    # background rows
    for row in b.ledger:
        if row["invoice_id"] is None and row["payment_id"] is None:
            if row["entity_id"] is not None or row["currency"] != b.base \
                    or row["account"] not in EXPENSES | {"CASH"}:
                v.append(f"row {row['row_id']}: background row with references or a wrong account")
    # C5
    seen: dict[tuple, str] = {}
    for inv in b.invoices:
        if inv.get("po_ref") is None or inv.get("service_period") is None:
            continue
        key = tuple(inv[f] for f in ("entity_id", "po_ref", "service_period", "currency", "total_minor"))
        if key in seen:
            v.append(f"C5 invoices {seen[key]} and {inv['invoice_id']} share an obligation key")
        seen[key] = inv["invoice_id"]
    return v


def check_folder(folder: str | Path) -> list[str]:
    return check_books(books_from_folder(folder))


# --- distractor patterns ----------------------------------------------------------


def boundary_reasons(b: Books, inv: dict) -> list[str]:
    """Why an invoice looks like a boundary case: date line, missing rate day, or near a bound."""
    reasons = []
    local = local_day(inv["issued_at"], b.offset)
    if written_day(inv["issued_at"]) != local:
        reasons.append("written_date")
    if inv["currency"] != b.base and (local.isoformat(), inv["currency"], b.base) not in b.rates:
        reasons.append("no_rate_day")
    if b.distance(local) <= 2:
        reasons.append("near_bound")
    return reasons


def pattern_counts(b: Books) -> dict[str, int]:
    """How many instances of each distractor pattern the public data shows."""
    payers = defaultdict(set)
    for pid, rows in b.pay_rows.items():
        for r in rows:
            if r["account"] == "AR" and r["invoice_id"] is not None:
                payers[r["invoice_id"]].add(pid)
    multi_invoice = sum(1 for p in b.payments
                        if len({r["invoice_id"] for r in b.pay_rows[p["payment_id"]] if r["account"] == "AR"}) >= 2)
    refund_customers = {(b.payment_customer(p), p["currency"]) for p in b.payments if p["amount_minor"] < 0}
    groups = defaultdict(list)
    for inv in b.invoices:
        if inv["po_ref"] is not None:
            groups[(inv["entity_id"], inv["currency"], inv["total_minor"], inv["po_ref"])].append(inv)
    return {
        "split_payment": sum(1 for s in payers.values() if len(s) >= 2) + multi_invoice,
        "advance": sum(1 for p in b.payments if p["amount_minor"] > 0 and p["invoice_refs"] is None
                       and (b.payment_customer(p), p["currency"]) not in refund_customers),
        "overpayment": sum(1 for p in b.payments if p["amount_minor"] > 0 and p["invoice_refs"]
                           and b.unallocated(p) > 0),
        "refund": sum(1 for p in b.payments if p["amount_minor"] < 0),
        "recurring_amount": sum(1 for g in groups.values() if len(g) >= 2),
        "boundary": sum(1 for inv in b.invoices if boundary_reasons(b, inv)),
    }


# --- reverting provenance ---------------------------------------------------------


def revert_tables(tables: dict, provenance: dict) -> dict:
    """Undo provenance.injections (EVALUATION_CONTRACT.md section 2) on copies of the public tables."""
    result = {name: [dict(r) for r in records] for name, records in tables.items()}
    for injection in reversed(provenance["injections"]):
        for change in reversed(injection["changes"]):
            table = result[TABLE_OF_FILE[change["file"]]]
            field = KEY_FIELDS[change["file"]]
            position = next((i for i, r in enumerate(table) if r[field] == change["key"]), None)
            if change["op"] == "add":
                del table[position]
            elif change["op"] == "remove":
                assert position is None
                table.append(copy.deepcopy(change["before"]))
            else:
                table[position] = copy.deepcopy(change["before"])
    return result

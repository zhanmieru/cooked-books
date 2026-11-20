"""Format rules of contract section 2, checked on the loaded public files.

Returns plain problem strings (hard failures) and warnings (undocumented
extra fields, which are allowed). The loader has already rejected files
that are not UTF-8, not JSON/CSV, or miss a field of section 03.
"""

from __future__ import annotations

import re
from collections import Counter

from . import money, timeutil
from .books import ACCOUNTS, _is_int

LEDGER_FIELDS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
                 "currency", "entity_id", "invoice_id", "payment_id")
INVOICE_FIELDS = ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                  "total_minor", "line_items", "po_ref", "service_period")
PAYMENT_FIELDS = ("doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs")
FX_FIELDS = ("date", "source_currency", "target_currency", "rate", "id")
LINE_ITEM_FIELDS = ("description", "quantity", "unit_price_minor", "amount_minor")
MANIFEST_FIELDS = ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
                   "timezone", "utc_offset", "record_counts", "document_count",
                   "currency_exponents", "rounding")
_SERVICE_PERIOD = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")
_LIMIT = 20  # problems listed per table before summarising


def _is_ts(value: object) -> bool:
    try:
        timeutil.parse_instant(value)
        return True
    except ValueError:
        return False


class _Collector:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.warnings: list[str] = []
        self._per_table: Counter = Counter()

    def add(self, table: str, message: str) -> None:
        self._per_table[table] += 1
        if self._per_table[table] <= _LIMIT:
            self.problems.append(f"{table}: {message}")
        elif self._per_table[table] == _LIMIT + 1:
            self.problems.append(f"{table}: more problems not listed")

    def extra_fields(self, table: str, records, documented) -> None:
        extra = sorted({k for r in records for k in r if k not in documented})
        if extra:
            self.warnings.append(f"{table}: undocumented fields {extra}")


def check_manifest(manifest: dict, tables: dict, out: _Collector) -> None:
    for name in MANIFEST_FIELDS:
        if name not in manifest:
            out.add("manifest", f"missing {name}")
    if manifest.get("schema_version") != "1.0":
        out.add("manifest", "schema_version is not '1.0'")
    for name in ("episode_id", "timezone", "base_currency"):
        if name in manifest and not isinstance(manifest[name], str):
            out.add("manifest", f"{name} is not a string")
    counts = manifest.get("record_counts")
    if isinstance(counts, dict):
        for table in ("ledger", "invoices", "payments", "fx"):
            if counts.get(table) != len(tables[table]) or not _is_int(counts.get(table)):
                out.add("manifest", f"record_counts.{table} = {counts.get(table)!r}, data has {len(tables[table])}")
    elif "record_counts" in manifest:
        out.add("manifest", "record_counts is not an object")
    documents = len(tables["invoices"]) + len(tables["payments"])
    if "document_count" in manifest and (manifest["document_count"] != documents
                                         or not _is_int(manifest["document_count"])):
        out.add("manifest", f"document_count = {manifest.get('document_count')!r}, data has {documents}")
    rounding = manifest.get("rounding")
    if "rounding" in manifest and not (isinstance(rounding, dict) and rounding.get("mode") in money.ROUNDING_MODES
                                       and rounding.get("unit") == "minor"):
        out.add("manifest", "rounding must be {mode: half_up|half_even, unit: minor}")


def check_ledger(rows, exponents: dict, out: _Collector) -> None:
    seen: Counter = Counter(r.get("row_id") for r in rows)
    for row in rows:
        rid = row.get("row_id")
        if not isinstance(rid, str):
            out.add("ledger", "row without row_id")
            continue
        if seen[rid] > 1:
            out.add("ledger", f"row_id {rid!r} repeats")
        if not _is_ts(row.get("timestamp")):
            out.add("ledger", f"row {rid!r}: timestamp needs ISO-8601 with an offset")
        if row.get("account") not in ACCOUNTS:
            out.add("ledger", f"row {rid!r}: unknown account {row.get('account')!r}")
        debit, credit = row.get("debit_minor"), row.get("credit_minor")
        if not (_is_int(debit) and _is_int(credit) and debit >= 0 and credit >= 0 and (debit > 0) != (credit > 0)):
            out.add("ledger", f"row {rid!r}: exactly one of debit_minor/credit_minor must be above zero, the other 0")
        if row.get("currency") not in exponents:
            out.add("ledger", f"row {rid!r}: currency {row.get('currency')!r} has no exponent in the manifest")
        for name in ("entity_id", "invoice_id", "payment_id"):
            if row.get(name) is not None and not isinstance(row.get(name), str):
                out.add("ledger", f"row {rid!r}: {name} is not a string or null")
    out.extra_fields("ledger", rows, LEDGER_FIELDS)


def check_invoices(records, exponents: dict, out: _Collector) -> None:
    iids: Counter = Counter(r.get("invoice_id") for r in records)
    for record in records:
        doc = record.get("doc_id")
        where = f"invoice {doc!r}"
        for name in INVOICE_FIELDS:
            if name not in record:
                out.add("invoices", f"{where}: missing {name} (absent values are explicit null)")
        for name in ("doc_id", "invoice_id", "entity_id", "currency"):
            if not isinstance(record.get(name), str):
                out.add("invoices", f"{where}: {name} is not a string")
        if isinstance(record.get("invoice_id"), str) and iids[record["invoice_id"]] > 1:
            out.add("invoices", f"{where}: invoice_id repeats")
        for name in ("issued_at", "due_at"):
            if not _is_ts(record.get(name)):
                out.add("invoices", f"{where}: {name} needs ISO-8601 with an offset")
        if record.get("currency") not in exponents:
            out.add("invoices", f"{where}: currency has no exponent in the manifest")
        total = record.get("total_minor")
        if not (_is_int(total) and total > 0):
            out.add("invoices", f"{where}: total_minor is not a positive integer")
        items = record.get("line_items")
        if not isinstance(items, list) or not items:
            out.add("invoices", f"{where}: line_items is not a non-empty list")
        else:
            amounts = []
            for item in items:
                if not isinstance(item, dict) or any(k not in item for k in LINE_ITEM_FIELDS):
                    out.add("invoices", f"{where}: line item lacks {LINE_ITEM_FIELDS}")
                    continue
                if not _is_int(item["amount_minor"]) or not _is_int(item["unit_price_minor"]):
                    out.add("invoices", f"{where}: line item money is not an integer")
                    continue
                amounts.append(item["amount_minor"])
            if len(amounts) == len(items) and _is_int(total) and sum(amounts) != total:
                out.add("invoices", f"{where}: total_minor != sum of line_items amount_minor")
        for name in ("po_ref", "service_period"):
            value = record.get(name)
            if value is not None and not isinstance(value, str):
                out.add("invoices", f"{where}: {name} is not a string or null")
        period = record.get("service_period")
        if isinstance(period, str) and not _SERVICE_PERIOD.fullmatch(period):
            out.add("invoices", f"{where}: service_period is not YYYY-MM")
    out.extra_fields("invoices", records, INVOICE_FIELDS)


def check_payments(records, exponents: dict, out: _Collector) -> None:
    pids: Counter = Counter(r.get("payment_id") for r in records)
    for record in records:
        where = f"payment {record.get('doc_id')!r}"
        for name in ("doc_id", "payment_id", "currency"):
            if not isinstance(record.get(name), str):
                out.add("payments", f"{where}: {name} is not a string")
        if isinstance(record.get("payment_id"), str) and pids[record["payment_id"]] > 1:
            out.add("payments", f"{where}: payment_id repeats")
        if not _is_ts(record.get("timestamp")):
            out.add("payments", f"{where}: timestamp needs ISO-8601 with an offset")
        if record.get("currency") not in exponents:
            out.add("payments", f"{where}: currency has no exponent in the manifest")
        amount = record.get("amount_minor")
        if not (_is_int(amount) and amount != 0):
            out.add("payments", f"{where}: amount_minor is not a non-zero integer")
        refs = record.get("invoice_refs")
        if refs is not None and not (isinstance(refs, list) and refs and all(isinstance(r, str) for r in refs)):
            out.add("payments", f"{where}: invoice_refs must be null or a non-empty list of strings")
    out.extra_fields("payments", records, PAYMENT_FIELDS)


def check_fx(records, out: _Collector) -> None:
    for record in records:
        key = record.get("id")
        base_key = ":".join("" if record.get(c) is None else str(record.get(c)) for c in FX_FIELDS[:3])
        if key != base_key:
            out.add("fx", f"rate key {base_key!r} repeats (contradictory rates)")
        try:
            timeutil.parse_date(record.get("date"))
        except ValueError:
            out.add("fx", f"row {key!r}: date is not YYYY-MM-DD")
        for name in ("source_currency", "target_currency"):
            if not isinstance(record.get(name), str):
                out.add("fx", f"row {key!r}: {name} is not a string")
        if record.get("source_currency") == record.get("target_currency"):
            out.add("fx", f"row {key!r}: source and target currency are the same")
        try:
            money.parse_rate(record.get("rate"))
        except ValueError:
            out.add("fx", f"row {key!r}: rate is not a positive decimal string")
    out.extra_fields("fx", records, FX_FIELDS)


def check_formats(manifest: dict, tables: dict) -> tuple[list[str], list[str]]:
    """(problems, warnings) for the six public files."""
    out = _Collector()
    check_manifest(manifest, tables, out)
    exponents = manifest.get("currency_exponents") if isinstance(manifest.get("currency_exponents"), dict) else {}
    check_ledger(tables["ledger"], exponents, out)
    check_invoices(tables["invoices"], exponents, out)
    check_payments(tables["payments"], exponents, out)
    check_fx(tables["fx"], out)
    docs: Counter = Counter(r.get("doc_id") for r in list(tables["invoices"]) + list(tables["payments"]))
    for doc_id, count in docs.items():
        if count > 1:
            out.add("documents", f"doc_id {doc_id!r} is used by {count} documents")
    return out.problems, out.warnings

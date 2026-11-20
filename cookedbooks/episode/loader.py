"""Read the public part of an episode folder (EVALUATION_CONTRACT.md, sections 2, 3, 7).

Isolation: this module opens only the six public file names below. It never
opens, stats or lists `truth.json` or `provenance.json`, and it never lists
the folder. Optional files are detected by trying to open them.

Leniency: the jury loads folders it built or relabelled by hand, so only the
fields of section 03 of the case are required (REQUIRED_FIELDS). Our own
extra fields (`po_ref`, `service_period`) and unknown columns or keys are
optional and kept as they are. There are no integrity checks: no hashes, no
cross-file validation, no comparison with the manifest counts, no minimum
size. ValueError is raised only for a folder that cannot be read.

IDs are opaque strings. They are never stripped, case folded, parsed or
sorted. Records keep file order.

Copies: PublicEpisode never hands out its internal state. `manifest`,
`policy`, `tables` and every getter return a fresh deep copy, so callers may
mutate what they get. Because `tables` copies on each access, bind it once
instead of reading `episode.tables[...]` inside a loop.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any

TABLES = ("ledger", "invoices", "payments", "fx")

MANIFEST_FILE = "manifest.json"
POLICY_FILE = "policy.json"
TABLE_FILES = {
    "ledger": "ledger.csv",
    "invoices": "invoices.jsonl",
    "payments": "payments.jsonl",
    "fx": "fx.csv",
}
OPTIONAL_TABLES = frozenset({"invoices", "payments", "fx"})

# Section 03 fields, i.e. everything in contract section 2 not marked "наше поле".
REQUIRED_FIELDS = {
    "ledger": (
        "row_id", "timestamp", "account", "debit_minor", "credit_minor",
        "currency", "entity_id", "invoice_id", "payment_id",
    ),
    "invoices": (
        "doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
        "currency", "total_minor", "line_items",
    ),
    "payments": (
        "doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs",
    ),
    "fx": ("date", "source_currency", "target_currency", "rate"),
}

_CSV_INT_FIELDS = {"ledger": frozenset({"debit_minor", "credit_minor"}), "fx": frozenset()}
_JSON_ID_FIELDS = {"invoices": ("doc_id", "invoice_id"), "payments": ("doc_id", "payment_id")}
_JSON_MONEY_FIELDS = {"invoices": ("total_minor",), "payments": ("amount_minor",)}
_LINE_ITEM_MONEY_FIELDS = ("unit_price_minor", "amount_minor")
_INT_TEXT = re.compile(r"-?[0-9]+")


class PublicEpisode:
    """The six public files of one episode, indexed for lookup by exact ID."""

    __slots__ = ("_manifest", "_policy", "_tables", "_indexes", "_existing_ids")

    def __init__(self, manifest: dict, policy: dict, tables: dict[str, tuple[dict, ...]]) -> None:
        self._manifest = manifest
        self._policy = policy
        self._tables = tables
        ledger, invoices, payments, fx = (tables[name] for name in TABLES)
        inv_doc = _index(invoices, "doc_id")
        pay_doc = _index(payments, "doc_id")
        # Per table: the indexes get_record tries in order (contract section 3).
        self._indexes = {
            "ledger": (_index(ledger, "row_id"),),
            "invoices": (inv_doc, _index(invoices, "invoice_id")),
            "payments": (pay_doc, _index(payments, "payment_id")),
            "fx": (_index(fx, "id"),),
            # Not a table: get_document looks at doc_id only.
            "_documents": (inv_doc, pay_doc),
        }
        ids: set[str] = set()
        for records, field in (
            (ledger, "row_id"), (invoices, "doc_id"), (invoices, "invoice_id"),
            (payments, "doc_id"), (payments, "payment_id"), (fx, "id"),
        ):
            ids.update(r[field] for r in records if isinstance(r.get(field), str))
        self._existing_ids = frozenset(ids)

    @property
    def manifest(self) -> dict:
        return copy.deepcopy(self._manifest)

    @property
    def policy(self) -> dict:
        return copy.deepcopy(self._policy)

    @property
    def tables(self) -> dict[str, tuple[dict, ...]]:
        """All four tables, records in file order (a fresh copy on every access)."""
        return copy.deepcopy(self._tables)

    def get_record(self, table: str, record_id: str) -> dict | None:
        """Look a record up by exact ID in its table's ID space.

        ledger: row_id. invoices: doc_id, then invoice_id. payments: doc_id,
        then payment_id. fx: the `date:SRC:TGT` key (with `#n` suffix for
        repeats). Raises KeyError for an unknown table; returns None for an
        unknown ID.
        """
        if not isinstance(table, str) or table not in TABLES:
            raise KeyError(table)
        return self._find(table, self._indexes[table], record_id)

    def get_document(self, doc_id: str) -> dict | None:
        """Look `doc_id` up among invoices, then payments. None if absent."""
        for table, index in zip(("invoices", "payments"), self._indexes["_documents"]):
            found = self._find(table, (index,), doc_id)
            if found is not None:
                return found
        return None

    def existing_ids(self) -> frozenset[str]:
        """Every ID a tool can return: row_id, doc_id, invoice_id, payment_id, fx key."""
        return self._existing_ids

    def _find(self, table: str, indexes: tuple[dict[str, int], ...], record_id: Any) -> dict | None:
        if not isinstance(record_id, str):
            return None
        for index in indexes:
            position = index.get(record_id)
            if position is not None:
                return copy.deepcopy(self._tables[table][position])
        return None


def load_public(episode_dir: str | os.PathLike) -> PublicEpisode:
    """Load manifest, policy and the four tables from `episode_dir`."""
    root = Path(episode_dir)
    manifest_text = _read_text(root / MANIFEST_FILE, MANIFEST_FILE, optional=False)
    manifest = _parse_json_object(manifest_text, MANIFEST_FILE)

    policy_text = _read_text(root / POLICY_FILE, POLICY_FILE, optional=True)
    policy = {} if policy_text is None or not policy_text.strip() else _parse_json_object(policy_text, POLICY_FILE)

    tables: dict[str, tuple[dict, ...]] = {}
    for name in TABLES:
        file_name = TABLE_FILES[name]
        text = _read_text(root / file_name, file_name, optional=name in OPTIONAL_TABLES)
        if text is None:
            records: list[dict] = []
        elif file_name.endswith(".csv"):
            records = _parse_csv(text, name, file_name)
        else:
            records = _parse_jsonl(text, name, file_name)
        if name == "fx":
            _add_fx_ids(records)
        tables[name] = tuple(records)
    return PublicEpisode(manifest, policy, tables)


def _read_text(path: Path, name: str, *, optional: bool) -> str | None:
    """Return the file's text, or None when an optional file is missing."""
    try:
        # newline="" keeps quoted line breaks intact for the csv module.
        with open(path, encoding="utf-8-sig", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        if optional:
            return None
        raise ValueError(f"episode folder has no {name}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{name} is not valid UTF-8") from None
    except OSError:
        raise ValueError(f"cannot read {name}") from None


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def _parse_json_object(text: str, name: str) -> dict:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _parse_csv(text: str, table: str, name: str) -> list[dict]:
    # Not str.splitlines: it also breaks on U+2028, \x0c and friends inside values.
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    int_fields = _CSV_INT_FIELDS[table]
    records: list[dict] = []
    try:
        header = next(reader, None)
        if header is None:
            return records  # a completely empty file is an empty table
        if len(set(header)) != len(header):
            raise ValueError(f"{name}: repeated column name in header")
        missing = [column for column in REQUIRED_FIELDS[table] if column not in header]
        if missing:
            raise ValueError(f"{name}: missing column(s) {', '.join(missing)}")
        for row in reader:
            if all(not cell.strip() for cell in row):
                continue  # blank line
            if len(row) > len(header):
                raise ValueError(f"{name} line {reader.line_num}: more fields than columns")
            record: dict[str, Any] = {}
            for position, column in enumerate(header):
                raw = row[position] if position < len(row) else ""
                value: Any = None if raw in ("", "null") else raw
                if value is not None and column in int_fields:
                    value = _parse_int(value, name, reader.line_num, column)
                record[column] = value
            records.append(record)
    except csv.Error as exc:
        raise ValueError(f"{name}: malformed CSV: {exc}") from None
    return records


def _parse_int(text: str, name: str, line: int, column: str) -> int:
    if not _INT_TEXT.fullmatch(text):
        raise ValueError(f"{name} line {line}: {column} is not an integer")
    return int(text)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_jsonl(text: str, table: str, name: str) -> list[dict]:
    records: list[dict] = []
    # Only \n, \r and \r\n end a line; JSON strings may hold a raw U+2028.
    for line_no, line in enumerate(io.StringIO(text, newline=None), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            raise ValueError(f"{name} line {line_no}: invalid JSON: {exc}") from None
        if not isinstance(record, dict):
            raise ValueError(f"{name} line {line_no}: expected a JSON object")
        missing = [field for field in REQUIRED_FIELDS[table] if field not in record]
        if missing:
            raise ValueError(f"{name} line {line_no}: missing field(s) {', '.join(missing)}")
        for field in _JSON_ID_FIELDS[table]:
            if record[field] is not None and not isinstance(record[field], str):
                raise ValueError(f"{name} line {line_no}: {field} must be a string or null")
        for field in _JSON_MONEY_FIELDS[table]:
            if record[field] is not None and not _is_int(record[field]):
                raise ValueError(f"{name} line {line_no}: {field} is not an integer")
        if table == "invoices" and isinstance(record["line_items"], list):
            for item in record["line_items"]:
                if not isinstance(item, dict):
                    continue
                for field in _LINE_ITEM_MONEY_FIELDS:
                    if item.get(field) is not None and not _is_int(item[field]):
                        raise ValueError(f"{name} line {line_no}: line_items {field} is not an integer")
        records.append(record)
    return records


def _add_fx_ids(records: list[dict]) -> None:
    """Add `id` = `date:SRC:TGT`; repeats of a key get `#2`, `#3`, ... in file order."""
    seen: dict[str, int] = {}
    for record in records:
        parts = (record.get(field) for field in ("date", "source_currency", "target_currency"))
        key = ":".join("" if part is None else part for part in parts)
        seen[key] = seen.get(key, 0) + 1
        record["id"] = key if seen[key] == 1 else f"{key}#{seen[key]}"


def _index(records: tuple[dict, ...], field: str) -> dict[str, int]:
    """Map each string value of `field` to its first record position."""
    index: dict[str, int] = {}
    for position, record in enumerate(records):
        value = record.get(field)
        if isinstance(value, str):
            index.setdefault(value, position)
    return index

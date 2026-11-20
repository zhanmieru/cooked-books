"""Undo and redo the changes recorded in provenance.json (contract section 2).

Revert: walk `injections` from the end, inside each walk `changes` from the
end and apply the inverse (`add` -> delete `key`; `remove` -> put `before`
back; `modify` -> replace with `before`). Every step is checked instead of
guessed: an added key must be present and equal to `after`, a removed key
must be absent, a modified record must equal `after`. The first change that
does not apply cleanly stops the walk and is reported.

`apply_injection` is the forward direction (used to replay injections one
by one on the clean books).

Records are compared in the loader's shape: ledger and fx values are text
or None, `debit_minor`/`credit_minor` are integers, and fx records carry
their `id` key. Comparison is type-exact (1, 1.0 and true differ).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

FILE_TABLE = {"ledger.csv": "ledger", "invoices.jsonl": "invoices", "payments.jsonl": "payments", "fx.csv": "fx"}
KEY_FIELD = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "id"}
_LEDGER_INT = ("debit_minor", "credit_minor")
_FX_COLUMNS = ("date", "source_currency", "target_currency")

Tables = dict[str, list[dict]]


@dataclass
class ChangeProblem:
    injection: int
    change: int
    code: str
    message: str
    key: Any = None

    def to_dict(self) -> dict:
        return {"injection": self.injection, "change": self.change, "code": self.code,
                "message": self.message, "key": self.key}


@dataclass
class RevertResult:
    tables: Tables | None
    problems: list[ChangeProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.tables is not None


def fx_key(record: dict) -> str:
    return ":".join("" if record.get(c) is None else str(record.get(c)) for c in _FX_COLUMNS)


def normalize(table: str, record: Any) -> Any:
    """Bring a provenance record into the loader's shape for its table."""
    if not isinstance(record, dict):
        return record
    out = dict(record)
    if table in ("ledger", "fx"):
        for name, value in out.items():
            if value == "" or value == "null":
                out[name] = None
            elif table == "ledger" and name in _LEDGER_INT and isinstance(value, str):
                text = value.lstrip("-")
                out[name] = int(value) if text.isascii() and text.isdigit() else value
    if table == "fx":
        out.pop("id", None)
    return out


def _canon(table: str, record: Any) -> str:
    value = normalize(table, record)
    if table == "fx" and isinstance(value, dict):
        value = dict(value)
        value.pop("id", None)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _renumber_fx(records: list[dict]) -> None:
    seen: dict[str, int] = {}
    for record in records:
        key = fx_key(record)
        seen[key] = seen.get(key, 0) + 1
        record["id"] = key if seen[key] == 1 else f"{key}#{seen[key]}"


def _positions(records: list[dict], key_field: str, key: Any) -> list[int]:
    return [i for i, r in enumerate(records) if r.get(key_field) == key]


def _record_key(table: str, record: dict) -> Any:
    return fx_key(record) if table == "fx" else record.get(KEY_FIELD[table])


def validate_change(change: Any) -> str | None:
    """Return a problem message, or None if the change is well formed."""
    if not isinstance(change, dict):
        return "change is not an object"
    if change.get("file") not in FILE_TABLE:
        return f"change names file {change.get('file')!r}, not one of the four public tables"
    op = change.get("op")
    if op not in ("add", "modify", "remove"):
        return f"unknown op {op!r}"
    if not isinstance(change.get("key"), str):
        return "change key is not a string"
    table = FILE_TABLE[change["file"]]
    for side in ("before", "after"):
        value = change.get(side, None)
        must_be_null = (op == "add" and side == "before") or (op == "remove" and side == "after")
        if must_be_null:
            if value is not None:
                return f"{op} must have {side}: null"
        elif not isinstance(value, dict):
            return f"{op} needs a record in {side}"
        elif _record_key(table, value) != change["key"]:
            return f"{side} record's key differs from the change key"
    return None


def _apply(tables: Tables, change: dict, *, inverse: bool) -> str | None:
    """Apply one change (or its inverse) in place. Returns a problem message or None."""
    table = FILE_TABLE[change["file"]]
    records = tables[table]
    key = change["key"]
    key_field = KEY_FIELD[table]
    op = change["op"]
    present = _positions(records, key_field, key)
    if len(present) > 1:
        return f"key {key!r} is not unique in {change['file']}"
    # What must be there now, and what replaces it.
    if inverse:
        expect, put = {"add": ("after", None), "remove": (None, "before"), "modify": ("after", "before")}[op]
    else:
        expect, put = {"add": (None, "after"), "remove": ("before", None), "modify": ("before", "after")}[op]
    if expect is None:
        if present:
            return f"key {key!r} is still present in {change['file']}"
    else:
        if not present:
            return f"key {key!r} is missing from {change['file']}"
        if _canon(table, records[present[0]]) != _canon(table, change[expect]):
            return f"record {key!r} in {change['file']} does not equal the change's {expect!r}"
    new = None if put is None else normalize(table, copy.deepcopy(change[put]))
    if present and new is not None:
        records[present[0]] = new
    elif present:
        del records[present[0]]
    elif new is not None:
        records.append(new)
    if table == "fx":
        _renumber_fx(records)
    return None


def _injection_changes(injection: Any) -> list | None:
    if not isinstance(injection, dict) or not isinstance(injection.get("changes"), list):
        return None
    return injection["changes"]


def apply_injection(tables: Tables, injection: Any, index: int = 0) -> list[ChangeProblem]:
    """Apply one injection forward, in place. Stops at the first problem."""
    changes = _injection_changes(injection)
    if changes is None:
        return [ChangeProblem(index, -1, "malformed_injection", "injection has no changes list")]
    for position, change in enumerate(changes):
        message = validate_change(change) or _apply(tables, change, inverse=False)
        if message:
            return [ChangeProblem(index, position, "does_not_apply", message,
                                  change.get("key") if isinstance(change, dict) else None)]
    return []


def revert(tables: dict, provenance: Any) -> RevertResult:
    """Return the pre-injection tables, or the first change that does not undo cleanly."""
    work: Tables = {name: copy.deepcopy(list(tables.get(name, ()))) for name in KEY_FIELD}
    if not isinstance(provenance, dict) or not isinstance(provenance.get("injections"), list):
        return RevertResult(None, [ChangeProblem(-1, -1, "malformed_provenance", "provenance has no injections list")])
    injections = provenance["injections"]
    for index in range(len(injections) - 1, -1, -1):
        changes = _injection_changes(injections[index])
        if changes is None:
            return RevertResult(None, [ChangeProblem(index, -1, "malformed_injection", "injection has no changes list")])
        for position in range(len(changes) - 1, -1, -1):
            change = changes[position]
            message = validate_change(change) or _apply(work, change, inverse=True)
            if message:
                key = change.get("key") if isinstance(change, dict) else None
                return RevertResult(None, [ChangeProblem(index, position, "does_not_revert", message, key)])
    return RevertResult(work)

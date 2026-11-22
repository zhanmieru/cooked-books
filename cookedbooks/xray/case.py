"""Build the case file of one issue as plain data (no text formatting here).

Sources: the public files through `load_public`, `truth.json` through
`load_truth` (tolerant of hand-written truth), and `provenance.json` read
here with `json`. A missing or unreadable `provenance.json` is not an error:
the history and look-alike parts of the case are then empty with a note.

IDs are opaque strings. An ID is resolved by exact lookup in every ID space
(row_id, doc_id, invoice_id, payment_id, fx key). When the jury relabels a
folder, one string may name records of several spaces (contract section 7);
then every match is kept and the entry is flagged ambiguous. Matches are in
table order (ledger, invoices, payments, fx), then row position.

Nothing here writes to the folder or lists it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cookedbooks.episode.loader import TABLES, load_public
from cookedbooks.grader.truth import load_truth

PROVENANCE_FILE = "provenance.json"
# change.file -> (table, key field) (contract section 2, provenance.json).
CHANGE_TABLES = {
    "ledger.csv": ("ledger", "row_id"),
    "invoices.jsonl": ("invoices", "doc_id"),
    "payments.jsonl": ("payments", "doc_id"),
}
# (space, table, field), in table order.
SPACES = (
    ("row_id", "ledger", "row_id"),
    ("doc_id", "invoices", "doc_id"),
    ("invoice_id", "invoices", "invoice_id"),
    ("doc_id", "payments", "doc_id"),
    ("payment_id", "payments", "payment_id"),
    ("fx_key", "fx", "id"),
)
TABLE_ORDER = {name: i for i, name in enumerate(TABLES)}
DEFAULT_DUPLICATE_KEY = ("entity_id", "po_ref", "service_period", "currency", "total_minor")
# Look-alike kinds that mimic each error type (DESIGN.md, section 5, «Корректный двойник»).
TWIN_KINDS = {
    "duplicate_invoice": ("recurring_amount",),
    "unallocated_payment": ("advance", "overpayment", "split_payment", "refund"),
    "fx_error": ("boundary",),
    "cutoff_error": ("boundary",),
}
TWIN_KINDS_PARTIAL = ("overpayment", "advance", "split_payment", "refund")

Ref = tuple[str, int]  # (table, row position)


class XrayError(ValueError):
    """A folder that cannot be shown; the message is short and has no path."""


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_provenance(episode_dir: str | os.PathLike) -> tuple[dict | None, str | None]:
    """Return (provenance, None) or (None, reason in Russian)."""
    try:
        with open(Path(episode_dir) / PROVENANCE_FILE, encoding="utf-8-sig") as handle:
            text = handle.read()
    except FileNotFoundError:
        return None, "в папке нет provenance.json"
    except (OSError, UnicodeDecodeError):
        return None, "provenance.json не читается"
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        return None, "provenance.json не разбирается как JSON"
    if not isinstance(value, dict):
        return None, "provenance.json не JSON-объект"
    return value, None


def _dicts(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strs(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _same_issue_id(value: object, issue_id: str) -> bool:
    if isinstance(value, str):
        return value == issue_id
    return _is_int(value) and str(value) == issue_id


def parse_instant(text: object) -> datetime | None:
    """An aware datetime from an ISO-8601 string with offset, else None."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


class EpisodeView:
    """Everything of one folder that the case file needs, indexed by exact ID."""

    def __init__(self, episode_dir: str | os.PathLike) -> None:
        try:
            public = load_public(episode_dir)
        except ValueError as exc:
            raise XrayError(f"папка эпизода не читается ({exc})") from None
        try:
            self.issues = load_truth(episode_dir)
        except ValueError as exc:
            raise XrayError(f"эталон не читается ({exc})") from None
        self.manifest = public.manifest
        self.policy = public.policy
        self.tables = public.tables  # bound once: the property deep-copies
        self.provenance, self.provenance_note = read_provenance(episode_dir)
        prov = self.provenance or {}
        self.injections = _dicts(prov.get("injections"))
        self.distractors = [
            {"kind": d.get("kind"), "ids": _strs(d.get("ids"))} for d in _dicts(prov.get("distractors"))
        ]
        exponents = self.manifest.get("currency_exponents")
        self.exponents = {
            k: v for k, v in exponents.items() if isinstance(k, str) and _is_int(v) and v >= 0
        } if isinstance(exponents, dict) else {}

        self._spaces: list[tuple[str, str, dict[str, list[int]]]] = []
        for space, table, fname in SPACES:
            self._spaces.append((space, table, self._index(table, fname)))
        self._by_invoice = self._index("ledger", "invoice_id")
        self._by_payment = self._index("ledger", "payment_id")
        self._primary = {table: self._index(table, fname) for table, fname in CHANGE_TABLES.values()}

        # (table, position) -> "+" added or "~" modified, by any injection.
        self.changed: dict[Ref, str] = {}
        for injection in self.injections:
            for change in _dicts(injection.get("changes")):
                target = CHANGE_TABLES.get(change.get("file"))
                key = change.get("key")
                op = change.get("op")
                if target is None or not isinstance(key, str) or op not in ("add", "modify"):
                    continue
                for position in self._primary[target[0]].get(key, []):
                    ref = (target[0], position)
                    if self.changed.get(ref) != "+":
                        self.changed[ref] = "+" if op == "add" else "~"

    def _index(self, table: str, fname: str) -> dict[str, list[int]]:
        index: dict[str, list[int]] = {}
        for position, record in enumerate(self.tables[table]):
            value = record.get(fname)
            if isinstance(value, str):
                index.setdefault(value, []).append(position)
        return index

    def record(self, ref: Ref) -> dict:
        return self.tables[ref[0]][ref[1]]

    def resolve(self, record_id: str) -> list[Ref]:
        """Every distinct record that `record_id` names in any space, in file order."""
        refs: list[Ref] = []
        for _space, table, index in self._spaces:
            for position in index.get(record_id, []):
                if (table, position) not in refs:
                    refs.append((table, position))
        refs.sort(key=lambda ref: (TABLE_ORDER[ref[0]], ref[1]))
        return refs

    def children(self, ref: Ref) -> list[Ref]:
        """Ledger rows that reference an invoice (invoice_id) or a payment (payment_id)."""
        table, _ = ref
        record = self.record(ref)
        if table == "invoices" and isinstance(record.get("invoice_id"), str):
            return [("ledger", p) for p in self._by_invoice.get(record["invoice_id"], [])]
        if table == "payments" and isinstance(record.get("payment_id"), str):
            return [("ledger", p) for p in self._by_payment.get(record["payment_id"], [])]
        return []

    def find_issue(self, issue_id: str) -> int | None:
        for position, issue in enumerate(self.issues):
            if issue["issue_id"] == issue_id:
                return position
        return None

    def injection_of(self, issue_id: str) -> dict | None:
        for injection in self.injections:
            ids = injection.get("issue_ids")
            if isinstance(ids, list) and any(_same_issue_id(v, issue_id) for v in ids):
                return injection
        return None


@dataclass
class Entry:
    """One ID and the records it names."""
    id: str
    refs: list[Ref]

    @property
    def ambiguous(self) -> bool:
        return len(self.refs) > 1


@dataclass
class FieldChange:
    name: str
    before: Any
    after: Any


@dataclass
class Change:
    file: Any
    op: Any
    key: Any
    record: dict | None          # `after` for add, `before` for remove
    fields: list[FieldChange]    # for modify
    currency: Any


@dataclass
class Twin:
    kind: str
    entries: list[Entry]
    reason: str                  # fixed sentence for the kind (DESIGN.md, section 5)
    detail: dict | None = None   # computed facts, when the kind allows it


@dataclass
class Case:
    position: int
    issue: dict
    injection: dict | None
    siblings: list[str]
    entries: list[Entry]
    affected: list[Ref]
    evidence: list[Ref]
    evidence_set: list[str]
    changes: list[Change] | None   # None: history unavailable
    history_note: str | None
    twin: Twin | None
    twin_note: str | None

    def counts(self) -> tuple[int, int, int]:
        """(affected IDs found, evidence IDs found, changes): invariant under relabel and reorder."""
        found = {entry.id for entry in self.entries if entry.refs}
        affected = sum(1 for i in self.issue["affected_ids"] if i in found)
        evidence = sum(1 for i in self.evidence_set if i in found)
        return affected, evidence, len(self.changes or [])


def _ordered_unique(ids: list[str]) -> list[str]:
    out: list[str] = []
    for value in ids:
        if value not in out:
            out.append(value)
    return out


def _changes(injection: dict) -> list[Change]:
    out: list[Change] = []
    for change in _dicts(injection.get("changes")):
        before, after, op = change.get("before"), change.get("after"), change.get("op")
        fields: list[FieldChange] = []
        record = None
        if op == "modify" and isinstance(before, dict) and isinstance(after, dict):
            for name in list(before) + [k for k in after if k not in before]:
                if before.get(name) != after.get(name) or (name in before) != (name in after):
                    fields.append(FieldChange(name, before.get(name), after.get(name)))
        elif op == "add" and isinstance(after, dict):
            record = after
        elif op == "remove" and isinstance(before, dict):
            record = before
        source = after if isinstance(after, dict) else before if isinstance(before, dict) else {}
        out.append(Change(change.get("file"), op, change.get("key"), record, fields, source.get("currency")))
    return out


def twin_kinds(issue_type: object, injection: dict | None) -> tuple[str, ...]:
    kinds = TWIN_KINDS.get(issue_type, ()) if isinstance(issue_type, str) else ()
    if issue_type == "unallocated_payment" and injection is not None and (
        injection.get("variant") == "partial" or injection.get("scenario") == "S1"
    ):
        return TWIN_KINDS_PARTIAL
    return kinds


def _currencies(view: EpisodeView, refs: list[Ref]) -> list[Any]:
    return [view.record(r).get("currency") for r in refs if r[0] in ("invoices", "payments")]


def _pick_twin(view: EpisodeView, case_refs: list[Ref], issue: dict, kinds: tuple[str, ...]) -> dict | None:
    wanted = _currencies(view, case_refs) or [issue.get("currency")]
    for kind in kinds:
        candidates = [d for d in view.distractors if d["kind"] == kind and d["ids"]]
        for candidate in candidates:
            refs = [r for i in candidate["ids"] for r in view.resolve(i)]
            if any(c in wanted for c in _currencies(view, refs)):
                return candidate
        if candidates:
            return candidates[0]
    return None


def _docs(view: EpisodeView, entries: list[Entry], table: str) -> list[dict]:
    out: list[dict] = []
    for entry in entries:
        for ref in entry.refs:
            if ref[0] == table and view.record(ref) not in out:
                out.append(view.record(ref))
    return out


FIXED_REASONS = {
    "recurring_amount": "Ежемесячный счёт тому же клиенту на ту же сумму: месяц услуги "
                        "(service_period) другой, значит и обязательство другое.",
    "split_payment": "Оплата разбита на части между счетами и платежами, но каждый платёж "
                     "распределён полностью: в авансы ничего не ушло, открытого остатка нет.",
    "advance": "Настоящий аванс: на момент платежа у клиента нет открытых счетов в этой валюте, "
               "поэтому остаток в CUSTOMER_ADVANCES верен.",
    "overpayment": "Переплата: платёж закрывает счёт полностью, излишек уходит в CUSTOMER_ADVANCES. "
                   "Других открытых счетов клиента в этой валюте нет, значит излишек — настоящий аванс.",
    "refund": "Возврат после двойной оплаты: второй платёж без ссылки ушёл в авансы (счёт уже закрыт), "
              "потом деньги вернули платежом с отрицательной суммой. Открытого счёта у клиента нет.",
    "boundary": "Граница: записанная дата отличается от местной, или курс взят за день без курса, "
                "или проводка рядом с границей периода, но по ту же сторону по местному времени. "
                "По правилам policy.json ошибки нет.",
}


def _recurring_reason(view: EpisodeView, entries: list[Entry]) -> dict | None:
    invoices = _docs(view, entries, "invoices")
    if len(invoices) < 2:
        return None
    key = view.policy.get("duplicate_key")
    if not (isinstance(key, list) and key and all(isinstance(k, str) for k in key)):
        key = list(DEFAULT_DUPLICATE_KEY)
    first, second = invoices[0], invoices[1]
    return {
        "a": first.get("doc_id"), "b": second.get("doc_id"),
        "same": [k for k in key if first.get(k) == second.get(k)],
        "diff": [(k, first.get(k), second.get(k)) for k in key if first.get(k) != second.get(k)],
    }


def _overpayment_reason(view: EpisodeView, entries: list[Entry]) -> dict | None:
    invoices, payments = _docs(view, entries, "invoices"), _docs(view, entries, "payments")
    if len(invoices) != 1 or len(payments) != 1:
        return None
    inv, pay = invoices[0], payments[0]
    if inv.get("currency") != pay.get("currency") or not (
        _is_int(inv.get("total_minor")) and _is_int(pay.get("amount_minor"))
    ):
        return None
    credited = 0
    if isinstance(inv.get("invoice_id"), str):
        for position in view._by_invoice.get(inv["invoice_id"], []):
            row = view.tables["ledger"][position]
            if row.get("account") == "AR" and _is_int(row.get("credit_minor")):
                credited += row["credit_minor"]
    return {"invoice": inv.get("doc_id"), "payment": pay.get("doc_id"), "currency": inv.get("currency"),
            "excess": pay["amount_minor"] - inv["total_minor"], "open": inv["total_minor"] - credited}


def _advance_reason(view: EpisodeView, entries: list[Entry]) -> dict | None:
    invoices, payments = _docs(view, entries, "invoices"), _docs(view, entries, "payments")
    if len(payments) != 1:
        return None
    paid = parse_instant(payments[0].get("timestamp"))
    later = [i.get("doc_id") for i in invoices
             if paid is not None and (t := parse_instant(i.get("issued_at"))) is not None and t > paid]
    return {"later": later} if later else None


def build_twin(view: EpisodeView, case_refs: list[Ref], issue: dict,
               injection: dict | None) -> tuple[Twin | None, str | None]:
    if view.provenance is None:
        return None, "история изменений недоступна: " + (view.provenance_note or "")
    kinds = twin_kinds(issue.get("type"), injection)
    if not kinds:
        return None, "для этого типа ошибки двойник не определён"
    chosen = _pick_twin(view, case_refs, issue, kinds)
    if chosen is None:
        return None, "в эпизоде нет подходящего двойника (" + ", ".join(kinds) + ")"
    entries = [Entry(i, view.resolve(i)) for i in _ordered_unique(chosen["ids"])]
    computed = {
        "recurring_amount": _recurring_reason,
        "overpayment": _overpayment_reason,
        "advance": _advance_reason,
    }.get(chosen["kind"])
    detail = computed(view, entries) if computed else None
    return Twin(chosen["kind"], entries, FIXED_REASONS.get(chosen["kind"], ""), detail), None


def build_case(view: EpisodeView, position: int) -> Case:
    issue = view.issues[position]
    issue_id = issue["issue_id"]
    affected_ids = list(issue["affected_ids"])
    evidence_set = list(issue["evidence_sets"][0]) if issue["evidence_sets"] else []
    entries = [Entry(i, view.resolve(i)) for i in _ordered_unique(affected_ids + evidence_set)]
    by_id = {entry.id: entry for entry in entries}
    affected = [r for i in _ordered_unique(affected_ids) for r in by_id[i].refs]
    evidence = [r for i in _ordered_unique(evidence_set) for r in by_id[i].refs]

    injection = view.injection_of(issue_id)
    siblings: list[str] = []
    if injection is not None and isinstance(injection.get("issue_ids"), list):
        siblings = [str(v) for v in injection["issue_ids"]
                    if (isinstance(v, str) or _is_int(v)) and not _same_issue_id(v, issue_id)]
    if view.provenance is None:
        changes, note = None, "история изменений недоступна: " + (view.provenance_note or "")
    elif injection is None:
        changes, note = None, "история изменений недоступна: этой ошибки нет в provenance.json"
    else:
        changes, note = _changes(injection), None
    twin, twin_note = build_twin(view, affected, issue, injection)
    return Case(position, issue, injection, siblings, entries, affected, evidence, evidence_set,
                changes, note, twin, twin_note)


def build_all(view: EpisodeView) -> list[Case]:
    return [build_case(view, position) for position in range(len(view.issues))]

"""Small shared helpers for policies: action builders, a seeded RNG, record bookkeeping.

Standard library only; imports nothing from `cookedbooks` outside this package.
Using these helpers is optional: a policy only has to follow the interface in
`cookedbooks.policies` (`make_policy(seed, **options)` -> object with `act`).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Protocol

TABLES = ("ledger", "invoices", "payments", "fx")
ERROR_TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")
PAGE_LIMIT = 50
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10

Action = dict[str, Any]
Observation = dict[str, Any]
Record = dict[str, Any]


class Policy(Protocol):
    """What the worker expects from `make_policy(...)`."""

    def act(self, observation: Observation) -> Action:  # pragma: no cover - protocol
        ...


def seeded_rng(seed: int, *labels: object) -> random.Random:
    """A `random.Random` seeded from sha256 of the seed and labels (never `hash()`)."""
    text = ":".join([str(seed), *(str(label) for label in labels)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def is_int(value: Any) -> bool:
    """True for a JSON integer (bool excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_reset(observation: Observation) -> bool:
    """True for the first observation of an episode."""
    return isinstance(observation, dict) and "manifest" in observation and "task" in observation


def episode_id_of(observation: Observation) -> str:
    """`manifest.episode_id` of a reset observation, or ""."""
    manifest = observation.get("manifest") if isinstance(observation, dict) else None
    value = manifest.get("episode_id") if isinstance(manifest, dict) else None
    return value if isinstance(value, str) else ""


def list_records(table: str, cursor: str | None = None, limit: int = PAGE_LIMIT) -> Action:
    return {"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": limit}}


def get_record(table: str, record_id: str) -> Action:
    return {"name": "get_record", "args": {"table": table, "id": record_id}}


def get_document(doc_id: str) -> Action:
    return {"name": "get_document", "args": {"doc_id": doc_id}}


def calculator(expression: str) -> Action:
    return {"name": "calculator", "args": {"expression": expression}}


def submit(episode_id: str, findings: list[dict]) -> Action:
    return {"name": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings[:MAX_FINDINGS]}}}


def finding(kind: str, affected_ids: list[str], amount_minor: int, currency: str,
            evidence_ids: list[str], explanation: str) -> dict:
    """One structurally valid finding (evidence capped at the contract's 10 IDs)."""
    return {
        "type": kind,
        "affected_ids": list(affected_ids),
        "claimed_amount_minor": int(amount_minor),
        "currency": currency,
        "evidence_ids": list(evidence_ids)[:MAX_EVIDENCE_IDS],
        "explanation": explanation[:1000],
    }


def page_records(observation: Observation) -> tuple[list[Record], str | None] | None:
    """(records, next_cursor) if the observation is a successful list_records page, else None."""
    result = observation.get("result") if isinstance(observation, dict) else None
    if not isinstance(result, dict) or not isinstance(result.get("records"), list):
        return None
    records = [r for r in result["records"] if isinstance(r, dict)]
    cursor = result.get("next_cursor")
    return records, cursor if isinstance(cursor, str) else None


class Pager:
    """Reads tables page by page, in the order given, and keeps every record in file order.

    `next_action(observation)` returns the next list_records action, or None
    when every table is read (or only `reserve` steps remain).
    """

    def __init__(self, tables: tuple[str, ...], *, limit: int = PAGE_LIMIT, reserve: int = 1) -> None:
        self.tables = tables
        self.limit = limit
        self.reserve = reserve
        self.records: dict[str, list[Record]] = {table: [] for table in tables}
        self._index = 0
        self._cursor: str | None = None
        self._pending = False

    def next_action(self, observation: Observation) -> Action | None:
        if self._pending:
            self._pending = False
            page = page_records(observation)
            table = self.tables[self._index]
            if page is None:  # the call failed: move on to the next table
                self._index += 1
                self._cursor = None
            else:
                records, cursor = page
                self.records[table].extend(records)
                if cursor is None:
                    self._index += 1
                    self._cursor = None
                else:
                    self._cursor = cursor
        remaining = observation.get("remaining_steps") if isinstance(observation, dict) else None
        if self._index >= len(self.tables) or (is_int(remaining) and remaining <= self.reserve):
            return None
        self._pending = True
        return list_records(self.tables[self._index], self._cursor, self.limit)

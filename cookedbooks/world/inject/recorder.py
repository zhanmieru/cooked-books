"""Record every change an injector makes, so provenance can describe and revert it."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from cookedbooks.world.model import World


@dataclass(frozen=True)
class Change:
    table: str  # "ledger" | "invoices" | "payments"
    op: str  # "add" | "modify" | "remove"
    ref: str  # internal ref of the record
    before: Any  # the record before (None for add); records are frozen dataclasses
    after: Any  # the record after (None for remove)


class ChangeRecorder:
    """The only way an injector changes the world. Changes are kept in order."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.changes: list[Change] = []

    def add(self, table: str, record: Any) -> Any:
        records = self.world.table(table)
        if record.ref in records:
            raise ValueError(f"{table}: {record.ref} already exists")
        records[record.ref] = record
        self.changes.append(Change(table, "add", record.ref, None, record))
        return record

    def add_all(self, table: str, records: list[Any]) -> None:
        for record in records:
            self.add(table, record)

    def update(self, table: str, ref: str, **fields: Any) -> Any:
        """Replace fields of a record; returns the new record."""
        records = self.world.table(table)
        before = records[ref]
        after = dataclasses.replace(before, **fields)
        records[ref] = after
        self.changes.append(Change(table, "modify", ref, before, after))
        return after

    def remove(self, table: str, ref: str) -> Any:
        records = self.world.table(table)
        before = records.pop(ref)
        self.changes.append(Change(table, "remove", ref, before, None))
        return before

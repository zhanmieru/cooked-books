"""flag_all: the mass-flagging attack.

It reads the first page of invoices, payments and the ledger (three steps),
then submits as many structurally valid findings as the contract allows (30),
cycling through all four types over the first records it saw. Every ID it
uses exists, every amount is an integer, so nothing is rejected as invalid:
the attack is scored, and it must lose. Deterministic; `seed` is unused.
"""

from __future__ import annotations

from typing import Any

from cookedbooks.policies.base import (
    ERROR_TYPES, MAX_FINDINGS, PAGE_LIMIT, Action, Observation, Record, episode_id_of, finding, is_int,
    is_reset, list_records, page_records, submit,
)

EXPLANATION = "Flagged without analysis."
FIRST_PAGES = ("invoices", "payments", "ledger")


class FlagAll:
    def __init__(self) -> None:
        self._episode_id = ""
        self._base = ""
        self._pending: str | None = None
        self._queue: list[str] = []
        self._records: dict[str, list[Record]] = {}

    def act(self, observation: Observation) -> Action:
        if is_reset(observation):
            self._episode_id = episode_id_of(observation)
            manifest = observation.get("manifest") or {}
            base = manifest.get("base_currency") if isinstance(manifest, dict) else None
            self._base = base if isinstance(base, str) else ""
            self._queue = list(FIRST_PAGES)
            self._records = {table: [] for table in FIRST_PAGES}
            self._pending = None
        elif self._pending is not None:
            page = page_records(observation)
            if page is not None:
                self._records[self._pending] = page[0]
            self._pending = None
        remaining = observation.get("remaining_steps")
        if self._queue and not (is_int(remaining) and remaining <= 1):
            self._pending = self._queue.pop(0)
            return list_records(self._pending, None, PAGE_LIMIT)
        return submit(self._episode_id, self._findings())

    def _findings(self) -> list[dict]:
        records = self._records
        invoices: list[Record] = records.get("invoices", [])
        payments: list[Record] = records.get("payments", [])
        ledger: list[Record] = records.get("ledger", [])
        out: list[dict] = []
        for i in range(MAX_FINDINGS):
            kind = ERROR_TYPES[i % len(ERROR_TYPES)]
            inv = _pick(invoices, i)
            nxt = _pick(invoices, i + 1)
            pay = _pick(payments, i)
            row = _pick(ledger, i)
            inv_doc = _str(inv, "doc_id")
            amount = _amount(inv, "total_minor")
            currency = _str(inv, "currency") or self._base
            if kind == "duplicate_invoice":
                ids = [inv_doc, _str(nxt, "doc_id")]
                out.append(finding(kind, ids, amount, currency, ids, EXPLANATION))
            elif kind == "unallocated_payment":
                ids = [_str(pay, "doc_id"), inv_doc]
                out.append(finding(kind, ids, _amount(pay, "amount_minor"), _str(pay, "currency") or currency,
                                   ids, EXPLANATION))
            elif kind == "fx_error":
                out.append(finding(kind, [inv_doc], amount, self._base or currency,
                                   [inv_doc, _str(row, "row_id")], EXPLANATION))
            else:
                out.append(finding(kind, [inv_doc], amount, currency, [inv_doc, _str(row, "row_id")], EXPLANATION))
        return out


def _pick(records: list[Record], i: int) -> Record:
    return records[i % len(records)] if records else {}


def _str(record: Record, key: str) -> str:
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _amount(record: Record, key: str) -> int:
    value = record.get(key)
    return abs(value) if is_int(value) else 0


def make_policy(seed: int, **options: Any) -> FlagAll:
    return FlagAll()

"""random_explorer: seeded random exploration, then a few random findings.

It takes 5 to 25 random actions over the tables and the documents it has
discovered (list pages, follow cursors, fetch records and documents by IDs
it saw, run the calculator) and now and then an invalid one (an unknown
action, table or ID, bad arguments, an expression outside the grammar).
Then it submits 1 to 3 findings of random types built from IDs, amounts and
currencies it actually saw. The RNG is seeded from sha256 of the seed and the
episode ID, so the same seed on the same episode gives the same run.
"""

from __future__ import annotations

from typing import Any

from cookedbooks.policies.base import (
    ERROR_TYPES, TABLES, Action, Observation, calculator, episode_id_of, finding, get_document, get_record,
    is_int, is_reset, list_records, page_records, seeded_rng, submit,
)

ID_FIELDS = {"ledger": ("row_id",), "invoices": ("doc_id", "invoice_id"), "payments": ("doc_id", "payment_id"),
             "fx": ("id",)}
INVALID_ACTIONS: tuple[Action, ...] = (
    {"name": "delete_records", "args": {}},
    {"name": "list_records", "args": {"table": "customers", "cursor": None, "limit": 10}},
    {"name": "list_records", "args": {"table": "ledger", "cursor": "not-a-cursor", "limit": 10}},
    {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 0}},
    {"name": "get_record", "args": {"table": "invoices", "id": "no-such-record"}},
    {"name": "get_document", "args": {"doc_id": 42}},
    {"name": "calculator", "args": {"expression": "2 ** 10"}},
    {"name": "calculator", "args": {"expression": "1 / 0"}},
    {"name": "get_record"},
    "not an action",
)


class RandomExplorer:
    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._reset("")

    def _reset(self, episode_id: str) -> None:
        self._rng = seeded_rng(self._seed, "random_explorer", episode_id)
        self._episode_id = episode_id
        self._budget = self._rng.randint(5, 25)
        self._taken = 0
        self._last: Any = None
        self._ids: dict[str, list[str]] = {table: [] for table in TABLES}
        self._docs: list[str] = []
        self._cursors: dict[str, str] = {}
        self._amounts: list[int] = []
        self._currencies: list[str] = []
        self._row_ids: list[str] = []

    def act(self, observation: Observation) -> Action:
        if is_reset(observation):
            self._reset(episode_id_of(observation))
            manifest = observation.get("manifest") or {}
            base = manifest.get("base_currency") if isinstance(manifest, dict) else None
            if isinstance(base, str):
                self._currencies.append(base)
        else:
            self._learn(observation)
        remaining = observation.get("remaining_steps")
        if self._taken >= self._budget or (is_int(remaining) and remaining <= 1):
            return self._submit()
        self._taken += 1
        self._last = self._choose()
        return self._last

    # --- bookkeeping ------------------------------------------------------------------

    def _learn(self, observation: Observation) -> None:
        last = self._last if isinstance(self._last, dict) else {}
        args = last.get("args") if isinstance(last.get("args"), dict) else {}
        table = args.get("table") if last.get("name") in ("list_records", "get_record") else None
        page = page_records(observation)
        records = []
        if page is not None and table in self._ids:
            records, cursor = page
            if cursor is not None:
                self._cursors[table] = cursor
            else:
                self._cursors.pop(table, None)
        elif isinstance(observation.get("result"), dict) and last.get("name") in ("get_record", "get_document"):
            records = [observation["result"]]
            if table not in self._ids:
                table = "payments" if "payment_id" in observation["result"] else "invoices"
        for record in records:
            self._remember(table, record)

    def _remember(self, table: Any, record: dict) -> None:
        if table not in ID_FIELDS:
            return
        for name in ID_FIELDS[table]:
            value = record.get(name)
            if isinstance(value, str) and value not in self._ids[table]:
                self._ids[table].append(value)
                if name == "doc_id":
                    self._docs.append(value)
                if name == "row_id":
                    self._row_ids.append(value)
        for name in ("total_minor", "amount_minor", "debit_minor", "credit_minor"):
            value = record.get(name)
            if is_int(value) and value > 0 and value not in self._amounts:
                self._amounts.append(value)
        currency = record.get("currency")
        if isinstance(currency, str) and currency not in self._currencies:
            self._currencies.append(currency)

    # --- choices ----------------------------------------------------------------------

    def _choose(self) -> Any:
        rng = self._rng
        roll = rng.random()
        if roll < 0.35 or not self._docs:
            if roll >= 0.9:
                return INVALID_ACTIONS[rng.randrange(len(INVALID_ACTIONS))]
            table = TABLES[rng.randrange(len(TABLES))]
            cursor = self._cursors.get(table) if rng.random() < 0.6 else None
            return list_records(table, cursor, rng.randint(1, 60))
        if roll < 0.55:
            tables = [t for t in TABLES if self._ids[t]]
            table = tables[rng.randrange(len(tables))]
            ids = self._ids[table]
            return get_record(table, ids[rng.randrange(len(ids))])
        if roll < 0.70:
            return get_document(self._docs[rng.randrange(len(self._docs))])
        if roll < 0.85:
            a = self._amounts[rng.randrange(len(self._amounts))] if self._amounts else rng.randint(1, 999)
            b = rng.randint(1, 1000)
            op = "+-*/"[rng.randrange(4)]
            return calculator(f"({a} {op} {b}) / 100")
        return INVALID_ACTIONS[rng.randrange(len(INVALID_ACTIONS))]

    def _submit(self) -> Action:
        rng = self._rng
        findings = []
        for _ in range(rng.randint(1, 3)):
            kind = ERROR_TYPES[rng.randrange(len(ERROR_TYPES))]
            pool = self._docs or [""]
            n_affected = 2 if kind in ("duplicate_invoice", "unallocated_payment") else 1
            affected = [pool[rng.randrange(len(pool))] for _ in range(n_affected)]
            evidence = list(affected)
            if self._row_ids and rng.random() < 0.5:
                evidence.append(self._row_ids[rng.randrange(len(self._row_ids))])
            amount = self._amounts[rng.randrange(len(self._amounts))] if self._amounts else 0
            currency = self._currencies[rng.randrange(len(self._currencies))] if self._currencies else ""
            findings.append(finding(kind, affected, amount, currency, evidence, "Random guess."))
        return submit(self._episode_id, findings)


def make_policy(seed: int, **options: Any) -> RandomExplorer:
    return RandomExplorer(seed)

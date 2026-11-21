"""Write the eight episode files (EVALUATION_CONTRACT.md, section 2).

IDs are assigned here, over live records and every record a change touched.
File order is chronological; ties are broken by a seeded draw, never by ID.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookedbooks.world.fx import EXPONENTS, FX_MIN_MINOR, FX_MIN_RELATIVE
from cookedbooks.world.ids import assign_ids
from cookedbooks.world.inject.base import InjectionOutcome, InjectionRequest, IssueDraft, Ref
from cookedbooks.world.inject.recorder import Change
from cookedbooks.world.model import Invoice, LedgerRow, Payment, World
from cookedbooks.world.rng import stream
from cookedbooks.world.timeutil import iso, offset_text

SCHEMA_VERSION = "1.0"
LEDGER_COLUMNS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
                  "currency", "entity_id", "invoice_id", "payment_id")
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
FILES = {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl"}


@dataclass
class AppliedInjection:
    request: InjectionRequest
    outcome: InjectionOutcome
    changes: list[Change]


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: str
    seed: int
    attempt: int
    generator_version: str
    timezone: str
    team_config: dict


class _OrderedSet(dict):
    def push(self, value: Any) -> None:
        if value is not None:
            self.setdefault(value, None)


def _collect_spaces(world: World, injections: list[AppliedInjection]) -> dict[str, list[str]]:
    rows, docs, invoice_ids, payment_ids, entities = (_OrderedSet() for _ in range(5))
    snapshots = [c.before for a in injections for c in a.changes] + [c.after for a in injections for c in a.changes]
    records = [*world.invoices.values(), *world.payments.values(), *world.ledger.values(),
               *(s for s in snapshots if s is not None)]
    for record in records:
        if isinstance(record, Invoice):
            docs.push(record.ref)
            invoice_ids.push(record.ref)
            entities.push(record.customer)
        elif isinstance(record, Payment):
            docs.push(record.ref)
            payment_ids.push(record.ref)
            for ref in record.invoice_refs or ():
                invoice_ids.push(ref)
        elif isinstance(record, LedgerRow):
            rows.push(record.ref)
            invoice_ids.push(record.invoice)
            payment_ids.push(record.payment)
            entities.push(record.customer)
    return {"row": list(rows), "doc": list(docs), "invoice_id": list(invoice_ids),
            "payment_id": list(payment_ids), "entity": list(entities)}


class _Renderer:
    def __init__(self, ids: dict[str, dict[str, str]]) -> None:
        self.ids = ids

    def invoice(self, inv: Invoice) -> dict:
        return {
            "doc_id": self.ids["doc"][inv.ref],
            "invoice_id": self.ids["invoice_id"][inv.ref],
            "entity_id": self.ids["entity"][inv.customer],
            "issued_at": iso(inv.issued_utc, inv.issued_offset),
            "due_at": iso(inv.due_utc, inv.due_offset),
            "currency": inv.currency,
            "total_minor": inv.total_minor,
            "line_items": [
                {"description": i.description, "quantity": i.quantity,
                 "unit_price_minor": i.unit_price_minor, "amount_minor": i.amount_minor}
                for i in inv.line_items
            ],
            "po_ref": inv.po_ref,
            "service_period": inv.service_period,
        }

    def payment(self, pay: Payment) -> dict:
        refs = None if pay.invoice_refs is None else [self.ids["invoice_id"][r] for r in pay.invoice_refs]
        return {
            "doc_id": self.ids["doc"][pay.ref],
            "payment_id": self.ids["payment_id"][pay.ref],
            "timestamp": iso(pay.ts_utc, pay.ts_offset),
            "currency": pay.currency,
            "amount_minor": pay.amount_minor,
            "invoice_refs": refs,
        }

    def row(self, row: LedgerRow) -> dict:
        def opt(space: str, ref: str | None) -> str | None:
            return None if ref is None else self.ids[space][ref]
        return {
            "row_id": self.ids["row"][row.ref],
            "timestamp": iso(row.ts_utc, row.ts_offset),
            "account": row.account,
            "debit_minor": row.debit_minor,
            "credit_minor": row.credit_minor,
            "currency": row.currency,
            "entity_id": opt("entity", row.customer),
            "invoice_id": opt("invoice_id", row.invoice),
            "payment_id": opt("payment_id", row.payment),
        }

    def record(self, record: Any) -> dict | None:
        if record is None:
            return None
        if isinstance(record, Invoice):
            return self.invoice(record)
        if isinstance(record, Payment):
            return self.payment(record)
        return self.row(record)

    def key(self, table: str, ref: str) -> str:
        return self.ids["row" if table == "ledger" else "doc"][ref]

    def ref(self, ref: Ref) -> str:
        space, value = ref
        if space in ("invoice", "payment"):
            return self.ids["doc"][value]
        if space == "row":
            return self.ids["row"][value]
        if space in ("invoice_id", "payment_id"):
            return self.ids[space][value]
        if space == "fx":
            return value
        raise ValueError(f"unknown reference space {space!r}")


def _ordered(world: World, seed: int, attempt: int) -> tuple[list[Invoice], list[Payment], list[LedgerRow]]:
    """Chronological order; ties broken by seeded draws (drawn in creation order, never from IDs)."""
    rng = stream(seed, attempt, "order")
    tie = {ref: rng.getrandbits(48) for ref in [*world.invoices, *world.payments, *world.ledger]}
    postings: dict[str, int] = {}
    for row in world.ledger.values():
        postings.setdefault(row.posting, rng.getrandbits(48))
    invoices = sorted(world.invoices.values(), key=lambda i: (i.issued_utc, tie[i.ref]))
    payments = sorted(world.payments.values(), key=lambda p: (p.ts_utc, tie[p.ref]))
    ledger = sorted(world.ledger.values(), key=lambda r: (r.ts_utc, postings[r.posting], tie[r.ref]))
    return invoices, payments, ledger


def _issue(renderer: _Renderer, issue_id: str, draft: IssueDraft) -> dict:
    return {
        "issue_id": issue_id,
        "type": draft.type,
        "affected_ids": [renderer.ref(r) for r in draft.affected],
        "expected_amount_minor": draft.expected_amount_minor,
        "currency": draft.currency,
        "evidence_sets": [[renderer.ref(r) for r in evidence] for evidence in draft.evidence_sets],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
    path.write_text("".join(lines), encoding="utf-8")


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[list[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])


def export_episode(world: World, injections: list[AppliedInjection], meta: EpisodeMeta,
                   output_dir: str | os.PathLike) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    renderer = _Renderer(assign_ids(meta.seed, meta.attempt, _collect_spaces(world, injections)))
    invoices, payments, ledger = _ordered(world, meta.seed, meta.attempt)
    fx_rows = sorted(world.fx.rows())

    currencies = {world.base_currency, *world.foreign_currencies}
    currencies.update(code for _, s, t, _ in fx_rows for code in (s, t))
    tl = world.timeline
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": meta.episode_id,
        "base_currency": world.base_currency,
        "period_start": iso(tl.period_start_utc, tl.offset),
        "period_end": iso(tl.period_end_utc, tl.offset),
        "timezone": meta.timezone,
        "utc_offset": offset_text(tl.offset),
        "record_counts": {"ledger": len(ledger), "invoices": len(invoices),
                          "payments": len(payments), "fx": len(fx_rows)},
        "document_count": len(invoices) + len(payments),
        "currency_exponents": {code: EXPONENTS[code] for code in sorted(currencies)},
        "rounding": {"mode": world.rounding, "unit": "minor"},
    }
    policy = {
        "fx_rate_date": "document_date",
        "fx_missing_rate": "previous_available",
        "fx_direction": "source_to_target",
        "rounding": world.rounding,
        "period_bounds": "inclusive",
        "cutoff_basis": "issued_at_local",
        "allocation_rule": "open_invoices_first",
        "duplicate_key": ["entity_id", "po_ref", "service_period", "currency", "total_minor"],
        "fx_materiality": {"min_relative": str(FX_MIN_RELATIVE), "min_minor": FX_MIN_MINOR},
        "amount_tolerance_minor": {
            "duplicate_invoice": 0, "unallocated_payment": 0, "fx_error": 0, "cutoff_error": 0,
        },
    }
    issues: list[dict] = []
    provenance_injections: list[dict] = []
    for applied in injections:
        issue_ids = []
        for draft in applied.outcome.issues:
            issue_id = f"ISS-{len(issues) + 1:04d}"
            issues.append(_issue(renderer, issue_id, draft))
            issue_ids.append(issue_id)
        provenance_injections.append({
            "issue_ids": issue_ids,
            "kind": applied.outcome.kind or applied.request.kind,
            "variant": applied.request.variant,
            "scenario": applied.request.scenario,
            "description": applied.outcome.description,
            "changes": [
                {"file": FILES[c.table], "op": c.op, "key": renderer.key(c.table, c.ref),
                 "before": renderer.record(c.before), "after": renderer.record(c.after)}
                for c in applied.changes
            ],
            "effects": list(applied.outcome.effects),
        })
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": meta.episode_id,
        "generator_version": meta.generator_version,
        "seed": meta.seed,
        "attempt": meta.attempt,
        "team_config": meta.team_config,
        "injections": provenance_injections,
        "distractors": [
            {"kind": d.kind, "ids": [renderer.ref(doc) for doc in d.docs]} for d in world.distractors
        ],
    }
    truth = {"schema_version": SCHEMA_VERSION, "episode_id": meta.episode_id, "issues": issues}

    _write_json(root / "manifest.json", manifest)
    _write_csv(root / "ledger.csv", LEDGER_COLUMNS,
               [[r[c] for c in LEDGER_COLUMNS] for r in map(renderer.row, ledger)])
    _write_jsonl(root / "invoices.jsonl", [renderer.invoice(i) for i in invoices])
    _write_jsonl(root / "payments.jsonl", [renderer.payment(p) for p in payments])
    _write_csv(root / "fx.csv", FX_COLUMNS, [[d.isoformat(), s, t, rate] for d, s, t, rate in fx_rows])
    _write_json(root / "policy.json", policy)
    _write_json(root / "truth.json", truth)
    _write_json(root / "provenance.json", provenance)

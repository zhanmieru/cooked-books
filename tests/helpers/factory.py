"""Write episode folders for tests, in the on-disk format of EVALUATION_CONTRACT.md section 2.

This module encodes the file format on its own (it does not import the
loader), so loader bugs show up as test failures instead of cancelling out.

IDs here are deliberately awkward: unprefixed, mixed case, purely numeric,
and the same string reused in different ID spaces. Code that guesses a table
from an ID's shape breaks on these fixtures.

Everything is deterministic: no randomness, no clock.

Run `python -m tests.helpers.factory` to rewrite tests/fixtures/{mini,clean,large}.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
EPISODE_FILES = (
    "manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
    "fx.csv", "policy.json", "truth.json", "provenance.json",
)
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def default_manifest(episode_id: str = "ep-test") -> dict:
    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "base_currency": "KZT",
        "period_start": "2026-07-01T00:00:00+05:00",
        "period_end": "2026-09-30T23:59:59+05:00",
        "timezone": "Asia/Almaty",
        "utc_offset": "+05:00",
        "currency_exponents": {"KZT": 2, "USD": 2},
        "rounding": {"mode": "half_up", "unit": "minor"},
    }


def default_policy() -> dict:
    return {
        "fx_rate_date": "document_date",
        "fx_missing_rate": "previous_available",
        "fx_direction": "source_to_target",
        "rounding": "half_up",
        "period_bounds": "inclusive",
        "cutoff_basis": "issued_at_local",
        "allocation_rule": "open_invoices_first",
        "duplicate_key": ["entity_id", "po_ref", "service_period", "currency", "total_minor"],
        "fx_materiality": {"min_relative": "0.005", "min_minor": 100},
        "amount_tolerance_minor": {
            "duplicate_invoice": 0, "unallocated_payment": 0, "fx_error": 0, "cutoff_error": 0,
        },
    }


def write_episode(
    directory: str | os.PathLike,
    *,
    manifest: dict | None = None,
    policy: dict | None = None,
    ledger: Iterable[dict] = (),
    invoices: Iterable[dict] = (),
    payments: Iterable[dict] = (),
    fx: Iterable[dict] = (),
    issues: list[dict] | None = None,
    provenance: dict | None = None,
) -> Path:
    """Write one episode folder and return its path.

    manifest=None writes default_manifest(); a given manifest is written as
    is, except that `record_counts` and `document_count` are filled from the
    data when it lacks them. policy=None writes default_policy() (delete the
    file afterwards to test a folder without one). issues=None writes no
    truth.json; a list (even empty) writes it in the wrapper form.
    provenance=None writes no provenance.json. CSV rows are written in the
    given order, None as an empty field; extra keys become extra columns.
    Episode files from an earlier write that are not written now are removed.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    ledger, invoices, payments, fx = (list(t) for t in (ledger, invoices, payments, fx))

    manifest = copy.deepcopy(manifest) if manifest is not None else default_manifest()
    manifest.setdefault("record_counts", {
        "ledger": len(ledger), "invoices": len(invoices), "payments": len(payments), "fx": len(fx),
    })
    manifest.setdefault("document_count", len(invoices) + len(payments))
    policy = default_policy() if policy is None else policy

    files: dict[str, str] = {
        "manifest.json": _json(manifest),
        "policy.json": _json(policy),
        "ledger.csv": _csv(LEDGER_COLUMNS, ledger),
        "invoices.jsonl": _jsonl(invoices),
        "payments.jsonl": _jsonl(payments),
        "fx.csv": _csv(FX_COLUMNS, fx),
    }
    if issues is not None:
        files["truth.json"] = _json({
            "schema_version": "1.0",
            "episode_id": manifest.get("episode_id"),
            "issues": list(issues),
        })
    if provenance is not None:
        files["provenance.json"] = _json(provenance)

    for name in EPISODE_FILES:
        path = root / name
        if name in files:
            path.write_text(files[name], encoding="utf-8", newline="")
        elif path.is_file():
            path.unlink()
    return root


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def _csv(columns: tuple[str, ...], records: list[dict]) -> str:
    header = list(columns)
    for record in records:
        header.extend(key for key in record if key not in header)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for record in records:
        writer.writerow("" if record.get(c) is None else str(record[c]) for c in header)
    return buffer.getvalue()


# --- ready-made episodes -----------------------------------------------------


@dataclass(frozen=True)
class EpisodeFixture:
    """What a builder wrote. All ID tuples are in file order.

    r1/r2/r3 and d1/d2/d3 follow the jury's naming: known ledger rows and
    known documents (d1, d2 are invoices, d3 is a payment). u is an ID that
    exists nowhere in the public data. issues is what truth.json holds
    (empty for clean and large).
    """

    path: Path
    episode_id: str
    row_ids: tuple[str, ...]
    invoice_doc_ids: tuple[str, ...]
    invoice_ids: tuple[str, ...]
    payment_doc_ids: tuple[str, ...]
    payment_ids: tuple[str, ...]
    fx_ids: tuple[str, ...]
    r1: str
    r2: str
    r3: str
    d1: str
    d2: str
    d3: str
    u: str
    issues: tuple[dict, ...]

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.invoice_doc_ids + self.payment_doc_ids


def _ledger_row(row_id, timestamp, account, debit, credit, entity_id,
                invoice_id=None, payment_id=None, currency="KZT") -> dict:
    return {
        "row_id": row_id, "timestamp": timestamp, "account": account,
        "debit_minor": debit, "credit_minor": credit, "currency": currency,
        "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
    }


def _invoice(doc_id, invoice_id, entity_id, issued_at, due_at, total, po_ref, service_period,
             currency="KZT") -> dict:
    return {
        "doc_id": doc_id, "invoice_id": invoice_id, "entity_id": entity_id,
        "issued_at": issued_at, "due_at": due_at, "currency": currency,
        "total_minor": total,
        "line_items": [{"description": "Consulting", "quantity": 1,
                        "unit_price_minor": total, "amount_minor": total}],
        "po_ref": po_ref, "service_period": service_period,
    }


def _mini_public() -> dict:
    """Public data shared by mini and clean.

    D1 doc_id "7" (invoice_id "Kx") and D2 doc_id "kX" (invoice_id "5") are a
    duplicate pair; D3 is a payment with doc_id "5" and payment_id "7".
    Collisions on purpose: "7" is a row_id, a doc_id and a payment_id; "5" is
    an invoice_id and a doc_id; "Kx"/"kX" differ only by case.
    KZT debits 30000 = credits 30000.
    """
    ledger = [
        _ledger_row("7", "2026-08-03T10:00:00+05:00", "AR", 10000, 0, "c9", invoice_id="Kx"),
        _ledger_row("B7", "2026-08-17T10:00:00+05:00", "AR", 10000, 0, "c9", invoice_id="5"),
        _ledger_row("a", "2026-08-17T10:00:00+05:00", "REVENUE", 0, 20000, "c9"),
        _ledger_row("0042", "2026-08-20T09:30:00+05:00", "CASH", 10000, 0, "c9", payment_id="7"),
        _ledger_row("b7", "2026-08-20T09:30:00+05:00", "AR", 0, 10000, "c9",
                    invoice_id="Kx", payment_id="7"),
    ]
    invoices = [
        _invoice("7", "Kx", "c9", "2026-08-03T10:00:00+05:00", "2026-09-02T23:59:59+05:00",
                 10000, "po 11", "2026-08"),
        _invoice("kX", "5", "c9", "2026-08-17T10:00:00+05:00", "2026-09-16T23:59:59+05:00",
                 10000, "po 11", "2026-08"),
    ]
    payments = [
        {"doc_id": "5", "payment_id": "7", "timestamp": "2026-08-20T09:30:00+05:00",
         "currency": "KZT", "amount_minor": 10000, "invoice_refs": ["Kx"]},
    ]
    fx = [{"date": "2026-08-03", "source_currency": "USD", "target_currency": "KZT", "rate": "470.25"}]
    return {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}


MINI_ISSUE = {
    "issue_id": "1",
    "type": "duplicate_invoice",
    "affected_ids": ["7", "kX"],
    "expected_amount_minor": 10000,
    "currency": "KZT",
    "evidence_sets": [["7", "kX"]],
}


def _provenance(episode_id: str, injections: list[dict]) -> dict:
    return {
        "schema_version": "1.0", "episode_id": episode_id, "generator_version": "test-factory",
        "seed": None, "attempt": 0, "team_config": {}, "injections": injections, "distractors": [],
    }


def _describe(path: Path, episode_id: str, data: dict, issues: list[dict], r: tuple, d: tuple,
              u: str) -> EpisodeFixture:
    return EpisodeFixture(
        path=path,
        episode_id=episode_id,
        row_ids=tuple(x["row_id"] for x in data["ledger"]),
        invoice_doc_ids=tuple(x["doc_id"] for x in data["invoices"]),
        invoice_ids=tuple(x["invoice_id"] for x in data["invoices"]),
        payment_doc_ids=tuple(x["doc_id"] for x in data["payments"]),
        payment_ids=tuple(x["payment_id"] for x in data["payments"]),
        fx_ids=_fx_ids(data["fx"]),
        r1=r[0], r2=r[1], r3=r[2], d1=d[0], d2=d[1], d3=d[2], u=u,
        issues=tuple(copy.deepcopy(issues)),
    )


def _fx_ids(rows: list[dict]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    ids = []
    for row in rows:
        key = f"{row['date']}:{row['source_currency']}:{row['target_currency']}"
        seen[key] = seen.get(key, 0) + 1
        ids.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
    return tuple(ids)


def build_mini(directory: str | os.PathLike) -> EpisodeFixture:
    """Jury block-3 shape: 5 ledger rows, 2 invoices, 1 payment, 1 fx row, one duplicate_invoice."""
    episode_id = "ep-mini"
    data = _mini_public()
    issues = [copy.deepcopy(MINI_ISSUE)]
    injection = {
        "issue_ids": ["1"], "kind": "duplicate_invoice", "scenario": None,
        "description": "Invoice posted a second time under a new number",
        "changes": [
            {"file": "invoices.jsonl", "op": "add", "key": "kX", "before": None,
             "after": copy.deepcopy(data["invoices"][1])},
            {"file": "ledger.csv", "op": "add", "key": "B7", "before": None,
             "after": copy.deepcopy(data["ledger"][1])},
        ],
        "effects": [{"control": "C5", "currency": "KZT", "delta_minor": 10000}],
    }
    path = write_episode(directory, manifest=default_manifest(episode_id), issues=issues,
                         provenance=_provenance(episode_id, [injection]), **data)
    return _describe(path, episode_id, data, issues, ("7", "B7", "a"), ("7", "kX", "5"), "U")


def build_clean(directory: str | os.PathLike) -> EpisodeFixture:
    """Same public data as mini (only episode_id differs) and an empty truth."""
    episode_id = "ep-clean"
    data = _mini_public()
    path = write_episode(directory, manifest=default_manifest(episode_id), issues=[],
                         provenance=_provenance(episode_id, []), **data)
    return _describe(path, episode_id, data, [], ("7", "B7", "a"), ("7", "kX", "5"), "U")


def _large_row_id(i: int) -> str:
    # 37 is coprime with 1000, so bases are unique for i < 1000. File order is not ID order.
    base = (i * 37) % 1000
    return (str(base), f"Rw{base}", f"{base}x")[i % 3]


def build_large(directory: str | os.PathLike, n_rows: int = 120) -> EpisodeFixture:
    """At least 101 ledger rows (three pages of 50), a few documents, an empty truth."""
    if not 101 <= n_rows < 1000:
        raise ValueError("build_large needs 101 <= n_rows < 1000")
    episode_id = "ep-large"
    row_ids = [_large_row_id(i) for i in range(n_rows)]
    pairs = (("AR", "REVENUE"), ("CASH", "AR"), ("RENT_EXPENSE", "CASH"))
    ledger: list[dict] = []
    i = 0
    group = 0
    while i < n_rows:
        day = 1 + group % 28
        month = 7 + (group // 28) % 3
        ts = f"2026-{month:02d}-{day:02d}T1{group % 10}:00:00+05:00"
        amount = 1000 * (1 + group % 17)
        debit_account, credit_account = pairs[group % 3]
        if n_rows - i == 3:  # odd total: last group is two debits and one credit
            ledger.append(_ledger_row(row_ids[i], ts, debit_account, amount, 0, "c1"))
            ledger.append(_ledger_row(row_ids[i + 1], ts, debit_account, amount, 0, "c1"))
            ledger.append(_ledger_row(row_ids[i + 2], ts, credit_account, 0, 2 * amount, "c1"))
            i += 3
        else:
            ledger.append(_ledger_row(row_ids[i], ts, debit_account, amount, 0, "c1"))
            ledger.append(_ledger_row(row_ids[i + 1], ts, credit_account, 0, amount, "c1"))
            i += 2
        group += 1
    invoices = [
        # doc_id equal to a row_id on purpose.
        _invoice(row_ids[0], "9", "c1", "2026-07-01T10:00:00+05:00", "2026-07-31T23:59:59+05:00",
                 1000, None, None),
        _invoice("Q", "q", "c1", "2026-07-02T10:00:00+05:00", "2026-08-01T23:59:59+05:00",
                 2000, "po 2", "2026-07"),
        _invoice("9", "Rw37", "c2", "2026-07-03T10:00:00+05:00", "2026-08-02T23:59:59+05:00",
                 3000, None, "2026-07"),
    ]
    payments = [
        {"doc_id": "q ", "payment_id": "Q", "timestamp": "2026-07-05T12:00:00+05:00",
         "currency": "KZT", "amount_minor": 1000, "invoice_refs": ["9"]},
        {"doc_id": "10", "payment_id": "0010", "timestamp": "2026-07-06T12:00:00+05:00",
         "currency": "KZT", "amount_minor": -500, "invoice_refs": None},
    ]
    fx = [
        {"date": "2026-07-01", "source_currency": "USD", "target_currency": "KZT", "rate": "471.10"},
        {"date": "2026-07-02", "source_currency": "USD", "target_currency": "KZT", "rate": "470.80"},
        # Same key again: loads as "2026-07-01:USD:KZT#2".
        {"date": "2026-07-01", "source_currency": "USD", "target_currency": "KZT", "rate": "471.15"},
    ]
    data = {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}
    path = write_episode(directory, manifest=default_manifest(episode_id), issues=[],
                         provenance=_provenance(episode_id, []), **data)
    return _describe(path, episode_id, data, [], tuple(row_ids[:3]),
                     (invoices[0]["doc_id"], invoices[1]["doc_id"], payments[0]["doc_id"]), "U")


def write_fixture_folders(base: Path = FIXTURES_DIR) -> list[Path]:
    """(Re)write tests/fixtures/{mini,clean,large} for humans to look at."""
    return [build_mini(base / "mini").path, build_clean(base / "clean").path,
            build_large(base / "large").path]


if __name__ == "__main__":
    for written in write_fixture_folders():
        print(written)

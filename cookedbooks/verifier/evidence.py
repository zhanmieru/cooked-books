"""Evidence sufficiency: re-derive one issue from one evidence set alone.

The derivation sees only: the records the set names (ledger rows by
`row_id`, documents by `doc_id`; the canonical document ID of contract
section 4), the ledger rows linked to those documents by `invoice_id` /
`payment_id`, and `manifest`, `policy` and the whole of `fx.csv`. The rule
of the issue's type is run on that sub-book; the issue must come out with
the same type, affected IDs, amount and currency, and the set must contain
one of the canonical evidence sets the derivation relied on (so an fx_error
set must name the REVENUE row and the rate key actually used). Every ID
must exist through the tools.
"""

from __future__ import annotations

from typing import Iterable

from .books import Books, Config
from .predicates import FINDERS, issue_key

MAX_EVIDENCE = 10


def check_evidence(config: Config, tables: dict, existing_ids: Iterable[str], issue: dict,
                   evidence_set: object) -> list[str]:
    """Problems with this evidence set for this issue; empty list = sufficient."""
    if not (isinstance(evidence_set, list) and evidence_set and all(isinstance(i, str) for i in evidence_set)):
        return ["evidence set is not a non-empty list of strings"]
    problems = []
    if len(evidence_set) > MAX_EVIDENCE:
        problems.append(f"evidence set has {len(evidence_set)} IDs (limit {MAX_EVIDENCE})")
    existing = set(existing_ids)
    unknown = sorted(i for i in set(evidence_set) if i not in existing)
    if unknown:
        problems.append(f"evidence IDs not retrievable through the tools: {unknown}")
    finder = FINDERS.get(issue.get("type"))
    if finder is None:
        return problems + [f"unknown issue type {issue.get('type')!r}"]

    named = set(evidence_set)
    invoices = [r for r in tables["invoices"] if r.get("doc_id") in named]
    payments = [r for r in tables["payments"] if r.get("doc_id") in named]
    iids = {r.get("invoice_id") for r in invoices}
    pids = {r.get("payment_id") for r in payments}
    ledger = [
        r for r in tables["ledger"]
        if r.get("row_id") in named
        or (r.get("invoice_id") is not None and r.get("invoice_id") in iids)
        or (r.get("payment_id") is not None and r.get("payment_id") in pids)
    ]
    sub = Books(config, {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": list(tables["fx"])},
                partial=True)
    derived = [d for d in finder(sub).issues if issue_key(d) == issue_key(issue)]
    if not derived:
        got = [f"{d['affected_ids']} {d['expected_amount_minor']} {d['currency']}" for d in finder(sub).issues]
        return problems + [f"the {issue['type']} cannot be derived from this set alone (derived: {got or 'nothing'})"]
    if not any(set(canonical) <= named for canonical in derived[0]["evidence_sets"]):
        problems.append(f"set lacks records the derivation relies on; needs one of {derived[0]['evidence_sets']}")
    return problems

"""Issue finder: the four error rules of DESIGN.md section 5, on public data only.

`find_issues(books)` never looks at truth or provenance. It returns every
issue in the canonical form of the table "Когда это ошибка" (contract
section 4) and, separately, every anomaly: something the generator promises
never to produce ("Чего в книгах не бывает"), a C1/C2 violation, a
malformed posting, an undecidable FX lookup, or two issues that share a
document outside scenarios S1 and S2.

Legitimate look-alikes (recurring invoices, split payments, advances,
refunds, UTC/local date differences, weekend rates, delayed postings that
stay inside the period) fall out of the exact rules and are not reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from . import identities, money
from .books import Books, _sorted_ids

ISSUE_TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")
TYPE_CONTROL = {"duplicate_invoice": "C5", "unallocated_payment": "C4", "fx_error": "C3", "cutoff_error": "C6"}


def make_issue(issue_type: str, affected, amount: int, currency: str, evidence_sets) -> dict:
    """An issue in canonical form; ID lists are sorted only for stable output."""
    return {
        "type": issue_type,
        "affected_ids": list(_sorted_ids(affected)),
        "expected_amount_minor": amount,
        "currency": currency,
        "evidence_sets": sorted(list(_sorted_ids(s)) for s in evidence_sets),
    }


def issue_key(issue: dict) -> tuple:
    """(type, set of affected_ids, amount, currency): what truth must match."""
    return (issue["type"], frozenset(issue["affected_ids"]), issue["expected_amount_minor"], issue["currency"])


def evidence_key(issue: dict) -> frozenset:
    return frozenset(frozenset(s) for s in issue["evidence_sets"])


def anomaly(kind: str, ids, message: str) -> dict:
    return {"kind": kind, "ids": list(_sorted_ids(ids)), "message": message}


@dataclass
class FindResult:
    issues: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)

    def extend(self, other: "FindResult") -> None:
        self.issues.extend(other.issues)
        self.anomalies.extend(other.anomalies)

    def sort(self) -> "FindResult":
        self.issues.sort(key=lambda i: (i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"]))
        self.anomalies.sort(key=lambda a: (a["kind"], a["ids"], a["message"]))
        return self


def duplicate_issues(books: Books) -> FindResult:
    """Two invoices with one obligation key -> duplicate_invoice; three or more -> anomaly."""
    result = FindResult()
    for group in identities.c5_groups(books):
        doc_ids = [v.doc_id for v in group]
        if len(group) == 2:
            result.issues.append(make_issue("duplicate_invoice", doc_ids, group[0].total, group[0].currency,
                                            [doc_ids]))
        else:
            result.anomalies.append(anomaly("duplicate_group_too_large", doc_ids,
                                            f"{len(group)} invoices share one obligation key"))
    return result


def unallocated_issues(books: Books) -> FindResult:
    """Payment remainder with exactly one open invoice of equal remainder -> unallocated_payment."""
    result = FindResult()
    used_by: dict[str, list[str]] = defaultdict(list)
    for payment, candidates in identities.c4_pairs(books):
        ids = [payment.doc_id] + [c.doc_id for c in candidates]
        if len(candidates) > 1:
            result.anomalies.append(anomaly("ambiguous_open_invoice", ids,
                                            f"{len(candidates)} open invoices fit the payment remainder"))
            continue
        invoice = candidates[0]
        if invoice.remainder != payment.remainder:
            result.anomalies.append(anomaly(
                "unequal_open_invoice", ids,
                f"payment remainder {payment.remainder} != invoice remainder {invoice.remainder}"))
            continue
        used_by[invoice.doc_id].append(payment.doc_id)
        pair = [payment.doc_id, invoice.doc_id]
        result.issues.append(make_issue("unallocated_payment", pair, payment.remainder, payment.currency, [pair]))
    for invoice_doc, payment_docs in used_by.items():
        if len(payment_docs) > 1:
            result.anomalies.append(anomaly("open_invoice_shared", [invoice_doc] + payment_docs,
                                            "one open invoice is the counterpart of several payment remainders"))
    return result


def fx_issues(books: Books) -> FindResult:
    """Material difference between posted and correct base amount -> fx_error."""
    result = FindResult()
    cfg = books.config
    for item in identities.fx_diffs(books):
        doc_id = item.invoice.doc_id
        if item.correct is None:
            result.anomalies.append(anomaly("fx_rate_unavailable", [doc_id], item.reason))
            continue
        diff = item.diff
        if diff == 0:
            continue
        if money.is_material(diff, item.correct, cfg.min_minor, cfg.min_relative):
            evidence = [doc_id, str(item.invoice.revenue_row.get("row_id")), item.rate.key]
            result.issues.append(make_issue("fx_error", [doc_id], diff, cfg.base, [evidence]))
        else:
            result.anomalies.append(anomaly("fx_below_materiality", [doc_id],
                                            f"difference {diff} is above zero but below materiality"))
    return result


def cutoff_issues(books: Books) -> FindResult:
    """issued_at and posting on different sides of the period -> cutoff_error."""
    result = FindResult()
    for view in identities.cutoff_mismatches(books):
        ar, revenue = str(view.ar_row.get("row_id")), str(view.revenue_row.get("row_id"))
        result.issues.append(make_issue("cutoff_error", [view.doc_id], view.total, view.currency,
                                        [[view.doc_id, ar], [view.doc_id, revenue]]))
    return result


FINDERS = {
    "duplicate_invoice": duplicate_issues,
    "unallocated_payment": unallocated_issues,
    "fx_error": fx_issues,
    "cutoff_error": cutoff_issues,
}


def _allowed_overlap(first: dict, second: dict, shared: set[str]) -> bool:
    """S1: fx_error + unallocated_payment on one invoice. S2: duplicate_invoice + fx_error on one copy."""
    types = {first["type"], second["type"]}
    fx = first if first["type"] == "fx_error" else second
    if types == {"fx_error", "unallocated_payment"}:
        return shared == set(fx["affected_ids"])
    if types == {"fx_error", "duplicate_invoice"}:
        return shared == set(fx["affected_ids"])
    return False


def overlap_anomalies(issues: list[dict]) -> list[dict]:
    """Two issues on one document are only allowed in the S1 and S2 shapes."""
    out = []
    for first, second in combinations(issues, 2):
        shared = set(first["affected_ids"]) & set(second["affected_ids"])
        if shared and not _allowed_overlap(first, second, shared):
            out.append(anomaly("overlapping_issues", shared,
                               f"{first['type']} and {second['type']} share a document"))
    return out


def control_anomalies(books: Books) -> list[dict]:
    """Structural problems and C1/C2 violations: books that should never exist."""
    out = [anomaly(f"malformed:{v.code}", v.ids, v.message) for v in identities.check_structure(books)]
    for violation in identities.check_c1(books) + identities.check_c2(books):
        out.append(anomaly(f"{violation.control.lower()}_violation:{violation.code}", violation.ids,
                           f"{violation.message} ({violation.currency})"))
    return out


def find_issues(books: Books, types=ISSUE_TYPES) -> FindResult:
    """Every issue the rules define, and every anomaly, from the public books alone."""
    result = FindResult()
    for issue_type in types:
        result.extend(FINDERS[issue_type](books))
    result.anomalies.extend(overlap_anomalies(result.issues))
    result.anomalies.extend(control_anomalies(books))
    return result.sort()

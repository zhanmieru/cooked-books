"""Match findings to truth issues one-to-one (EVALUATION_CONTRACT.md, section 6).

Pure functions, no I/O. Findings are assumed to be structurally valid
(`cookedbooks.findings.validate_findings` returned None); issues come from
`cookedbooks.grader.truth`.

A finding fits an issue iff all five conditions hold, checked in this order:

1. `type` equal (exact string);
2. `set(affected_ids) == set(issue.affected_ids)` (exact strings);
3. `currency` equal (exact string);
4. `abs(claimed - expected) <= tau`, tau = tolerance for the issue's type;
5. every evidence ID exists, and the evidence set covers at least one of the
   issue's `evidence_sets`.

An issue with `matchable: False` (malformed in truth.json) fits nothing and
is reported as `issue_malformed`.

Strings are never stripped or case folded, and IDs are never parsed.

Pairs: a maximum-cardinality bipartite matching (repeated findings are
separate vertices). Among maximum matchings the choice is the one the
contract prescribes: issues in `issue_id` order by code points (ties between
equal `issue_id`s by file position); for each, the smallest finding index
that still allows a matching of the full size given the pairs already fixed;
an issue that no such matching can cover stays unmatched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from cookedbooks.findings import MAX_EVIDENCE_IDS

# Reasons for an unmatched finding, in the order the conditions are checked.
CONDITION_REASONS = (
    "wrong_type",
    "wrong_ids",
    "wrong_currency",
    "amount_off",
    "evidence_unknown",
    "evidence_insufficient",
)
# Condition rank of each reason: both evidence reasons belong to condition 5.
_REASON_RANK = {
    "wrong_type": 0,
    "wrong_ids": 1,
    "wrong_currency": 2,
    "amount_off": 3,
    "evidence_unknown": 4,
    "evidence_insufficient": 4,
    "issue_malformed": -1,  # never preferred over a well-formed issue
}
UNMATCHED_REASONS = CONDITION_REASONS + ("issue_malformed", "duplicate", "no_issues")


@dataclass(frozen=True)
class UnmatchedFinding:
    """One finding that got no issue: why, and which issue was its best candidate."""

    finding_index: int
    reason: str
    issue_id: str | None


@dataclass(frozen=True)
class MatchResult:
    """`pairs` holds `(issue_id, finding_index)` sorted; `unmatched` is in finding order."""

    pairs: tuple[tuple[str, int], ...]
    unmatched: tuple[UnmatchedFinding, ...]

    @property
    def tp(self) -> int:
        return len(self.pairs)


def mismatch_reason(
    finding: Mapping, issue: Mapping, existing_ids: AbstractSet[str], tolerance: int
) -> str | None:
    """Return None if `finding` fits `issue`, else the reason of the first failing condition.

    For condition 5 an unknown evidence ID (`evidence_unknown`) is reported
    before a missing evidence set (`evidence_insufficient`). An issue marked
    `matchable: False` gives `issue_malformed` without looking at its fields.
    """
    if issue.get("matchable", True) is False:
        return "issue_malformed"
    if finding["type"] != issue["type"]:
        return "wrong_type"
    if set(finding["affected_ids"]) != set(issue["affected_ids"]):
        return "wrong_ids"
    if finding["currency"] != issue["currency"]:
        return "wrong_currency"
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > tolerance:
        return "amount_off"
    evidence = set(finding["evidence_ids"])
    if not evidence.issubset(existing_ids):
        return "evidence_unknown"
    if not any(set(required) <= evidence for required in issue["evidence_sets"]):
        return "evidence_insufficient"
    return None


def fits(finding: Mapping, issue: Mapping, existing_ids: AbstractSet[str], tolerance: int) -> bool:
    """True iff all five match conditions hold."""
    return mismatch_reason(finding, issue, existing_ids, tolerance) is None


def can_share_finding(
    a: Mapping, b: Mapping, existing_ids: AbstractSet[str], tolerances: Mapping[str, int]
) -> bool:
    """True iff some structurally valid finding fits both issues `a` and `b` ("twin" issues).

    Exact condition: both matchable, same `type`, same `affected_ids` set, same
    `currency`, `|expected_a - expected_b| <= 2 * tau` (one integer claim within
    tau of both), and an evidence set of each whose union exists entirely and
    has at most MAX_EVIDENCE_IDS IDs. With twins in truth, a repeated finding
    can be a true positive (jury example M11).
    """
    if a.get("matchable", True) is False or b.get("matchable", True) is False:
        return False
    if a["type"] != b["type"] or a["currency"] != b["currency"]:
        return False
    if set(a["affected_ids"]) != set(b["affected_ids"]):
        return False
    tolerance = tolerances.get(a["type"], 0)
    if abs(a["expected_amount_minor"] - b["expected_amount_minor"]) > 2 * tolerance:
        return False
    for set_a in a["evidence_sets"]:
        for set_b in b["evidence_sets"]:
            union = set(set_a) | set(set_b)
            if len(union) <= MAX_EVIDENCE_IDS and union.issubset(existing_ids):
                return True
    return False


def _matching_size(issues: Sequence[int], edges: Mapping[int, Sequence[int]], banned: AbstractSet[int]) -> int:
    """Size of a maximum matching between `issues` and the findings not in `banned` (Kuhn's algorithm)."""
    owner: dict[int, int] = {}  # finding index -> issue

    def augment(issue: int, seen: set[int]) -> bool:
        for finding in edges[issue]:
            if finding in banned or finding in seen:
                continue
            seen.add(finding)
            if finding not in owner or augment(owner[finding], seen):
                owner[finding] = issue
                return True
        return False

    return sum(1 for issue in issues if augment(issue, set()))


def choose_pairs(order: Sequence[int], edges: Mapping[int, Sequence[int]]) -> list[tuple[int, int]]:
    """Pick the contract's maximum matching. `order` lists issues by priority; `edges[i]` is sorted.

    Returns `(issue, finding)` pairs in `order` order.
    """
    target = _matching_size(order, edges, frozenset())
    used: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for position, issue in enumerate(order):
        if target == 0:
            break
        later = order[position + 1:]
        for finding in edges[issue]:
            if finding in used:
                continue
            if 1 + _matching_size(later, edges, used | {finding}) == target:
                used.add(finding)
                chosen.append((issue, finding))
                target -= 1
                break
    return chosen


def match(
    findings: Sequence[Mapping],
    issues: Sequence[Mapping],
    existing_ids: AbstractSet[str],
    tolerances: Mapping[str, int],
) -> MatchResult:
    """Match `findings` to `issues` and explain every finding left over.

    `tolerances` maps an issue type to its tau; a type that is absent has tau 0.

    Reason for an unmatched finding:
    - `no_issues` when the truth is empty;
    - `duplicate` when it fits at least one issue (every such issue was taken
      by another finding, or the matching would not be maximum); the issue
      named is the first one it fits in issue order;
    - otherwise the reason of its best candidate: the issue for which the
      finding gets furthest through the conditions in the order above
      (type, ids, currency, amount, evidence), ties broken by issue order
      (`issue_id` by code points, then file position). A malformed issue is
      the best candidate only when every issue is malformed.
    """
    order = sorted(range(len(issues)), key=lambda k: (issues[k]["issue_id"], k))
    reasons: dict[tuple[int, int], str | None] = {}
    edges: dict[int, list[int]] = {k: [] for k in order}
    for k in order:
        issue = issues[k]
        tolerance = tolerances.get(issue["type"], 0) if isinstance(issue["type"], str) else 0
        for index, finding in enumerate(findings):
            reason = mismatch_reason(finding, issue, existing_ids, tolerance)
            reasons[(k, index)] = reason
            if reason is None:
                edges[k].append(index)

    chosen = choose_pairs(order, edges)
    pairs = tuple(sorted((issues[k]["issue_id"], index) for k, index in chosen))
    matched = {index for _, index in chosen}

    unmatched: list[UnmatchedFinding] = []
    for index in range(len(findings)):
        if index in matched:
            continue
        if not issues:
            unmatched.append(UnmatchedFinding(index, "no_issues", None))
            continue
        fitted = [k for k in order if reasons[(k, index)] is None]
        if fitted:
            unmatched.append(UnmatchedFinding(index, "duplicate", issues[fitted[0]]["issue_id"]))
            continue
        best = order[0]
        for k in order[1:]:
            if _REASON_RANK[reasons[(k, index)]] > _REASON_RANK[reasons[(best, index)]]:
                best = k
        unmatched.append(UnmatchedFinding(index, reasons[(best, index)], issues[best]["issue_id"]))
    return MatchResult(pairs=pairs, unmatched=tuple(unmatched))

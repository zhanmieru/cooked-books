"""Structure check for an agent's findings (EVALUATION_CONTRACT.md, section 5).

This module never touches ground truth. The environment imports it in closed
mode to accept or reject a `submit`, and the grader imports it before scoring.
It must not import `cookedbooks.grader`.

Everything is checked on the raw values, before any set normalisation: eleven
`evidence_ids` of which ten repeat one ID is still a structural error.

Things that are deliberately NOT structural errors (they validate fine and then
simply fail to match): an unknown `type` string, an unknown currency, IDs that
do not exist, empty `affected_ids`, empty `evidence_ids`, an `episode_id` that
differs from the manifest, and extra unknown keys at any level.
"""

from __future__ import annotations

MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
ERROR_TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")

# The six fields every finding must carry, in the order they are checked.
FINDING_FIELDS = (
    "type",
    "affected_ids",
    "claimed_amount_minor",
    "currency",
    "evidence_ids",
    "explanation",
)

# Every reason code validate_findings can return, for documentation and tests.
REASON_CODES = (
    "not_an_object",
    "missing_episode_id",
    "bad_episode_id",
    "missing_findings",
    "findings_not_a_list",
    "too_many_findings",
    "finding_not_an_object",
    "missing_field",
    "bad_type",
    "bad_affected_ids",
    "bad_amount",
    "bad_currency",
    "bad_evidence_ids",
    "too_many_evidence_ids",
    "bad_explanation",
    "explanation_too_long",
)


def _is_list_of_str(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_json_int(value: object) -> bool:
    """True for a JSON integer. `bool` is an `int` subclass in Python, so it is excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_finding(finding: object) -> str | None:
    if not isinstance(finding, dict):
        return "finding_not_an_object"
    for field in FINDING_FIELDS:
        if field not in finding:
            return "missing_field"
    if not isinstance(finding["type"], str):
        return "bad_type"
    if not _is_list_of_str(finding["affected_ids"]):
        return "bad_affected_ids"
    if not _is_json_int(finding["claimed_amount_minor"]):
        return "bad_amount"
    if not isinstance(finding["currency"], str):
        return "bad_currency"
    evidence = finding["evidence_ids"]
    if not isinstance(evidence, list):
        return "bad_evidence_ids"
    if len(evidence) > MAX_EVIDENCE_IDS:  # raw length, repeats count
        return "too_many_evidence_ids"
    if not _is_list_of_str(evidence):
        return "bad_evidence_ids"
    explanation = finding["explanation"]
    if not isinstance(explanation, str):
        return "bad_explanation"
    if len(explanation) > MAX_EXPLANATION_CHARS:  # len(str) counts code points
        return "explanation_too_long"
    return None


def validate_findings(findings: object) -> str | None:
    """Return None if `findings` is structurally valid, else a short stable reason code.

    The codes are listed in `REASON_CODES`. The first problem found wins: the
    top-level object, then the finding count, then each finding in list order.
    """
    if not isinstance(findings, dict):
        return "not_an_object"
    if "episode_id" not in findings:
        return "missing_episode_id"
    if not isinstance(findings["episode_id"], str):
        return "bad_episode_id"
    if "findings" not in findings:
        return "missing_findings"
    items = findings["findings"]
    if not isinstance(items, list):
        return "findings_not_a_list"
    if len(items) > MAX_FINDINGS:
        return "too_many_findings"
    for finding in items:
        reason = _validate_finding(finding)
        if reason is not None:
            return reason
    return None

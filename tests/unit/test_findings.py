"""Structure check of a submission (EVALUATION_CONTRACT.md, section 5)."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

import cookedbooks.findings as findings_module
from cookedbooks.findings import (
    ERROR_TYPES,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    REASON_CODES,
    validate_findings,
)


def good_finding() -> dict:
    return {
        "type": "duplicate_invoice",
        "affected_ids": ["7", "kX"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["7", "kX"],
        "explanation": "Same order billed twice.",
    }


def submission(*items: dict) -> dict:
    return {"episode_id": "ep-test", "findings": list(items)}


def with_field(field: str, value: object) -> dict:
    finding = good_finding()
    finding[field] = value
    return submission(finding)


def test_constants():
    assert (MAX_FINDINGS, MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS) == (30, 10, 1000)
    assert ERROR_TYPES == ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")


def test_valid_submission():
    assert validate_findings(submission(good_finding())) is None
    assert validate_findings(submission()) is None


@pytest.mark.parametrize("value", [None, [], "x", 1, (), [submission()]])
def test_not_an_object(value):
    assert validate_findings(value) == "not_an_object"


def test_top_level_fields():
    assert validate_findings({"findings": []}) == "missing_episode_id"
    assert validate_findings({"episode_id": 1, "findings": []}) == "bad_episode_id"
    assert validate_findings({"episode_id": None, "findings": []}) == "bad_episode_id"
    assert validate_findings({"episode_id": "e"}) == "missing_findings"
    assert validate_findings({"episode_id": "e", "findings": {}}) == "findings_not_a_list"
    assert validate_findings({"episode_id": "e", "findings": None}) == "findings_not_a_list"


def test_finding_count_limit():
    assert validate_findings(submission(*[good_finding() for _ in range(30)])) is None
    assert validate_findings(submission(*[good_finding() for _ in range(31)])) == "too_many_findings"


@pytest.mark.parametrize("item", [None, "x", 1, [], ["type"]])
def test_finding_not_an_object(item):
    assert validate_findings(submission(item)) == "finding_not_an_object"


@pytest.mark.parametrize("field", ["type", "affected_ids", "claimed_amount_minor", "currency",
                                   "evidence_ids", "explanation"])
def test_missing_field(field):
    finding = good_finding()
    del finding[field]
    assert validate_findings(submission(finding)) == "missing_field"


@pytest.mark.parametrize(("field", "value", "code"), [
    ("type", None, "bad_type"),
    ("type", 1, "bad_type"),
    ("type", ["duplicate_invoice"], "bad_type"),
    ("affected_ids", "7", "bad_affected_ids"),
    ("affected_ids", None, "bad_affected_ids"),
    ("affected_ids", [7], "bad_affected_ids"),
    ("affected_ids", ["7", None], "bad_affected_ids"),
    ("affected_ids", ("7",), "bad_affected_ids"),
    ("currency", None, "bad_currency"),
    ("currency", 398, "bad_currency"),
    ("evidence_ids", "7", "bad_evidence_ids"),
    ("evidence_ids", [7], "bad_evidence_ids"),
    ("evidence_ids", {"7": 1}, "bad_evidence_ids"),
    ("explanation", None, "bad_explanation"),
    ("explanation", 5, "bad_explanation"),
])
def test_wrong_field_types(field, value, code):
    assert validate_findings(with_field(field, value)) == code


@pytest.mark.parametrize("amount", ["10000", True, False, 10000.0, 1.5, None, [10000], float("nan")])
def test_amount_must_be_json_integer(amount):
    assert validate_findings(with_field("claimed_amount_minor", amount)) == "bad_amount"


@pytest.mark.parametrize("amount", [0, -1, 10000, 10**30])
def test_any_integer_amount_is_structurally_fine(amount):
    assert validate_findings(with_field("claimed_amount_minor", amount)) is None


def test_evidence_limit_counts_repeats_before_dedup():
    assert validate_findings(with_field("evidence_ids", ["7"] * 10)) is None
    assert validate_findings(with_field("evidence_ids", ["7"] * 10 + ["kX"])) == "too_many_evidence_ids"
    assert validate_findings(with_field("evidence_ids", ["7"] * 11)) == "too_many_evidence_ids"
    distinct = [str(i) for i in range(11)]
    assert validate_findings(with_field("evidence_ids", distinct)) == "too_many_evidence_ids"


def test_affected_ids_have_no_size_limit():
    assert validate_findings(with_field("affected_ids", ["7"] * 50)) is None


def test_explanation_limit_in_code_points():
    assert validate_findings(with_field("explanation", "a" * 1000)) is None
    assert validate_findings(with_field("explanation", "a" * 1001)) == "explanation_too_long"
    # 1000 non-ASCII code points (2-4 UTF-8 bytes each) are still fine.
    assert validate_findings(with_field("explanation", "ж" * 1000)) is None
    assert validate_findings(with_field("explanation", "😀" * 1000)) is None
    assert validate_findings(with_field("explanation", "😀" * 1001)) == "explanation_too_long"
    assert validate_findings(with_field("explanation", "")) is None


@pytest.mark.parametrize(("field", "value"), [
    ("type", "made_up_error"),
    ("type", ""),
    ("type", "Duplicate_Invoice"),
    ("currency", "XYZ"),
    ("currency", "kzt "),
    ("affected_ids", []),
    ("affected_ids", ["does-not-exist"]),
    ("evidence_ids", []),
    ("evidence_ids", ["U", "nope"]),
])
def test_not_structural_errors(field, value):
    assert validate_findings(with_field(field, value)) is None


def test_other_episode_id_and_extra_keys_are_fine():
    finding = good_finding()
    finding["confidence"] = 0.9
    value = {"episode_id": "someone-else", "findings": [finding], "notes": None}
    assert validate_findings(value) is None
    assert validate_findings({"episode_id": "", "findings": []}) is None


def test_first_bad_finding_wins_and_input_is_untouched():
    bad = good_finding()
    bad["explanation"] = "a" * 1001
    worse = good_finding()
    worse["claimed_amount_minor"] = True
    value = submission(good_finding(), bad, worse)
    before = copy.deepcopy(value)
    assert validate_findings(value) == "explanation_too_long"
    assert value == before


def test_every_returned_code_is_documented():
    source = Path(findings_module.__file__).read_text(encoding="utf-8")
    returned = {node.value.value for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)}
    assert returned == set(REASON_CODES)


def test_module_does_not_import_the_grader():
    source = Path(findings_module.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported <= {"__future__"}

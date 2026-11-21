"""Jury block 4, paired checks P01-P20 (docs/jury-checks.md, section 5).

The jury generates each closed episode D01-D20 once with
`judge_adapter.generate_episode`, copies the folder, reorders the records
and/or replaces every ID in the copy (truth and provenance included), and
loads both folders through `episode_dir`. We do the copy with our own
transformer (`cookedbooks.verify.transform`) in every relabel style, because
the jury's choice of new IDs is unknown and our code must not care.

| Pairs   | Recipes | Transformation |
|---------|---------|----------------|
| P01-P05 | D01-D05 | permutation    |
| P06-P10 | D06-D10 | relabelling    |
| P11-P20 | D11-D20 | both           |

Each case asserts the jury's four pass conditions (`check_pair`):

1. repeat: the same seed and recipe give value-equal files; a fixed scripted
   run (paging every table with cursors, lookups, a calculator call, errors,
   the oracle submit) gives identical observations, rewards and flags twice,
   after a reset that follows a finished or a half-done episode, and after
   an episode on another folder;
2. oracle: the oracle answer on the copy is accepted and scores perfectly;
   mapped back (or, for the colliding `int` style, mapped forward) it equals
   the original answer in types, affected IDs, amounts, currencies and
   evidence, compared as sets where the contract says sets;
3. score: `score_episode` gives the same seven keys on both folders for the
   oracle, a degraded answer and an invalid one, at 0 and at 37 inspection
   steps; the closed environment returns reward 0 on both;
4. no leak, nothing unavailable: every evidence ID of the copied truth is
   served by the tools as the mapped original record; the copied tables hold
   exactly the original records up to the mapping and order, and paging
   reaches all of them; manifest counts match; the observations hold only
   strings from the public files, the task text, error messages, cursors
   and calculator results (so no issue_id and nothing from provenance).

Extra, beyond the jury's wording: the independent verifier accepts the copy.
More seeds: `COOKEDBOOKS_PAIR_SEEDS=20 pytest -q tests/jury/test_block4_pairs.py`.
"""

from __future__ import annotations

import copy
import functools
import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import judge_adapter
from cookedbooks.env.errors import MESSAGES
from cookedbooks.episode.loader import TABLES, load_public
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.verifier import verify_episode
from cookedbooks.verify import transform
from cookedbooks.verify.transform import STYLES, map_findings, map_issue, map_record, restore_findings

MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
PAIR_SEEDS = tuple(range(int(os.environ.get("COOKEDBOOKS_PAIR_SEEDS", "0") or 0))) or (0, 20261122)
STEP_COUNTS = (0, 37)
PAGE_LIMIT = 50
RESERVE = 12  # actions kept back from paging for the lookups, errors and submit
CALC_EXPRESSION = "(0.1 + 0.2) * 3 - 1 / 8"
METRIC_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1", "return")
TABLE_FILE = {"ledger": "ledger.csv", "invoices": "invoices.jsonl", "payments": "payments.jsonl", "fx": "fx.csv"}
KEY_FIELD = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "id"}


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader-log.jsonl"))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("block4")


# --- small helpers ---------------------------------------------------------------------


def recipe(name: str) -> dict:
    path = MATRIX / f"{name}.json"
    if not path.is_file():
        pytest.fail(f"recipe missing: configs/matrix/{name}.json does not exist", pytrace=False)
    return json.loads(path.read_text(encoding="utf-8"))


def folder_values(folder: Path) -> dict[str, Any]:
    """Every episode file parsed: JSON by value, CSV as raw rows."""
    values: dict[str, Any] = {}
    for name in transform.EPISODE_FILES:
        path = folder / name
        if name.endswith(".csv"):
            header, rows, _ = transform.read_csv(path)
            values[name] = [header] + rows
        elif name.endswith(".jsonl"):
            values[name] = transform.read_jsonl(path)
        else:
            values[name] = transform.read_json(path)
    return values


def strings_in(value: Any) -> set[str]:
    """All string leaves of a JSON value (dict keys excluded)."""
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def finding_key(finding: dict) -> tuple:
    """A finding as the contract compares it: affected and evidence IDs are sets."""
    return (finding["type"], frozenset(finding["affected_ids"]), finding["claimed_amount_minor"],
            finding["currency"], frozenset(finding["evidence_ids"]))


def issue_key(issue: dict) -> tuple:
    return (issue["issue_id"], issue["type"], frozenset(issue["affected_ids"]),
            issue["expected_amount_minor"], issue["currency"],
            tuple(frozenset(s) for s in issue["evidence_sets"]))


def closed(folder: Path) -> dict:
    return {"mode": "closed", "episode_dir": str(folder), "team_config": {}}


# --- the scripted run --------------------------------------------------------------------


@dataclass
class Run:
    """One scripted episode: every (action, step result) and the records seen by paging."""

    transcript: list[Any]
    pages: dict[str, list[dict]]
    complete: bool  # every table was paged to the end


def scripted_run(env: Any, folder: Path, seed: int, findings: Any) -> Run:
    """Reset on `folder`, page every table, look a few records up, make errors, submit `findings`."""
    observation, info = env.reset(seed, closed(folder))
    transcript: list[Any] = [("reset", observation, info)]
    remaining = observation["remaining_steps"]

    def step(action: dict) -> dict:
        nonlocal remaining
        result = env.step(action)
        transcript.append((copy.deepcopy(action), *result))
        remaining = result[0]["remaining_steps"]
        return result[0]

    pages: dict[str, list[dict]] = {table: [] for table in TABLES}
    complete = True
    for table in TABLES:
        cursor = None
        while True:
            if remaining <= RESERVE:
                complete = False
                break
            obs = step({"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": PAGE_LIMIT}})
            pages[table].extend(obs["result"]["records"])
            cursor = obs["result"]["next_cursor"]
            if cursor is None:
                break
    for table in TABLES:
        if pages[table]:
            step({"name": "get_record", "args": {"table": table, "id": pages[table][0][KEY_FIELD[table]]}})
    for table in ("invoices", "payments"):
        if pages[table]:
            step({"name": "get_document", "args": {"doc_id": pages[table][-1]["doc_id"]}})
    step({"name": "calculator", "args": {"expression": CALC_EXPRESSION}})
    step({"name": "calculator", "args": {"expression": "1 / 0"}})
    step({"name": "get_record", "args": {"table": "journal", "id": "x"}})
    step({"name": "get_document", "args": {"doc_id": "\x00 not an id \x00"}})
    step({"name": "audit_everything", "args": {}})
    step({"name": "submit", "args": {"findings": findings}})
    step({"name": "list_records", "args": {"table": "ledger", "cursor": None}})  # after the end
    return Run(transcript, pages, complete)


def half_episode(env: Any, folder: Path, seed: int) -> None:
    """A few steps of an episode that is then abandoned."""
    env.reset(seed, closed(folder))
    env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 7}})
    env.step({"name": "calculator", "args": {"expression": "2 * 21"}})
    env.step({"name": "get_record", "args": {"table": "fx", "id": "nope"}})


# --- condition 1, once per original episode ---------------------------------------------


@dataclass
class Original:
    name: str
    seed: int
    folder: Path
    oracle: dict
    run: Run


@functools.lru_cache(maxsize=None)
def _original(name: str, seed: int, root: str) -> Original:
    """Generate the original twice and check condition 1 on it; cached per (recipe, seed)."""
    config = recipe(name)
    base = Path(root) / f"{name}-{seed}"
    first, second = base / "original", base / "repeat"
    judge_adapter.generate_episode(seed, copy.deepcopy(config), str(first))
    judge_adapter.generate_episode(seed, copy.deepcopy(config), str(second))
    assert folder_values(first) == folder_values(second), \
        f"[repeat] {name} seed {seed}: generating twice gives different files"

    oracle = oracle_findings(first)
    fresh = scripted_run(judge_adapter.make_env(), first, seed, oracle)
    env = judge_adapter.make_env()
    again = [scripted_run(env, first, seed, oracle) for _ in range(2)]  # reset after a finished episode
    half_episode(env, first, seed)
    after_half = scripted_run(env, first, seed, oracle)
    other = judge_adapter.make_env()
    scripted_run(other, second, seed + 1, {"episode_id": "x", "findings": []})
    on_other = scripted_run(other, first, seed, oracle)  # after an episode on another folder
    for label, run in (("second run", again[0]), ("run after a finished episode", again[1]),
                       ("run after a half-done episode", after_half), ("run after another folder", on_other)):
        assert run.transcript == fresh.transcript, f"[repeat] {name} seed {seed}: {label} differs from a fresh run"
    for entry in fresh.transcript[1:]:
        _, _, reward, _, _, info = entry
        assert reward == 0 and info == {}, f"[repeat] {name} seed {seed}: closed mode gave reward {reward}"
    return Original(name, seed, first, oracle, fresh)


# --- conditions 2-4 on one transformed copy ---------------------------------------------


def degraded(oracle: dict, folder: Path) -> dict:
    """An answer with one true finding missing and one false finding added."""
    answer = copy.deepcopy(oracle)
    findings = answer["findings"]
    if findings:
        bogus = copy.deepcopy(findings[0])
        bogus["claimed_amount_minor"] += 1
        findings.pop()
        findings.append(bogus)
    else:
        docs = [r["doc_id"] for r in load_public(folder).tables["invoices"][:2]]
        findings.append({"type": "duplicate_invoice", "affected_ids": docs, "claimed_amount_minor": 1,
                         "currency": "KZT", "evidence_ids": docs, "explanation": "made up"})
    return answer


def scores(folder: Path, findings: Any) -> list[dict]:
    return [judge_adapter.score_episode(str(folder), copy.deepcopy(findings), steps) for steps in STEP_COUNTS]


def check_oracle(orig: Original, dst: Path, mapping: dict, label: str) -> dict:
    """Condition 2. Returns the oracle answer on the copy."""
    answer = oracle_findings(dst)
    expected = map_findings(orig.oracle, mapping)
    assert answer["episode_id"] == orig.oracle["episode_id"], f"[oracle] {label}: episode_id changed"
    assert Counter(map(finding_key, answer["findings"])) == Counter(map(finding_key, expected["findings"])), \
        f"[oracle] {label}: oracle on the copy differs from the mapped original oracle"
    if not transform.is_ambiguous(mapping):
        back = restore_findings(answer, mapping)
        assert Counter(map(finding_key, back["findings"])) == Counter(map(finding_key, orig.oracle["findings"])), \
            f"[oracle] {label}: oracle on the copy, mapped back, differs from the original oracle"
    src_truth = transform.read_json(orig.folder / "truth.json")["issues"]
    dst_truth = transform.read_json(dst / "truth.json")["issues"]
    assert [issue_key(map_issue(i, mapping)) for i in src_truth] == [issue_key(i) for i in dst_truth], \
        f"[oracle] {label}: copied truth is not the mapped original truth"
    perfect = judge_adapter.score_episode(str(dst), copy.deepcopy(answer), 0)
    assert perfect["f1"] == 1 and perfect["fp"] == 0 and perfect["fn"] == 0, \
        f"[oracle] {label}: oracle on the copy is not accepted as perfect: {perfect}"
    return answer


def check_scores(orig: Original, dst: Path, mapping: dict, answer: dict, label: str) -> None:
    """Condition 3."""
    worse = degraded(orig.oracle, orig.folder)
    invalid = {"episode_id": orig.oracle["episode_id"]}
    for what, src_answer, dst_answer in (("oracle", orig.oracle, answer),
                                         ("degraded answer", worse, map_findings(worse, mapping)),
                                         ("invalid answer", invalid, invalid)):
        before, after = scores(orig.folder, src_answer), scores(dst, dst_answer)
        for steps, a, b in zip(STEP_COUNTS, before, after):
            assert set(a) == set(METRIC_KEYS) and a == b, \
                f"[score] {label}: {what} at {steps} inspection steps: original {a}, copy {b}"


def lookup_action(space: str, new_id: str) -> tuple[dict, str, str]:
    """The tool call that serves an evidence ID of `space`, and the (table, field) it must come back with."""
    if space == "row_id":
        return {"name": "get_record", "args": {"table": "ledger", "id": new_id}}, "ledger", "row_id"
    if space == "doc_id":
        return {"name": "get_document", "args": {"doc_id": new_id}}, "", "doc_id"
    if space == "invoice_id":
        return {"name": "get_record", "args": {"table": "invoices", "id": new_id}}, "invoices", "invoice_id"
    if space == "payment_id":
        return {"name": "get_record", "args": {"table": "payments", "id": new_id}}, "payments", "payment_id"
    return {"name": "get_record", "args": {"table": "fx", "id": new_id}}, "fx", "id"


SPACE_FIELDS = {"row_id": (("ledger", "row_id"),), "doc_id": (("invoices", "doc_id"), ("payments", "doc_id")),
                "invoice_id": (("invoices", "invoice_id"),), "payment_id": (("payments", "payment_id"),),
                "fx": (("fx", "id"),)}


def original_record(tables: dict, space: str, old_id: str) -> tuple[str, dict]:
    """(table, record) of an original evidence ID of `space`."""
    for table, field in SPACE_FIELDS[space]:
        for record in tables[table]:
            if record.get(field) == old_id:
                return table, record
    raise AssertionError(f"original evidence ID {old_id!r} is not a public record")


def evidence_space(old_id: str, mapping: dict) -> str:
    for space in transform.EVIDENCE_SPACES:
        if old_id in mapping[space]:
            return space
    return "fx"


def check_evidence(orig: Original, dst: Path, mapping: dict, seed: int, label: str) -> None:
    """Condition 4a: every evidence ID of the copied truth is served as the mapped original record."""
    src_tables = load_public(orig.folder).tables
    wanted = []
    for issue in transform.read_json(orig.folder / "truth.json")["issues"]:
        for evidence_set in issue["evidence_sets"]:
            for old_id in evidence_set:
                space = evidence_space(old_id, mapping)
                new_id = old_id if space == "fx" else mapping[space][old_id]
                table, record = original_record(src_tables, space, old_id)
                wanted.append((old_id, space, new_id, map_record(TABLE_FILE[table], record, mapping)))
    env = judge_adapter.make_env()
    remaining = 0
    for old_id, space, new_id, expected in wanted:
        if remaining <= 2:
            remaining = env.reset(seed, closed(dst))[0]["remaining_steps"]
        action, _, field = lookup_action(space, new_id)
        obs = env.step(action)[0]
        remaining = obs["remaining_steps"]
        assert obs["error"] is None, f"[evidence] {label}: evidence {new_id!r} ({space}, was {old_id!r}) is not served"
        got = obs["result"]
        assert got.get(field) == new_id and got == expected, \
            f"[evidence] {label}: {action} returned another record than the mapped original {old_id!r}"


def check_tables(orig: Original, dst: Path, mapping: dict, permute: bool, run: Run, label: str) -> None:
    """Condition 4b: same records up to mapping and order; paging reaches all; manifest counts hold."""
    src, new = load_public(orig.folder), load_public(dst)
    src_tables, new_tables = src.tables, new.tables
    assert new.manifest == src.manifest and new.policy == src.policy, f"[tables] {label}: manifest or policy changed"
    counts = new.manifest["record_counts"]
    for table in TABLES:
        mapped = [map_record(TABLE_FILE[table], r, mapping) for r in src_tables[table]]  # fx: unchanged
        assert Counter(map(canon, mapped)) == Counter(map(canon, new_tables[table])), \
            f"[tables] {label}: {table} does not hold the mapped original records"
        if not permute:
            assert mapped == list(new_tables[table]), f"[tables] {label}: {table} order changed without permutation"
        elif len(mapped) > 1:
            assert mapped != list(new_tables[table]), f"[tables] {label}: {table} was not reordered"
        assert counts[table] == len(new_tables[table]), f"[tables] {label}: record_counts.{table} is off"
        assert run.pages[table] == list(new_tables[table]), f"[tables] {label}: paging {table} misses records"
    assert new.manifest["document_count"] == len(new_tables["invoices"]) + len(new_tables["payments"])
    assert run.complete, f"[tables] {label}: the scripted walk could not page every table"


def check_no_leak(folder: Path, run: Run, label: str) -> None:
    """Condition 4c: observations hold only public strings and fixed texts."""
    public = load_public(folder)
    allowed = strings_in(list(public.tables.values())) | strings_in(public.manifest) | strings_in(public.policy)
    allowed |= set(MESSAGES) | set(MESSAGES.values())
    observed: set[str] = set()
    for entry in run.transcript:
        if entry[0] == "reset":
            observation = dict(entry[1])
            allowed.add(observation.pop("task"))
            observed |= strings_in(observation) | strings_in(entry[2])
            continue
        action, observation, _, _, _, info = entry
        result = observation["result"]
        if action["name"] == "calculator" and isinstance(result, str):
            allowed.add(result)
        if action["name"] == "list_records" and isinstance(result, dict) and result.get("next_cursor"):
            allowed.add(result["next_cursor"])
        observed |= strings_in(observation) | strings_in(info)
    leaked = observed - allowed
    assert not leaked, f"[leak] {label}: observations hold non-public strings {sorted(leaked)[:5]}"
    truth = transform.read_json(folder / "truth.json")
    issue_ids = {i["issue_id"] for i in truth["issues"]}
    assert not (issue_ids & observed) - allowed, f"[leak] {label}: an issue_id reached an observation"
    provenance = transform.read_json(folder / "provenance.json")
    hidden = strings_in(provenance) - allowed
    assert not hidden & observed, f"[leak] {label}: provenance text reached an observation"
    served = {canon(r) for table in run.pages.values() for r in table}
    current = {canon(r) for table in public.tables.values() for r in table}
    for change in transform._changes_of(provenance):
        before = change.get("before")
        if isinstance(before, dict) and canon(before) not in current:
            assert canon(before) not in served, f"[leak] {label}: a pre-injection record was served"


def check_pair(name: str, seed: int, *, permute: bool, relabel: bool, style: str, root: Path) -> None:
    """All four jury conditions (and the verifier) for one recipe, seed and transformation."""
    orig = _original(name, seed, str(root))
    label = f"{name} seed {seed} permute={permute} relabel={relabel} style={style}"
    dst = root / f"{name}-{seed}" / f"copy-{int(permute)}{int(relabel)}-{style}"
    mapping = transform.transform_episode(orig.folder, dst, permute=permute, relabel=relabel, seed=seed, style=style)

    answer = check_oracle(orig, dst, mapping, label)
    check_scores(orig, dst, mapping, answer, label)

    env = judge_adapter.make_env()
    run = scripted_run(env, dst, seed, answer)
    submit = next(e for e in run.transcript[1:] if e[0]["name"] == "submit")
    assert submit[1] == {"result": {"accepted": True}, "error": None, "remaining_steps": submit[1]["remaining_steps"]}, \
        f"[oracle] {label}: the oracle submit on the copy was not accepted"
    assert all(e[2] == 0 and e[5] == {} for e in run.transcript[1:]), f"[score] {label}: closed mode gave a reward"
    assert scripted_run(env, orig.folder, seed, orig.oracle).transcript == orig.run.transcript, \
        f"[repeat] {label}: the original replays differently after an episode on the copy"

    check_evidence(orig, dst, mapping, seed, label)
    check_tables(orig, dst, mapping, permute, run, label)
    check_no_leak(dst, run, label)
    check_no_leak(orig.folder, orig.run, label + " (original)")

    report = verify_episode(dst)
    assert report.ok, f"[verifier] {label}: the verifier rejects the copy:\n{report.render()}"


# --- the twenty pairs ----------------------------------------------------------------------

PERMUTE = pytest.mark.parametrize("seed,style", [(s, "hex") for s in PAIR_SEEDS], ids=lambda v: str(v))
RELABEL = pytest.mark.parametrize("seed,style", [(s, st) for s in PAIR_SEEDS for st in STYLES], ids=lambda v: str(v))


def _permuted(name: str, seed: int, style: str, root: Path) -> None:
    check_pair(name, seed, permute=True, relabel=False, style=style, root=root)


def _relabelled(name: str, seed: int, style: str, root: Path) -> None:
    check_pair(name, seed, permute=False, relabel=True, style=style, root=root)


def _both(name: str, seed: int, style: str, root: Path) -> None:
    check_pair(name, seed, permute=True, relabel=True, style=style, root=root)


@PERMUTE
def test_P01_D01_permuted(seed, style, workspace):
    _permuted("D01", seed, style, workspace)


@PERMUTE
def test_P02_D02_permuted(seed, style, workspace):
    _permuted("D02", seed, style, workspace)


@PERMUTE
def test_P03_D03_permuted(seed, style, workspace):
    _permuted("D03", seed, style, workspace)


@PERMUTE
def test_P04_D04_permuted(seed, style, workspace):
    _permuted("D04", seed, style, workspace)


@PERMUTE
def test_P05_D05_permuted(seed, style, workspace):
    _permuted("D05", seed, style, workspace)


@RELABEL
def test_P06_D06_relabelled(seed, style, workspace):
    _relabelled("D06", seed, style, workspace)


@RELABEL
def test_P07_D07_relabelled(seed, style, workspace):
    _relabelled("D07", seed, style, workspace)


@RELABEL
def test_P08_D08_relabelled(seed, style, workspace):
    _relabelled("D08", seed, style, workspace)


@RELABEL
def test_P09_D09_relabelled(seed, style, workspace):
    _relabelled("D09", seed, style, workspace)


@RELABEL
def test_P10_D10_relabelled(seed, style, workspace):
    _relabelled("D10", seed, style, workspace)


@RELABEL
def test_P11_D11_permuted_relabelled(seed, style, workspace):
    _both("D11", seed, style, workspace)


@RELABEL
def test_P12_D12_permuted_relabelled(seed, style, workspace):
    _both("D12", seed, style, workspace)


@RELABEL
def test_P13_D13_permuted_relabelled(seed, style, workspace):
    _both("D13", seed, style, workspace)


@RELABEL
def test_P14_D14_permuted_relabelled(seed, style, workspace):
    _both("D14", seed, style, workspace)


@RELABEL
def test_P15_D15_permuted_relabelled(seed, style, workspace):
    _both("D15", seed, style, workspace)


@RELABEL
def test_P16_D16_permuted_relabelled(seed, style, workspace):
    _both("D16", seed, style, workspace)


@RELABEL
def test_P17_D17_permuted_relabelled(seed, style, workspace):
    _both("D17", seed, style, workspace)


@RELABEL
def test_P18_D18_permuted_relabelled(seed, style, workspace):
    _both("D18", seed, style, workspace)


@RELABEL
def test_P19_D19_permuted_relabelled(seed, style, workspace):
    _both("D19", seed, style, workspace)


@RELABEL
def test_P20_D20_permuted_relabelled(seed, style, workspace):
    _both("D20", seed, style, workspace)

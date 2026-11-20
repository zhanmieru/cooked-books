"""`verify_episode`: check one generated episode from its files alone.

Order of work: load the public files, check formats, build the Config,
read truth.json and provenance.json strictly, find issues on the published
books, revert the provenance and check the clean books, replay injections
one by one, re-derive every truth issue from each of its evidence sets.
Every stage is guarded, so a strange folder gives a failed report, never an
exception.
"""

from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from cookedbooks.episode.loader import load_public

from . import identities
from .books import Books, Config, config_from
from .evidence import MAX_EVIDENCE, check_evidence
from .formats import check_formats
from .predicates import ISSUE_TYPES, TYPE_CONTROL, evidence_key, find_issues, issue_key
from .report import Report
from .revert import _canon, KEY_FIELD, apply_injection, revert

SCENARIOS = ("S1", "S2")


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
    except FileNotFoundError:
        return None, f"{path.name} is missing"
    except (OSError, UnicodeDecodeError):
        return None, f"{path.name} cannot be read as UTF-8"
    try:
        return json.loads(text, parse_constant=_reject_constant), None
    except ValueError as exc:
        return None, f"{path.name} is not valid JSON: {exc}"


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number {token}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _str_list(value: object, *, non_empty: bool = True, unique: bool = True) -> bool:
    return (isinstance(value, list) and (bool(value) or not non_empty)
            and all(isinstance(v, str) for v in value) and (not unique or len(set(value)) == len(value)))


# --- truth and provenance, read strictly ------------------------------------


def read_truth(path: Path, episode_id: Any, report: Report) -> list[dict] | None:
    """Well-formed truth issues, or None if truth.json is unusable. Problems go to `truth_matches`."""
    value, error = _read_json(path / "truth.json")
    if error:
        report.fail("truth_matches", "truth_unreadable", error)
        return None
    if not isinstance(value, dict) or not isinstance(value.get("issues"), list):
        report.fail("truth_matches", "truth_malformed", "truth.json must be an object with an issues list")
        return None
    if value.get("schema_version") != "1.0":
        report.fail("truth_matches", "truth_malformed", "truth.json schema_version is not '1.0'")
    if value.get("episode_id") != episode_id:
        report.fail("truth_matches", "truth_episode_id", "truth.json episode_id differs from the manifest")
    issues, seen = [], Counter()
    for position, raw in enumerate(value["issues"]):
        problems = []
        if not isinstance(raw, dict):
            report.fail("truth_matches", "truth_issue_malformed", f"issue {position} is not an object")
            continue
        if not isinstance(raw.get("issue_id"), str):
            problems.append("issue_id is not a string")
        if raw.get("type") not in ISSUE_TYPES:
            problems.append(f"unknown type {raw.get('type')!r}")
        if not _str_list(raw.get("affected_ids")):
            problems.append("affected_ids is not a non-empty list of distinct strings")
        if not _is_int(raw.get("expected_amount_minor")):
            problems.append("expected_amount_minor is not an integer")
        if not isinstance(raw.get("currency"), str):
            problems.append("currency is not a string")
        sets = raw.get("evidence_sets")
        if not (isinstance(sets, list) and sets and all(_str_list(s) and len(s) <= MAX_EVIDENCE for s in sets)):
            problems.append(f"evidence_sets must be a non-empty list of non-empty sets of at most {MAX_EVIDENCE} distinct IDs")
        if problems:
            report.fail("truth_matches", "truth_issue_malformed", f"issue {position}: " + "; ".join(problems),
                        issue_id=raw.get("issue_id"))
            continue
        seen[raw["issue_id"]] += 1
        issues.append(raw)
    for issue_id, count in sorted(seen.items()):
        if count > 1:
            report.fail("truth_matches", "truth_issue_id_repeats", f"issue_id {issue_id!r} is used {count} times")
    return issues


def read_provenance(path: Path, episode_id: Any, report: Report) -> dict | None:
    value, error = _read_json(path / "provenance.json")
    if error:
        for condition in ("clean_books", "injections"):
            report.fail(condition, "provenance_unreadable", error)
        return None
    if not isinstance(value, dict) or not isinstance(value.get("injections"), list):
        for condition in ("clean_books", "injections"):
            report.fail(condition, "provenance_malformed", "provenance.json must be an object with an injections list")
        return None
    if value.get("schema_version") != "1.0":
        report.fail("injections", "provenance_malformed", "provenance.json schema_version is not '1.0'")
    if value.get("episode_id") != episode_id:
        report.fail("injections", "provenance_episode_id", "provenance.json episode_id differs from the manifest")
    for name, ok in (("generator_version", isinstance(value.get("generator_version"), str)),
                     ("attempt", _is_int(value.get("attempt")) and value.get("attempt") >= 0),
                     ("team_config", isinstance(value.get("team_config"), dict))):
        if not ok:
            report.fail("injections", "provenance_malformed", f"provenance.json {name} is missing or malformed")
    for position, injection in enumerate(value["injections"]):
        problems = []
        if not isinstance(injection, dict):
            report.fail("injections", "injection_malformed", f"injection {position} is not an object")
            continue
        if not _str_list(injection.get("issue_ids")):
            problems.append("issue_ids is not a non-empty list of distinct strings")
        if not isinstance(injection.get("kind"), str):
            problems.append("kind is not a string")
        if injection.get("scenario") not in (None,) + SCENARIOS:
            problems.append(f"scenario {injection.get('scenario')!r} is not null, S1 or S2")
        if not (isinstance(injection.get("changes"), list) and injection["changes"]):
            problems.append("changes is not a non-empty list")
        if not isinstance(injection.get("effects"), list):
            problems.append("effects is not a list")
        if problems:
            report.fail("injections", "injection_malformed", f"injection {position}: " + "; ".join(problems))
    distractors = value.get("distractors", [])
    if not isinstance(distractors, list):
        report.fail("injections", "distractors_malformed", "distractors is not a list")
    return value


# --- the stages -------------------------------------------------------------


def _compare_with_truth(found: list[dict], truth: list[dict], report: Report) -> None:
    """Multiset equality on (type, affected set, amount, currency) and equal evidence sets."""
    remaining = list(found)
    unmatched_truth = []
    for issue in truth:
        position = next((i for i, f in enumerate(remaining) if issue_key(f) == issue_key(issue)), None)
        if position is None:
            unmatched_truth.append(issue)
            continue
        match = remaining.pop(position)
        if evidence_key(match) != evidence_key(issue):
            report.fail("truth_matches", "evidence_sets_differ",
                        f"truth issue {issue['issue_id']!r}: evidence sets differ from the canonical ones",
                        truth=issue["evidence_sets"], expected=match["evidence_sets"])
    for issue in unmatched_truth:
        near = next((f for f in remaining if f["type"] == issue["type"]
                     and set(f["affected_ids"]) & set(issue["affected_ids"])), None)
        if near is None:
            report.fail("truth_matches", "extra_truth_issue",
                        f"truth issue {issue['issue_id']!r} ({issue['type']} {sorted(issue['affected_ids'])}) "
                        "is not in the books", issue=issue)
            continue
        remaining.remove(near)
        if set(near["affected_ids"]) != set(issue["affected_ids"]):
            code = "wrong_affected_ids"
        elif near["currency"] != issue["currency"]:
            code = "wrong_currency"
        else:
            code = "wrong_amount"
        report.fail("truth_matches", code,
                    f"truth issue {issue['issue_id']!r} disagrees with the books ({code})",
                    truth=issue, books=near)
    for issue in remaining:
        report.fail("truth_matches", "missing_truth_issue",
                    f"the books contain a {issue['type']} on {issue['affected_ids']} that truth.json lacks",
                    issue=issue)


def _effects_problems(injection: dict, declared: list[dict], expected_controls: set[str]) -> list[str]:
    effects = injection.get("effects") or []
    problems = []
    controls = set()
    for effect in effects:
        if not isinstance(effect, dict) or not isinstance(effect.get("control"), str):
            problems.append("effect is not an object with a control")
            continue
        controls.add(effect["control"])
        matching = [i for i in declared if TYPE_CONTROL.get(i["type"]) == effect["control"]]
        if not matching:
            continue
        delta, currency = effect.get("delta_minor"), effect.get("currency")
        if not any(_is_int(delta) and abs(delta) == i["expected_amount_minor"] and currency == i["currency"]
                   for i in matching):
            problems.append(f"effect on {effect['control']} ({delta!r} {currency!r}) differs from its issue's "
                            "amount and currency")
    if controls != expected_controls:
        problems.append(f"effects name controls {sorted(controls)}, the declared issues break {sorted(expected_controls)}")
    return problems


def _check_injections(config: Config, clean: dict, published: dict, provenance: dict, truth: list[dict] | None,
                      report: Report) -> None:
    truth_by_id = {i["issue_id"]: i for i in truth or []}
    state = copy.deepcopy(clean)
    books = Books(config, state)
    prev = find_issues(books)
    prev_sigs = {v.signature() for v in identities.check_controls(books)}
    claimed: Counter = Counter()
    for index, injection in enumerate(provenance["injections"]):
        if not isinstance(injection, dict):
            return
        problems = apply_injection(state, injection, index)
        if problems:
            report.fail("injections", "injection_does_not_apply", problems[0].message, change=problems[0].to_dict())
            return
        books = Books(config, state)
        now = find_issues(books)
        violations = identities.check_controls(books)
        issue_ids = injection.get("issue_ids") if _str_list(injection.get("issue_ids")) else []
        claimed.update(issue_ids)
        unknown = [i for i in issue_ids if i not in truth_by_id]
        if unknown and truth is not None:
            report.fail("injections", "injection_unknown_issue", f"injection {index} names issues not in truth: {unknown}")
        declared = [truth_by_id[i] for i in issue_ids if i in truth_by_id]

        new = Counter(issue_key(i) for i in now.issues) - Counter(issue_key(i) for i in prev.issues)
        gone = Counter(issue_key(i) for i in prev.issues) - Counter(issue_key(i) for i in now.issues)
        if truth is not None and new != Counter(issue_key(i) for i in declared):
            report.fail("injections", "injection_issue_mismatch",
                        f"injection {index} creates issues that differ from its issue_ids",
                        created=sorted(_show(k) for k in new.elements()),
                        declared=sorted(_show(issue_key(i)) for i in declared))
        if gone:
            report.fail("injections", "injection_hides_issue", f"injection {index} removes an earlier issue",
                        removed=sorted(_show(k) for k in gone.elements()))
        old_anomalies = {(a["kind"], tuple(a["ids"])) for a in prev.anomalies}
        fresh = [a for a in now.anomalies if (a["kind"], tuple(a["ids"])) not in old_anomalies]
        if fresh:
            report.fail("injections", "injection_creates_anomaly", f"injection {index} creates anomalies",
                        anomalies=fresh)
        broken = {v.control for v in violations if v.signature() not in prev_sigs}
        expected = {TYPE_CONTROL[i["type"]] for i in declared}
        if truth is not None and broken != expected:
            report.fail("injections", "injection_breaks_other_controls",
                        f"injection {index} breaks {sorted(broken)}, its issues belong to {sorted(expected)}")
        for message in _effects_problems(injection, declared, expected):
            report.fail("injections", "injection_effects_mismatch", f"injection {index}: {message}")
        if injection.get("scenario") is None and declared and any(i["type"] != injection.get("kind") for i in declared):
            report.fail("injections", "injection_kind_mismatch",
                        f"injection {index} kind {injection.get('kind')!r} differs from its issue types")
        prev = now
        prev_sigs = {v.signature() for v in violations}
    else:
        if _tables_canon(state) != _tables_canon(published):
            report.fail("injections", "replay_differs", "replaying all injections does not give the published books")
    _check_distractors(provenance, published, truth_by_id, report)
    for issue_id in truth_by_id:
        if claimed[issue_id] != 1:
            report.fail("injections", "issue_not_claimed_once",
                        f"truth issue {issue_id!r} appears in {claimed[issue_id]} injections, not exactly one")


def _check_distractors(provenance: dict, published: dict, truth_by_id: dict, report: Report) -> None:
    """Distractor IDs are existing doc_ids, and no issue outside S1/S2 touches a distractor's documents."""
    distractors = provenance.get("distractors", [])
    if not isinstance(distractors, list):
        return
    doc_ids = {r.get("doc_id") for name in ("invoices", "payments") for r in published[name]}
    plain_issue_docs: dict[str, str] = {}
    for injection in provenance["injections"]:
        if isinstance(injection, dict) and injection.get("scenario") is None and _str_list(injection.get("issue_ids")):
            for issue_id in injection["issue_ids"]:
                for doc in truth_by_id.get(issue_id, {}).get("affected_ids", []):
                    plain_issue_docs[doc] = issue_id
    for position, item in enumerate(distractors):
        if not (isinstance(item, dict) and isinstance(item.get("kind"), str) and _str_list(item.get("ids"))):
            report.fail("injections", "distractors_malformed", f"distractor {position} needs a kind and a list of ids")
            continue
        unknown = sorted(i for i in item["ids"] if i not in doc_ids)
        if unknown:
            report.fail("injections", "distractor_unknown_id", f"distractor {position} names unknown doc_ids {unknown}")
        touched = sorted(i for i in item["ids"] if i in plain_issue_docs)
        if touched:
            report.fail("injections", "distractor_overlaps_issue",
                        f"distractor {position} ({item['kind']}) shares documents {touched} with an issue")


def _show(key: tuple) -> str:
    issue_type, ids, amount, currency = key
    return f"{issue_type} {sorted(ids)} {amount} {currency}"


def _tables_canon(tables: dict) -> dict:
    return {name: sorted(_canon(name, r) for r in tables.get(name, ())) for name in KEY_FIELD}


def _guard(report: Report, condition: str, stage: Callable[[], Any]) -> Any:
    """Run one stage; an unexpected exception becomes a failed condition, not a crash."""
    try:
        return stage()
    except Exception as exc:  # noqa: BLE001 - the report must never crash
        report.fail(condition, "internal_error", f"{type(exc).__name__} while checking: {exc}")
        return None


def verify_episode(episode_dir: str | os.PathLike) -> Report:
    """Check one episode folder against all jury block-2 conditions. Never raises."""
    report = Report(str(episode_dir))
    try:
        return _verify(Path(episode_dir), report)
    except Exception as exc:  # noqa: BLE001
        report.fail("readable", "internal_error", f"{type(exc).__name__}: {exc}")
        return report


def _verify(path: Path, report: Report) -> Report:
    try:
        episode = load_public(path)
    except Exception as exc:  # noqa: BLE001 - loader raises ValueError; anything else is still a failure
        report.fail("readable", "unreadable", str(exc))
        return report
    report.evaluated("readable")
    manifest, policy = episode.manifest, episode.policy
    tables = {name: list(records) for name, records in episode.tables.items()}
    existing = episode.existing_ids()
    report.episode_id = manifest.get("episode_id")

    problems, warnings = check_formats(manifest, tables)
    report.warnings.extend(warnings)
    for message in problems:
        report.fail("formats", "format", message)
    config, config_problems = config_from(manifest, policy)
    for message in config_problems:
        report.fail("formats", "config", message)
    report.evaluated("formats")

    truth = read_truth(path, report.episode_id, report)
    provenance = read_provenance(path, report.episode_id, report)
    report.truth_issue_count = None if truth is None else len(truth)
    if provenance is not None:
        report.injection_count = len(provenance["injections"])
    if config is None:
        report.fail("clean_books", "skipped", "manifest/policy cannot be interpreted")
        return report

    # Published books.
    published = _guard(report, "truth_matches", lambda: find_issues(Books(config, tables)))
    if published is not None:
        report.found_issues, report.anomalies = published.issues, published.anomalies
        for item in published.anomalies:
            report.fail("truth_matches", "anomaly", f"{item['kind']} on {item['ids']}: {item['message']}", anomaly=item)
        if truth is not None:
            _compare_with_truth(published.issues, truth, report)
            report.evaluated("truth_matches")

    # Clean books.
    clean = None
    if provenance is not None:
        result = revert(tables, provenance)
        if not result.ok:
            for problem in result.problems:
                report.fail("clean_books", problem.code, problem.message, change=problem.to_dict())
            report.fail("injections", "revert_failed", "provenance does not revert cleanly")
        else:
            clean = result.tables
            _guard(report, "clean_books", lambda: _check_clean(config, clean, report))
            report.evaluated("clean_books")

    if clean is not None and provenance is not None:
        _guard(report, "injections",
               lambda: _check_injections(config, clean, tables, provenance, truth, report))
        if truth is not None:
            report.evaluated("injections")

    if truth is not None:
        def evidence_stage() -> None:
            for issue in truth:
                for position, evidence_set in enumerate(issue["evidence_sets"]):
                    for message in check_evidence(config, tables, existing, issue, evidence_set):
                        report.fail("evidence", "evidence_insufficient",
                                    f"issue {issue['issue_id']!r} set {position}: {message}",
                                    issue_id=issue["issue_id"], evidence_set=evidence_set)
        _guard(report, "evidence", evidence_stage)
        report.evaluated("evidence")

    if truth is not None and provenance is not None:
        if (len(truth) == 0) != (len(provenance["injections"]) == 0):
            report.fail("clean_episode", "clean_mismatch",
                        f"truth has {len(truth)} issues but provenance has {len(provenance['injections'])} injections")
        if published is not None and not published.issues and truth:
            report.fail("clean_episode", "clean_mismatch", "the books are clean but truth.json lists issues")
        report.evaluated("clean_episode")
    return report


def _check_clean(config: Config, clean: dict, report: Report) -> None:
    books = Books(config, clean)
    for violation in identities.check_controls(books):
        report.fail("clean_books", f"clean_{violation.control}", violation.message, violation=violation.to_dict())
    result = find_issues(books)
    for issue in result.issues:
        report.fail("clean_books", "clean_has_issue", f"reverted books still contain a {issue['type']} on "
                    f"{issue['affected_ids']}", issue=issue)
    for item in result.anomalies:
        report.fail("clean_books", "clean_has_anomaly", f"{item['kind']} on {item['ids']}", anomaly=item)

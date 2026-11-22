"""`python -m cookedbooks.xray`: the read-only, jury-side case file of one injected error."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cookedbooks.verify.transform import STYLES, transform_episode
from cookedbooks.xray.__main__ import main
from cookedbooks.xray.case import EpisodeView, build_all, build_case
from cookedbooks.xray.render import fmt_money, render_case, render_list, show_id

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = sorted(p for split in ("train", "dev") for p in (ROOT / "episodes" / split).iterdir() if p.is_dir())
WITH_ISSUES = [p for p in COMMITTED if json.loads((p / "truth.json").read_text(encoding="utf-8"))["issues"]]
CLEAN = [p for p in COMMITTED if p not in WITH_ISSUES]
MINI = ROOT / "tests" / "fixtures" / "mini"
# Every error type, scenario S1 (dev-10), scenario S2 (dev-09), unallocated payments (dev-04).
TRANSFORM_SOURCES = [ROOT / "episodes" / "dev" / name for name in ("dev-10", "dev-09", "dev-04")]
XRAY_DIR = ROOT / "cookedbooks" / "xray"


def run(capsys, *args: str) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def run_process(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "cookedbooks.xray", *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)


def digest(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


# --- 1, 2: committed episodes ----------------------------------------------------------------


def test_committed_episodes_have_issues_and_clean_ones():
    assert WITH_ISSUES and CLEAN


@pytest.mark.parametrize("folder", WITH_ISSUES, ids=lambda p: p.name)
def test_every_committed_issue_renders_its_ids(folder, capsys):
    view = EpisodeView(folder)
    for case in build_all(view):
        text = render_case(view, case)
        issue = case.issue
        assert issue["issue_id"] in text
        assert issue["type"] in text
        for record_id in issue["affected_ids"] + issue["evidence_sets"][0]:
            assert show_id(record_id) in text
        assert case.changes, "committed episodes carry provenance for every issue"
        assert case.twin is not None or case.twin_note
    for issue in view.issues:
        code, out, err = run(capsys, folder, "--issue", issue["issue_id"], "--color", "never")
        assert code == 0 and err == "" and issue["issue_id"] in out
    code, _, _ = run(capsys, folder, "--all", "--color", "never")
    assert code == 0


@pytest.mark.parametrize("folder", CLEAN, ids=lambda p: p.name)
def test_clean_episode_lists_no_errors(folder, capsys):
    code, out, err = run(capsys, folder, "--color", "never")
    assert code == 0 and err == ""
    assert "ошибок нет" in out
    assert "двойники:" in out


def test_list_mode_one_line_per_issue(capsys):
    folder = ROOT / "episodes" / "dev" / "dev-10"
    code, out, _ = run(capsys, folder, "--color", "never")
    view = EpisodeView(folder)
    assert code == 0
    for issue in view.issues:
        assert sum(issue["issue_id"] in line and issue["type"] in line for line in out.splitlines()) == 1


def test_duplicate_case_names_the_differing_key_field():
    view = EpisodeView(ROOT / "episodes" / "dev" / "dev-03")
    case = build_case(view, 0)
    assert case.twin is not None and case.twin.kind == "recurring_amount"
    text = render_case(view, case)
    assert "service_period" in text and "10 211,30 USD" in text


# --- 3: relabelled and reordered copies -------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("source", TRANSFORM_SOURCES, ids=lambda p: p.name)
def test_transformed_copy_builds_the_same_cases(source, style, tmp_path, capsys):
    target = tmp_path / "copy"
    transform_episode(source, target, permute=True, relabel=True, seed=11, style=style)
    original, copy = EpisodeView(source), EpisodeView(target)
    before, after = build_all(original), build_all(copy)
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert new.counts() == old.counts()
        assert new.issue["type"] == old.issue["type"]
        assert new.issue["expected_amount_minor"] == old.issue["expected_amount_minor"]
        assert new.issue["currency"] == old.issue["currency"]
        assert all(entry.refs for entry in new.entries)
        text = render_case(copy, new)
        for record_id in new.issue["affected_ids"] + new.evidence_set:
            assert show_id(record_id) in text
    if style == "int":
        # The int style gives rows, documents and invoices the same names: collisions are shown, not guessed.
        assert any(entry.ambiguous for case in after for entry in case.entries)
        assert "неоднозначен" in "".join(render_case(copy, case) for case in after)
    code, _, err = run(capsys, target, "--all", "--color", "never")
    assert code == 0 and err == ""


# --- 4: determinism, colour, no paths ----------------------------------------------------------


def test_output_is_deterministic_and_has_no_path(tmp_path, capsys):
    source = ROOT / "episodes" / "dev" / "dev-10"
    folder = tmp_path / "episode"
    shutil.copytree(source, folder)
    first = run(capsys, folder, "--all", "--color", "never")
    second = run(capsys, folder, "--all", "--color", "never")
    assert first == second and first[0] == 0
    for path in (folder, folder.resolve(), tmp_path, source, ROOT):
        assert str(path) not in first[1]
    process_run = run_process(folder, "--all", "--color", "never")
    assert process_run.returncode == 0 and process_run.stdout == first[1]


def test_colour_switch(capsys):
    folder = ROOT / "episodes" / "dev" / "dev-08"
    _, plain, _ = run(capsys, folder, "--all", "--color", "never")
    _, coloured, _ = run(capsys, folder, "--all", "--color", "always")
    assert "\x1b" not in plain
    assert "\x1b" in coloured
    # auto under pytest's captured (non-TTY) stdout is plain
    _, auto, _ = run(capsys, folder, "--all")
    assert auto == plain


# --- 5: folders we did not create --------------------------------------------------------------


def test_without_provenance_sections_two_and_three_degrade(tmp_path, capsys):
    folder = tmp_path / "episode"
    shutil.copytree(ROOT / "episodes" / "dev" / "dev-10", folder)
    (folder / "provenance.json").unlink()
    code, out, err = run(capsys, folder, "--all", "--color", "never")
    assert code == 0 and err == ""
    issues = EpisodeView(folder).issues
    assert out.count("история изменений недоступна") >= 2 * len(issues)
    code, out, _ = run(capsys, folder, "--color", "never")
    assert code == 0 and "история изменений недоступна" in out


def test_malformed_truth_issue_does_not_crash(tmp_path, capsys):
    folder = tmp_path / "episode"
    shutil.copytree(ROOT / "episodes" / "dev" / "dev-03", folder)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    good = truth["issues"][0]
    truth["issues"] = [
        dict(good, issue_id=7),                                   # integer issue_id
        {"issue_id": "broken", "type": 5, "affected_ids": "x"},  # malformed fields
        "not an object",
        good,
    ]
    (folder / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    code, out, err = run(capsys, folder, "--all", "--color", "never")
    assert code == 0 and err == "" and "Traceback" not in out
    assert "Дело 7 " in out and "Дело broken " in out and "Дело #2 " in out
    code, out, _ = run(capsys, folder, "--issue", "7", "--color", "never")
    assert code == 0 and "duplicate_invoice" in out
    process_run = run_process(folder, "--issue", "broken", "--color", "never")
    assert process_run.returncode == 0 and "Traceback" not in process_run.stderr


def test_small_hand_written_episode(capsys):
    code, out, err = run(capsys, MINI, "--all", "--color", "never")
    assert code == 0 and err == ""
    assert "Дело" in out


def test_ambiguous_id_shows_every_match(tmp_path, capsys):
    """An affected ID that is also a row_id: both records are shown, with a note."""
    folder = tmp_path / "episode"
    shutil.copytree(ROOT / "episodes" / "dev" / "dev-03", folder)
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    doc_id = truth["issues"][0]["affected_ids"][0]
    lines = (folder / "ledger.csv").read_text(encoding="utf-8").split("\n")
    row_id = lines[1].split(",")[0]
    (folder / "ledger.csv").write_text("\n".join([lines[0], lines[1].replace(row_id, doc_id, 1)] + lines[2:]),
                                       encoding="utf-8")
    view = EpisodeView(folder)
    case = build_case(view, 0)
    entry = next(e for e in case.entries if e.id == doc_id)
    assert [ref[0] for ref in entry.refs] == ["ledger", "invoices"]
    code, out, _ = run(capsys, folder, "--issue", truth["issues"][0]["issue_id"], "--color", "never")
    assert code == 0 and "неоднозначен" in out


# --- 6: read-only ------------------------------------------------------------------------------


@pytest.mark.parametrize("folder", [ROOT / "episodes" / "dev" / "dev-10", MINI], ids=lambda p: p.name)
def test_folder_is_unchanged(folder, capsys):
    before = digest(folder)
    assert run(capsys, folder, "--all", "--color", "always")[0] == 0
    assert run(capsys, folder)[0] == 0
    assert digest(folder) == before


# --- 7: exit 2 ---------------------------------------------------------------------------------


def _short_error(process_run: subprocess.CompletedProcess, *paths: Path) -> None:
    assert process_run.returncode == 2
    assert process_run.stdout == ""
    lines = process_run.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith("xray: ")
    assert "Traceback" not in process_run.stderr
    for path in paths:
        assert str(path) not in process_run.stderr


def test_unknown_issue_exits_2():
    folder = ROOT / "episodes" / "dev" / "dev-03"
    _short_error(run_process(folder, "--issue", "ISS-9999"), folder, ROOT)
    _short_error(run_process(folder, "--issue", "iss-0001"), folder, ROOT)  # IDs are not case folded


def test_missing_folder_and_missing_truth_exit_2(tmp_path):
    _short_error(run_process(tmp_path / "nope"), tmp_path)
    folder = tmp_path / "episode"
    shutil.copytree(ROOT / "episodes" / "dev" / "dev-03", folder)
    (folder / "truth.json").unlink()
    _short_error(run_process(folder, "--all"), tmp_path)


# --- 8: isolation ------------------------------------------------------------------------------

ALLOWED = {"cookedbooks.episode.loader", "cookedbooks.grader.truth"}


def _imports(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: relative import"
            module = node.module or ""
            names.append(module)
            names += [f"{module}.{alias.name}" for alias in node.names if module == "cookedbooks"]
    return names


@pytest.mark.parametrize("path", sorted(XRAY_DIR.glob("*.py")), ids=lambda p: p.name)
def test_xray_imports_only_stdlib_loader_and_truth(path):
    for name in _imports(path):
        top = name.split(".")[0]
        if top == "cookedbooks":
            assert name in ALLOWED or name == "cookedbooks.xray" or name.startswith("cookedbooks.xray."), name
        else:
            assert top in sys.stdlib_module_names or top == "__future__", name


def test_nothing_on_the_agent_or_grader_side_imports_xray():
    paths = [ROOT / "judge_adapter.py"]
    for package in ("env", "policies", "grader", "episode"):
        paths += sorted((ROOT / "cookedbooks" / package).rglob("*.py"))
    for path in paths:
        for name in _imports(path):
            assert not (name == "cookedbooks.xray" or name.startswith("cookedbooks.xray.")), (path.name, name)


@pytest.mark.parametrize("path", sorted(XRAY_DIR.glob("*.py")), ids=lambda p: p.name)
def test_xray_has_no_float_and_no_hash(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("float", "hash", "eval", "exec"), node.func.id
        if isinstance(node, ast.Constant):
            assert not isinstance(node.value, float), node.value


# --- 9: money ----------------------------------------------------------------------------------

EXPONENTS = {"JPY": 0, "USD": 2, "KWD": 3}


@pytest.mark.parametrize("amount, currency, expected", [
    (1021130, "USD", "10 211,30 USD"),
    (5, "USD", "0,05 USD"),
    (0, "USD", "0,00 USD"),
    (100, "USD", "1,00 USD"),
    (-1021130, "USD", "-10 211,30 USD"),
    (-5, "USD", "-0,05 USD"),
    (1234567, "JPY", "1 234 567 JPY"),
    (999, "JPY", "999 JPY"),
    (-1000, "JPY", "-1 000 JPY"),
    (4147225, "KWD", "4 147,225 KWD"),
    (1, "KWD", "0,001 KWD"),
    (-681811, "KWD", "-681,811 KWD"),
    (12345, "XYZ", "12345 мин. ед. XYZ"),
    (-12345, "XYZ", "-12345 мин. ед. XYZ"),
    (10 ** 20 + 1, "USD", "1 000 000 000 000 000 000,01 USD"),
])
def test_money_format(amount, currency, expected):
    assert fmt_money(amount, currency, EXPONENTS) == expected


def test_money_rejects_bool_and_bad_exponent():
    assert fmt_money(True, "USD", EXPONENTS) == "true USD"
    assert fmt_money(100, "USD", {"USD": True}) == "100 мин. ед. USD"
    assert fmt_money(100, "USD", {"USD": -1}) == "100 мин. ед. USD"


def test_render_list_of_mini():
    view = EpisodeView(MINI)
    text = render_list(view)
    assert text.count("\n") >= 2

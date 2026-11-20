"""Verifier CLI exit codes and import isolation from cookedbooks.world."""

import ast
import json
import subprocess
import sys
from pathlib import Path

from cookedbooks.verifier.__main__ import main
from tests.helpers.verifier_cases import lookalike_book

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "cookedbooks" / "verifier"


def _good(tmp_path):
    book, _ = lookalike_book()
    return book.write(tmp_path / "good")


def _bad(tmp_path):
    book, _ = lookalike_book()
    path = book.write(tmp_path / "bad")
    (path / "truth.json").write_text('{"schema_version": "1.0", "episode_id": "ep-v", "issues": [{"issue_id": "1", '
                                     '"type": "fx_error", "affected_ids": ["10"], "expected_amount_minor": 1, '
                                     '"currency": "KZT", "evidence_sets": [["10"]]}]}', encoding="utf-8")
    return path


def test_main_exit_codes(tmp_path, capsys):
    good, bad = _good(tmp_path), _bad(tmp_path)
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert main([str(good), str(bad)]) == 1
    assert main([str(tmp_path / "missing")]) == 1
    out = capsys.readouterr().out
    assert out.startswith("OK") and "FAIL" in out


def test_main_json(tmp_path, capsys):
    good, bad = _good(tmp_path), _bad(tmp_path)
    assert main(["--json", str(good), str(bad)]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["ok"] for line in lines] == [True, False]
    assert set(lines[0]["conditions"]) >= {"formats", "clean_books", "truth_matches", "injections", "evidence"}
    assert lines[1]["problems"]


def test_module_entry_point(tmp_path):
    good = _good(tmp_path)
    run = lambda *args: subprocess.run([sys.executable, "-m", "cookedbooks.verifier", *args], cwd=ROOT,
                                       capture_output=True, text=True)
    assert run(str(good)).returncode == 0
    assert run(str(_bad(tmp_path))).returncode == 1
    assert run().returncode == 2


def _imported_names(node) -> list[str]:
    """Absolute-ish names an import node can bind; relative imports resolve inside cookedbooks."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level == 1:
            module = "cookedbooks.verifier" + (f".{module}" if module else "")
        elif node.level >= 2:
            module = "cookedbooks" + (f".{module}" if module else "")
        return [module] + [f"{module}.{alias.name}" for alias in node.names]
    return []


def test_verifier_never_imports_world_ast():
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 8
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for name in _imported_names(node):
                assert name != "cookedbooks.world" and not name.startswith("cookedbooks.world."), \
                    f"{path.name} imports {name}"
            if isinstance(node, ast.Call):
                called = getattr(node.func, "id", getattr(node.func, "attr", ""))
                assert called not in ("__import__", "import_module"), f"{path.name} imports dynamically"
        assert "world" not in path.read_text(encoding="utf-8").replace("`cookedbooks.world`", "")


def test_verifier_never_imports_world_at_runtime(tmp_path):
    good = _good(tmp_path)
    code = (
        "import sys\n"
        "from cookedbooks.verifier import verify_episode\n"
        f"assert verify_episode({str(good)!r}).ok\n"
        "bad = sorted(m for m in sys.modules if m == 'cookedbooks.world' or m.startswith('cookedbooks.world.'))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

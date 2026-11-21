"""The verification run behind `make verify`.

`python -m cookedbooks.verify` runs the whole pytest suite in a subprocess,
reads the JUnit XML it writes, and maps test names to the jury's check IDs:
`test_C07_...` belongs to C07, `test_R12_...` to test 12 of block 3,
`test_M03_...` to example M03. Several tests may share one ID; the ID counts
as passed only when it has at least one test and all of its tests passed.

The summary is printed and written to `reports/verify_report.md`. Nothing in
the report depends on the clock except the duration line.

The jury blocks are data: `BLOCK_SPECS` lists each block's rows as
(label, ID prefix, count). A block whose tests do not exist yet has only a
label (a string) and shows "not implemented yet"; switching it on is a
one-line change.

The pytest run also loads this module as a plugin (`-p cookedbooks.verify`):
at the end of the session `pytest_sessionfinish` dumps
`cookedbooks.pipeline.STATS` (generator attempts the verifier rejected) to the
file named by $COOKEDBOOKS_STATS_FILE, and the report prints it.

Standard library only, and kept importable on older Python 3 releases so
that `make wheels-check` runs with whatever `python3` the host has.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "reports" / "verify_report.md"
REPORT_ENV = "COOKEDBOOKS_REPORT"
STATS_ENV = "COOKEDBOOKS_STATS_FILE"

# `test_C07_get_record_r1`, `test_R12[case]`, `test_M03` -> ("C", "07").
_TEST_ID = re.compile(r"^test_([A-Z])(\d{2})(?=_|\[|$)")

PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"


# --------------------------------------------------------------------------
# Test results


@dataclass
class Results:
    """Outcome of one pytest run."""

    exit_code: int
    # One entry per test case: (test id like "C07" or None, outcome).
    cases: List[Tuple[Optional[str], str]] = field(default_factory=list)
    # `cookedbooks.pipeline.STATS.to_dict()` of the run, when it was recorded.
    generation: Optional[Dict[str, Any]] = None

    def count(self, outcome: str) -> int:
        return sum(1 for _, o in self.cases if o == outcome)

    def by_id(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for test_id, outcome in self.cases:
            if test_id is not None:
                out.setdefault(test_id, []).append(outcome)
        return out

    @property
    def green(self) -> bool:
        return self.exit_code == 0 and self.count(FAILED) == 0


def test_id_of(name: str) -> Optional[str]:
    """Jury ID carried by a test function name, or None."""
    m = _TEST_ID.match(name)
    return m.group(1) + m.group(2) if m else None


def parse_junit(xml_text: str, exit_code: int) -> Results:
    """Read pytest's JUnit XML into `Results`.

    A case with a <failure> or <error> child failed (an error in setup or
    teardown counts), one with <skipped> was skipped, the rest passed.
    """
    root = ET.fromstring(xml_text)
    results = Results(exit_code=exit_code)
    for case in root.iter("testcase"):
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            outcome = FAILED
        elif "skipped" in tags:
            outcome = SKIPPED
        else:
            outcome = PASSED
        results.cases.append((test_id_of(case.get("name", "")), outcome))
    return results


def run_pytest(pytest_args: Sequence[str]) -> Results:
    """Run pytest in a subprocess from the repo root and collect results."""
    with tempfile.TemporaryDirectory(prefix="cookedbooks-verify-") as tmp:
        junit = Path(tmp) / "junit.xml"
        stats = Path(tmp) / "generation.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "cookedbooks.verify",
            "--junitxml",
            str(junit),
            *pytest_args,
        ]
        env = dict(os.environ)
        env[STATS_ENV] = str(stats)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if not junit.is_file():
            return Results(exit_code=proc.returncode or 1)
        results = parse_junit(junit.read_text(encoding="utf-8"), proc.returncode)
        results.generation = read_generation(stats)
        return results


def read_generation(path: Path) -> Optional[Dict[str, Any]]:
    """The generation statistics the plugin wrote, or None if there are none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """Pytest hook (this module is loaded with `-p cookedbooks.verify`): dump pipeline.STATS.

    Writes only when $COOKEDBOOKS_STATS_FILE is set and the pipeline was used.
    """
    path = os.environ.get(STATS_ENV)
    pipeline = sys.modules.get("cookedbooks.pipeline")
    if not path or pipeline is None:
        return
    Path(path).write_text(json.dumps(pipeline.STATS.to_dict(), sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------
# Jury blocks


@dataclass
class Row:
    label: str
    value: str
    # IDs that are failing or have no test, shown so a red row says why.
    problems: List[str] = field(default_factory=list)


@dataclass
class Block:
    number: int
    title: str
    rows: List[Row]


def id_row(results: Results, label: str, prefix: str, count: int) -> Row:
    """Row "passed/total" for IDs prefix01..prefixNN."""
    by_id = results.by_id()
    passed = 0
    problems: List[str] = []
    for n in range(1, count + 1):
        test_id = "%s%02d" % (prefix, n)
        outcomes = by_id.get(test_id, [])
        if outcomes and all(o == PASSED for o in outcomes):
            passed += 1
        elif not outcomes:
            problems.append(test_id + " (no test)")
        elif FAILED in outcomes:
            problems.append(test_id + " (failed)")
        else:
            problems.append(test_id + " (skipped)")
    return Row(label, "%d/%d" % (passed, count), problems)


# (number, title, rows): each row is (label, test ID prefix, count of IDs).
# rows = a label string: the block's tests are not written yet ("not implemented yet").
BLOCK_SPECS: List[Tuple[int, str, Any]] = [
    (1, "Contract and isolation", [("C01-C20", "C", 20)]),
    (2, "Data and ground truth", [("D01-D20", "D", 20)]),
    (3, "Matcher and reward", [("tests 1-20 (R01-R20)", "R", 20), ("examples M01-M12", "M", 12)]),
    (4, "Reproducibility", "P01-P20"),  # -> [("P01-P20", "P", 20)] once tests/jury/test_block4_pairs.py exists
]


def make_block(number: int, title: str, rows: Any) -> Callable[[Results], Block]:
    """A function `Results -> Block` for one entry of BLOCK_SPECS."""

    def block(results: Results) -> Block:
        if isinstance(rows, str):
            return Block(number, title, [Row(rows, "not implemented yet")])
        return Block(number, title, [id_row(results, label, prefix, count) for label, prefix, count in rows])

    block.__name__ = "block_%d" % number
    return block


BLOCKS: List[Callable[[Results], Block]] = [make_block(*spec) for spec in BLOCK_SPECS]


# --------------------------------------------------------------------------
# Report


def render(results: Results, duration_s: Optional[float] = None) -> str:
    """The summary as Markdown. Same text goes to the console and the file."""
    lines = [
        "# CookedBooks verification",
        "",
        "Python %s on %s." % (platform.python_version(), platform.machine() or "unknown"),
        "",
        "pytest: %d passed, %d failed, %d skipped, exit code %d."
        % (results.count(PASSED), results.count(FAILED), results.count(SKIPPED), results.exit_code),
        "",
        "| Block | Jury checks | Result |",
        "|---|---|---|",
    ]
    problems: List[str] = []
    for make_block in BLOCKS:
        block = make_block(results)
        for i, row in enumerate(block.rows):
            name = "%d. %s" % (block.number, block.title) if i == 0 else ""
            lines.append("| %s | %s | %s |" % (name, row.label, row.value))
            if row.problems:
                problems.append("- Block %d, %s: %s" % (block.number, row.label, ", ".join(row.problems)))
    lines.append("")
    if results.generation is not None:
        lines += [generation_line(results.generation), ""]
    if problems:
        lines += ["Not passing:", ""] + problems + [""]
    lines.append("Verdict: %s" % ("PASS" if results.green else "FAIL"))
    if duration_s is not None:
        lines += ["", "Duration: %.1f s" % duration_s]
    return "\n".join(lines) + "\n"


def generation_line(stats: Dict[str, Any]) -> str:
    """'Generator attempts rejected by the verifier: k of n ...' from pipeline.STATS.to_dict()."""
    line = "Generator attempts rejected by the verifier: %d of %d (%d episodes accepted" % (
        stats.get("rejected", 0), stats.get("attempts", 0), stats.get("episodes", 0))
    if stats.get("failed_episodes"):
        line += ", %d ran out of attempts" % stats["failed_episodes"]
    line += ")."
    codes = stats.get("codes") or {}
    if codes:
        line += " Codes: " + ", ".join("%s x%d" % (code, codes[code]) for code in sorted(codes)) + "."
    return line


def write_report(text: str, path: Path) -> Path:
    """Write the report; fall back to the temp dir if `path` is read-only."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "reports" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(text, encoding="utf-8")
        return fallback


# --------------------------------------------------------------------------
# Vendored wheels


# name-version(-build)?-python-abi-platform.whl
_WHEEL = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)


def normalize(name: str) -> str:
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path: Path) -> Dict[str, str]:
    pins: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if not sep:
            raise ValueError("requirements.lock: not an exact pin: %r" % raw)
        pins[normalize(name.strip())] = version.strip()
    return pins


def check_wheels(lock: Path, wheel_dir: Path) -> List[str]:
    """Problems with the vendored wheel set; empty list means it matches.

    The set must hold exactly one universal (`none-any`) wheel per pin in
    the lock file, with the pinned version, and nothing else.
    """
    pins = read_lock(lock)
    problems: List[str] = []
    found: Dict[str, str] = {}
    files = sorted(p.name for p in wheel_dir.iterdir()) if wheel_dir.is_dir() else []
    for fname in files:
        if fname.startswith("."):
            continue
        m = _WHEEL.match(fname)
        if not m:
            problems.append("not a wheel: %s" % fname)
            continue
        if m.group("abi") != "none" or m.group("plat") != "any":
            problems.append("not a universal wheel: %s" % fname)
        name = normalize(m.group("name"))
        if name in found:
            problems.append("two wheels for %s" % name)
        found[name] = m.group("version")
    for name in sorted(pins):
        if name not in found:
            problems.append("missing wheel: %s==%s" % (name, pins[name]))
        elif found[name] != pins[name]:
            problems.append("wrong version: %s %s, lock says %s" % (name, found[name], pins[name]))
    for name in sorted(set(found) - set(pins)):
        problems.append("wheel not in requirements.lock: %s" % name)
    return problems


# --------------------------------------------------------------------------
# Entry point


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m cookedbooks.verify",
        description="Run the test suite and print the jury-block summary.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(os.environ.get(REPORT_ENV) or DEFAULT_REPORT),
        help="where to write the Markdown report (default: reports/verify_report.md, or $%s)" % REPORT_ENV,
    )
    parser.add_argument(
        "--check-wheels",
        action="store_true",
        help="only check that vendor/wheels matches requirements.lock",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra pytest arguments after --, for example test paths",
    )
    args = parser.parse_args(argv)

    if args.check_wheels:
        problems = check_wheels(REPO_ROOT / "requirements.lock", REPO_ROOT / "vendor" / "wheels")
        for p in problems:
            print(p)
        print("wheels: OK" if not problems else "wheels: %d problem(s)" % len(problems))
        return 0 if not problems else 1

    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]

    start = time.monotonic()
    results = run_pytest(pytest_args)
    text = render(results, time.monotonic() - start)
    print()
    print(text, end="")
    where = write_report(text, args.report)
    print("Report: %s" % where)
    return 0 if results.green else 1

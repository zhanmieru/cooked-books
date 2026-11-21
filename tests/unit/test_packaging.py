"""Packaging: vendored wheels, the Dockerfile, the Makefile and the verify report."""

import re
from pathlib import Path

from cookedbooks.verify import (
    BLOCKS,
    PASSED,
    Results,
    check_wheels,
    parse_junit,
    render,
    test_id_of as id_of,
)

ROOT = Path(__file__).resolve().parents[2]


def _junit(*cases: str) -> str:
    return "<testsuites><testsuite>%s</testsuite></testsuites>" % "".join(cases)


def _case(name: str, child: str = "") -> str:
    return '<testcase classname="tests.x" name="%s">%s</testcase>' % (name, child)


# --- vendored wheels -------------------------------------------------------


def test_vendored_wheels_match_lock_exactly():
    assert check_wheels(ROOT / "requirements.lock", ROOT / "vendor" / "wheels") == []


def test_wheel_check_flags_platform_extra_and_missing(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("pytest==9.2.0\nSorted_Containers==2.4.0\nattrs==1.0\n")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in [
        "pytest-9.2.0-py3-none-any.whl",
        "sorted_containers-2.4.0-py2.py3-none-any.whl",
        "attrs-1.1-py3-none-any.whl",
        "numpy-2.0.0-cp311-cp311-manylinux_2_17_x86_64.whl",
    ]:
        (wheels / name).write_bytes(b"")
    problems = check_wheels(lock, wheels)
    assert "wrong version: attrs 1.1, lock says 1.0" in problems
    assert any(p.startswith("not a universal wheel: numpy") for p in problems)
    assert "wheel not in requirements.lock: numpy" in problems
    assert len(problems) == 3

    (wheels / "pytest-9.2.0-py3-none-any.whl").unlink()
    assert "missing wheel: pytest==9.2.0" in check_wheels(lock, wheels)


# --- Dockerfile, .dockerignore, Makefile ------------------------------------


def test_dockerfile_pins_base_and_installs_offline():
    text = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^FROM python:3\.11\.\d+-slim\S*@sha256:[0-9a-f]{64}\s*$", text, re.M)
    assert "--no-index" in text and "--find-links ./vendor/wheels" in text
    assert re.search(r"^USER (?!0\b|root\b)\S+", text, re.M)
    # Dependency layer before the source copy.
    assert text.index("COPY vendor/wheels/") < text.index("COPY . .")
    assert 'CMD ["python", "-m", "cookedbooks.verify"]' in text


def test_dockerignore_keeps_local_state_out():
    lines = set((ROOT / ".dockerignore").read_text().split())
    assert {".git", ".venv", ".local", ".hypothesis", ".pytest_cache", "reports"} <= lines


def test_make_verify_runs_offline_with_jury_limits():
    text = (ROOT / "Makefile").read_text()
    for flag in ("--network none", "--cpus $(CPUS)", "--memory $(MEMORY)", "--read-only"):
        assert flag in text
    assert re.search(r"^CPUS\s+\?= 2$", text, re.M)
    assert re.search(r"^MEMORY\s+\?= 4g$", text, re.M)
    assert re.search(r"^DOCKER\s+\?= docker$", text, re.M)
    assert re.search(r"build --network none", text)
    for target in ("help", "verify", "test", "image", "shell", "clean", "wheels-check"):
        assert re.search(r"^%s:.*## " % re.escape(target), text, re.M), target


# --- verify: test names to jury IDs ----------------------------------------


def test_test_id_of():
    assert id_of("test_C07_get_record_r1") == "C07"
    assert id_of("test_R12[case-3]") == "R12"
    assert id_of("test_M03") == "M03"
    assert id_of("test_C7_short") is None
    assert id_of("test_C070_three_digits") is None
    assert id_of("test_loader_accepts_rows") is None


def test_parse_junit_outcomes():
    xml = _junit(
        _case("test_C01_a"),
        _case("test_C02_a", "<failure message='x'/>"),
        _case("test_C03_a", "<error message='teardown'/>"),
        _case("test_C04_a", "<skipped/>"),
        _case("test_plain"),
    )
    r = parse_junit(xml, exit_code=1)
    assert r.cases == [
        ("C01", "passed"),
        ("C02", "failed"),
        ("C03", "failed"),
        ("C04", "skipped"),
        (None, "passed"),
    ]
    assert not r.green


def test_id_counts_only_when_all_its_tests_pass():
    cases = [("C%02d" % n, PASSED) for n in range(1, 21)]
    cases += [("C15", "failed"), ("C20", "skipped")]
    cases += [("R%02d" % n, PASSED) for n in range(1, 20)]  # R20 has no test
    cases += [("M%02d" % n, PASSED) for n in range(1, 13)]
    cases += [("D%02d" % n, PASSED) for n in range(1, 20)]  # D20 has no test
    cases += [("D07", PASSED), ("D07", "failed")]  # one of D07's two tests fails
    r = Results(exit_code=1, cases=cases)
    block1, block2, block3, block4 = [b(r) for b in BLOCKS]
    assert block1.rows[0].value == "18/20"
    assert block1.rows[0].problems == ["C15 (failed)", "C20 (skipped)"]
    assert block2.rows[0].label == "D01-D20" and block2.rows[0].value == "18/20"
    assert block2.rows[0].problems == ["D07 (failed)", "D20 (no test)"]
    assert block3.rows[0].value == "19/20"
    assert block3.rows[0].problems == ["R20 (no test)"]
    assert block3.rows[1].value == "12/12"
    assert block4.rows[0].value == "not implemented yet"


def test_report_prints_generation_stats_when_recorded():
    r = Results(exit_code=0, cases=[("C01", PASSED)])
    assert "Generator attempts" not in render(r)
    r.generation = {"episodes": 40, "failed_episodes": 0, "attempts": 42, "rejected": 2,
                    "codes": {"truth_mismatch": 2, "anomaly": 1}}
    assert ("Generator attempts rejected by the verifier: 2 of 42 (40 episodes accepted). "
            "Codes: anomaly x1, truth_mismatch x2.") in render(r)


def test_stats_plugin_writes_pipeline_stats(tmp_path, monkeypatch):
    import cookedbooks.verify as verify
    from cookedbooks import pipeline

    stats = pipeline.GenerationStats()
    stats.record_attempt(("x",))
    stats.record_attempt(None)
    stats.episodes = 1
    monkeypatch.setattr(pipeline, "STATS", stats)
    target = tmp_path / "stats.json"
    monkeypatch.setenv(verify.STATS_ENV, str(target))
    verify.pytest_sessionfinish(None, 0)
    assert verify.read_generation(target) == stats.to_dict()
    assert verify.read_generation(tmp_path / "missing.json") is None


def test_report_is_stable_without_duration():
    r = Results(exit_code=0, cases=[("C01", PASSED)])
    assert render(r) == render(r)
    assert "Duration" not in render(r)
    text = render(r, duration_s=1.0)
    assert "Verdict: PASS" in text and "Duration: 1.0 s" in text
    assert "Verdict: FAIL" in render(Results(exit_code=1, cases=[("C01", PASSED)]))

"""Jury block 1, contract and isolation: C01-C20 (docs/jury-checks.md, section 2).

Every test goes through judge_adapter.make_env(), as the jury does.
Fixtures: C = clean_episode, E = mini_episode, L = large_episode, U = `.u`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.env import core
from tests.helpers.factory import LEDGER_COLUMNS, _mini_public

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = ("ledger", "invoices", "payments", "fx")


def act(name, **args) -> dict:
    return {"name": name, "args": args}


def start(fixture, mode="closed", team_config=None):
    env = judge_adapter.make_env()
    config = {"mode": mode, "episode_dir": str(fixture.path), "team_config": team_config or {}}
    observation, info = env.reset(0, config)
    return env, observation, info


def oracle(fixture) -> dict:
    return {"episode_id": fixture.episode_id, "findings": [
        {"type": i["type"], "affected_ids": list(i["affected_ids"]),
         "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
         "evidence_ids": list(i["evidence_sets"][0]), "explanation": "Same order billed twice."}
        for i in fixture.issues]}


def json_roundtrip(value) -> bool:
    return json.loads(json.dumps(value, allow_nan=False)) == value


def walk_table(env, table, limit=50):
    """All pages of a table; returns (pages, step results)."""
    cursor, pages, results = None, [], []
    while True:
        result = env.step(act("list_records", table=table, cursor=cursor, limit=limit))
        results.append(result)
        assert result[0]["error"] is None
        pages.append(result[0]["result"])
        cursor = result[0]["result"]["next_cursor"]
        if cursor is None:
            return pages, results


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- C01-C04 ----------------------------------------------------------------------


def test_C01_reset_returns_json_pair(clean_episode):
    env = judge_adapter.make_env()
    pair = env.reset(0, {"mode": "closed", "episode_dir": str(clean_episode.path), "team_config": {}})
    assert isinstance(pair, tuple) and len(pair) == 2
    observation, info = pair
    assert info == {}
    assert json_roundtrip(observation) and json_roundtrip(info)


def test_C02_first_observation(clean_episode):
    _, observation, _ = start(clean_episode)
    assert set(observation) == {"task", "manifest", "policy", "remaining_steps"}
    assert isinstance(observation["task"], str) and observation["task"]
    assert isinstance(observation["manifest"], dict) and isinstance(observation["policy"], dict)
    assert observation["remaining_steps"] == 100
    text = json.dumps(observation).lower()
    for banned in ("seed", "truth", "provenance", "issue_id", "evidence_sets", "expected_amount"):
        assert banned not in text


def test_C03_manifest_matches_data(clean_episode):
    env, observation, _ = start(clean_episode)
    manifest = observation["manifest"]
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
                "timezone", "record_counts", "document_count", "rounding"):
        assert key in manifest, key
    assert manifest["episode_id"] == clean_episode.episode_id
    counts = {}
    for table in TABLES:
        pages, _ = walk_table(env, table)
        counts[table] = sum(len(p["records"]) for p in pages)
    assert manifest["record_counts"] == counts
    assert manifest["document_count"] == counts["invoices"] + counts["payments"]


def test_C04_step_returns_five_values(clean_episode):
    env, _, _ = start(clean_episode)
    result = env.step(act("list_records", table="ledger", cursor=None, limit=50))
    assert isinstance(result, tuple) and len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert set(observation) == {"result", "error", "remaining_steps"}
    assert type(terminated) is bool and type(truncated) is bool
    assert isinstance(reward, (int, float)) and not isinstance(reward, bool)
    assert isinstance(info, dict)
    assert set(observation["result"]) == {"records", "next_cursor"}
    assert json_roundtrip(list(result))


# --- C05-C10 ----------------------------------------------------------------------


def test_C05_pagination_reaches_every_row(large_episode):
    env, _, _ = start(large_episode)
    pages, _ = walk_table(env, "ledger", limit=50)
    assert len(pages) >= 3
    assert all(len(p["records"]) <= 50 for p in pages)
    ids = [r["row_id"] for p in pages for r in p["records"]]
    assert ids == list(large_episode.row_ids)   # file order, no gaps, no repeats
    assert len(ids) == len(set(ids)) >= 101
    for table, expected in (("invoices", large_episode.invoice_doc_ids),
                            ("payments", large_episode.payment_doc_ids)):
        pages, _ = walk_table(env, table)
        assert [r["doc_id"] for p in pages for r in p["records"]] == list(expected)
    pages, _ = walk_table(env, "fx")
    assert [r["id"] for p in pages for r in p["records"]] == list(large_episode.fx_ids)


def test_C06_same_cursor_same_page(large_episode):
    env, _, _ = start(large_episode)
    first = env.step(act("list_records", table="ledger", cursor=None, limit=50))[0]["result"]
    cursor = first["next_cursor"]
    a = env.step(act("list_records", table="ledger", cursor=cursor, limit=50))[0]["result"]
    b = env.step(act("list_records", table="ledger", cursor=cursor, limit=50))[0]["result"]
    assert a == b and a["records"]
    again = env.step(act("list_records", table="ledger", cursor=None, limit=50))[0]["result"]
    assert again == first


def test_C07_get_record_r1(clean_episode):
    env, _, _ = start(clean_episode)
    observation = env.step(act("get_record", table="ledger", id=clean_episode.r1))[0]
    assert observation["error"] is None
    record = observation["result"]
    assert record == _mini_public()["ledger"][0]
    assert set(record) == set(LEDGER_COLUMNS)


_opened: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if _opened is not None and event in {"open", "os.listdir", "os.scandir", "subprocess.Popen",
                                         "os.system", "socket.connect", "sqlite3.connect"}:
        first = args[0] if args else ""
        _opened.append(f"{event}:{os.fsdecode(first) if isinstance(first, (str, bytes, os.PathLike)) else first}")


sys.addaudithook(_audit)


@pytest.fixture
def audit():
    global _opened
    _opened = []
    yield _opened
    _opened = None


def test_C08_get_document_d1(clean_episode, audit):
    env, _, _ = start(clean_episode)
    audit.clear()
    observation = env.step(act("get_document", doc_id=clean_episode.d1))[0]
    assert audit == []   # no file, process or network access during the step
    assert observation["error"] is None
    assert observation["result"] == _mini_public()["invoices"][0]


def test_C09_calculator_decimal(clean_episode):
    env, _, _ = start(clean_episode)
    observation = env.step(act("calculator", expression="0.1+0.2"))[0]
    assert observation == {"result": "0.3", "error": None, "remaining_steps": 99}


@pytest.mark.parametrize("expression", [
    "open('/etc/passwd').read()",
    "__import__('os').system('touch {marker}')",
    "().__class__.__base__.__subclasses__()",
    "exec('1')", "eval('1')", "import os", "2**10", "x", "'a'*3",
])
def test_C10_calculator_rejects_non_arithmetic(clean_episode, tmp_path, audit, expression):
    marker = tmp_path / "executed"
    env, _, _ = start(clean_episode)
    audit.clear()
    result = env.step(act("calculator", expression=expression.format(marker=marker)))
    assert result[0] == {"result": None, "error": {"code": "calc_error", "message": result[0]["error"]["message"]},
                         "remaining_steps": 99}
    assert result[1:] == (0.0, False, False, {})
    assert audit == [] and not marker.exists()


# --- C11-C14 ----------------------------------------------------------------------


def test_C11_get_record_unknown(clean_episode):
    env, _, _ = start(clean_episode)
    for table in TABLES:
        observation, _, terminated, truncated, _ = env.step(act("get_record", table=table, id=clean_episode.u))
        assert observation["result"] is None
        assert observation["error"]["code"] == "unknown_id"
        assert set(observation["error"]) == {"code", "message"}
        assert clean_episode.u not in observation["error"]["message"]
        assert (terminated, truncated) == (False, False)
    assert observation["remaining_steps"] == 96
    after = env.step(act("get_record", table="ledger", id=clean_episode.r1))[0]
    assert after["result"]["row_id"] == clean_episode.r1


def test_C12_get_document_unknown(clean_episode):
    env, _, _ = start(clean_episode)
    observation = env.step(act("get_document", doc_id=clean_episode.u))[0]
    assert observation["result"] is None
    assert observation["error"]["code"] == "unknown_id"
    assert observation["remaining_steps"] == 99
    text = json.dumps(observation)
    for doc_id in clean_episode.doc_ids:
        assert json.dumps(doc_id) not in text   # no other document revealed
    assert env.step(act("get_document", doc_id=clean_episode.d2))[0]["result"]["doc_id"] == clean_episode.d2


def test_C13_unknown_action(clean_episode):
    env, _, _ = start(clean_episode)
    observation, reward, terminated, truncated, info = env.step(act("drop_table", table="ledger"))
    assert observation["result"] is None and observation["error"]["code"] == "unknown_action"
    assert observation["remaining_steps"] == 99
    assert (reward, terminated, truncated, info) == (0, False, False, {})


def test_C14_good_bad_good(clean_episode):
    env, _, _ = start(clean_episode)
    first = env.step(act("get_record", table="ledger", id=clean_episode.r1))[0]
    bad = env.step(act("get_record", table="ledger", id=clean_episode.u))[0]
    good = env.step(act("get_record", table="ledger", id=clean_episode.r2))[0]
    assert [first["remaining_steps"], bad["remaining_steps"], good["remaining_steps"]] == [99, 98, 97]
    assert bad["error"]["code"] == "unknown_id"
    assert first["result"]["row_id"] == clean_episode.r1
    assert good["result"]["row_id"] == clean_episode.r2 and good["error"] is None


# --- C15-C18 ----------------------------------------------------------------------


def test_C15_submit_ends_and_answer_is_frozen(mini_episode):
    env, _, _ = start(mini_episode)
    env.step(act("get_document", doc_id=mini_episode.d1))
    observation, reward, terminated, truncated, _ = env.step(act("submit", findings=oracle(mini_episode)))
    assert observation["result"] == {"accepted": True} and observation["error"] is None
    assert terminated is True and truncated is False and reward == 0
    answer, steps = env.final_findings, env.inspection_steps
    for action in (act("submit", findings={"episode_id": mini_episode.episode_id, "findings": []}),
                   act("get_record", table="ledger", id=mini_episode.r1), "junk"):
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation["error"]["code"] == "episode_over"
        assert (reward, terminated, truncated, info) == (0, True, False, {})
    assert env.final_findings == answer == oracle(mini_episode)
    assert env.inspection_steps == steps == 1


def test_C15_scores_unchanged_after_extra_actions(mini_episode, tmp_path, monkeypatch):
    pytest.importorskip("cookedbooks.grader.score")
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    env, _, _ = start(mini_episode)
    env.step(act("calculator", expression="1"))
    env.step(act("submit", findings=oracle(mini_episode)))
    before = judge_adapter.score_episode(str(mini_episode.path), env.final_findings, env.inspection_steps)
    for _ in range(5):
        env.step(act("submit", findings={"episode_id": "x", "findings": []}))
    after = judge_adapter.score_episode(str(mini_episode.path), env.final_findings, env.inspection_steps)
    assert before == after
    assert (before["tp"], before["fp"], before["fn"]) == (1, 0, 0)


def test_C16_submit_as_action_100(mini_episode):
    env, _, _ = start(mini_episode)
    for i in range(99):
        observation, _, terminated, truncated, _ = env.step(act("calculator", expression="1+1"))
        assert (terminated, truncated) == (False, False)
        assert observation["remaining_steps"] == 99 - i
    observation, reward, terminated, truncated, _ = env.step(act("submit", findings=oracle(mini_episode)))
    assert observation["result"] == {"accepted": True}
    assert (terminated, truncated) == (True, False)
    assert observation["remaining_steps"] == 0
    assert env.inspection_steps == 99


def test_C17_100_actions_without_submit(mini_episode):
    env, _, _ = start(mini_episode)
    for _ in range(99):
        assert env.step(act("get_record", table="ledger", id=mini_episode.r1))[3] is False
    observation, _, terminated, truncated, _ = env.step(act("get_record", table="ledger", id=mini_episode.r1))
    assert truncated is True and terminated is False
    assert observation["remaining_steps"] == 0
    assert env.final_findings == {"episode_id": mini_episode.episode_id, "findings": []}
    late = env.step(act("submit", findings=oracle(mini_episode)))
    assert late[0]["error"]["code"] == "episode_over" and late[3] is True
    assert env.final_findings["findings"] == []


def test_C18_time_limit_injected_clock(mini_episode, monkeypatch):
    clock = FakeClock(100.0)
    monkeypatch.setattr(core.time, "monotonic", clock)
    env, _, _ = start(mini_episode)
    clock.now = 130.0
    assert env.step(act("calculator", expression="1"))[3] is False
    clock.now = 160.5
    observation, _, terminated, truncated, _ = env.step(act("submit", findings=oracle(mini_episode)))
    assert observation["result"] is None and observation["error"]["code"] == "time_limit"
    assert truncated is True and terminated is False
    assert env.final_findings == {"episode_id": mini_episode.episode_id, "findings": []}


def test_C18_time_limit_real_clock(mini_episode):
    env, _, _ = start(mini_episode, team_config={"time_limit_s": 0.05})
    time.sleep(0.1)
    observation, _, terminated, truncated, _ = env.step(act("submit", findings=oracle(mini_episode)))
    assert observation["error"]["code"] == "time_limit"
    assert truncated is True and terminated is False
    assert env.final_findings == {"episode_id": mini_episode.episode_id, "findings": []}


# --- C19-C20 ----------------------------------------------------------------------


def test_C19_closed_reward_is_zero(mini_episode):
    env, _, info = start(mini_episode, mode="closed")
    assert info == {}
    results = [env.step(act("get_document", doc_id=mini_episode.d1)),
               env.step(act("get_document", doc_id=mini_episode.d2)),
               env.step(act("get_record", table="ledger", id=mini_episode.u)),
               env.step(act("submit", findings=oracle(mini_episode))),
               env.step(act("calculator", expression="1"))]
    assert results[3][2] is True
    for observation, reward, _, _, info in results:
        assert type(reward) is float and reward == 0
        assert info == {}
        text = json.dumps(observation)
        for banned in ("metrics", "return", "precision", "recall", "f1", "tp", "fp", "fn"):
            assert f'"{banned}"' not in text


_LEAK_KEYS = {"issue_id", "issues", "evidence_sets", "expected_amount_minor", "injections",
              "distractors", "seed", "metrics", "precision", "recall", "f1", "tp", "fp", "fn",
              "return", "reward"}
_LEAK_WORDS = ("truth", "provenance", "seed", "grader", "traceback", ".json", ".csv", ".jsonl")


def _keys(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _keys(inner)


def _full_walk(env, fixture) -> list:
    """Every page, record and document, every error path, then the end of the episode."""
    outputs = []
    for table in TABLES:
        pages, results = walk_table(env, table)
        outputs += results
        id_field = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "id"}[table]
        for page in pages:
            for record in page["records"]:
                outputs.append(env.step(act("get_record", table=table, id=record[id_field])))
    for doc_id in fixture.doc_ids:
        outputs.append(env.step(act("get_document", doc_id=doc_id)))
    for action in ("junk", {"name": "read_file", "args": {"path": "truth.json"}},
                   {"name": "list_records"}, act("list_records", table="truth"),
                   act("list_records", table="ledger", cursor="bad"),
                   act("list_records", table="ledger", limit=0),
                   act("get_record", table="ledger", id=fixture.u),
                   act("get_record", table="provenance", id=fixture.u),
                   act("get_document", doc_id=fixture.u), act("get_document", doc_id="truth.json"),
                   act("get_document", doc_id=str(fixture.path / "truth.json")),
                   act("calculator", expression="open('truth.json')"), act("calculator", expression="1/0")):
        outputs.append(env.step(action))
    outputs.append(env.step(act("submit", findings={"episode_id": fixture.episode_id, "findings": "?"})))
    outputs.append(env.step(act("get_record", table="ledger", id=fixture.r1)))
    return outputs


def _assert_no_leak(outputs, fixture):
    for output in outputs:
        assert json_roundtrip(list(output) if isinstance(output, tuple) else output)
        if isinstance(output, tuple):
            _, reward, _, _, info = output
            assert reward == 0 and info == {}
        assert not set(_keys(output)) & _LEAK_KEYS
        text = json.dumps(output)
        lowered = text.lower()
        for word in _LEAK_WORDS:
            assert word not in lowered, word
        for path in (fixture.path, fixture.path.parent, fixture.path.resolve()):
            assert str(path) not in text


@pytest.mark.parametrize("which", ["clean_episode", "mini_episode"])
def test_C20_no_leak_anywhere(which, request, audit):
    fixture = request.getfixturevalue(which)
    # Corrupt the closed files: the environment must not need, open or reveal them.
    (fixture.path / "truth.json").write_text('{"issues": [{"issue_id": "LEAKED"}]} {{{', encoding="utf-8")
    (fixture.path / "provenance.json").write_text('{"seed": "LEAKED"', encoding="utf-8")
    audit.clear()
    env, observation, info = start(fixture)
    outputs = [observation, info] + _full_walk(env, fixture)
    assert outputs[-2][2] is True and outputs[-1][0]["error"]["code"] == "episode_over"
    _assert_no_leak(outputs, fixture)
    assert "LEAKED" not in json.dumps(outputs)
    opened = [entry for entry in audit if not entry.startswith("open:") or "truth" in entry
              or "provenance" in entry]
    assert opened == []     # nothing but the public files; no listing, process, socket or SQL
    public = {"manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"}
    episode_opens = {Path(e[len("open:"):]).name for e in audit if e.startswith("open:")
                     and Path(e[len("open:"):]).parent == fixture.path}
    assert "manifest.json" in episode_opens and episode_opens <= public


def test_C20_time_and_step_truncation_do_not_leak(clean_episode, monkeypatch):
    clock = FakeClock(0.0)
    monkeypatch.setattr(core.time, "monotonic", clock)
    env, _, _ = start(clean_episode)
    clock.now = 61.0
    outputs = [env.step(act("calculator", expression="1")), env.step("junk")]
    env.reset(0, {"mode": "closed", "episode_dir": str(clean_episode.path)})
    outputs += [env.step(act("calculator", expression="1")) for _ in range(101)]
    _assert_no_leak(outputs, clean_episode)


_SUBPROCESS_SCRIPT = textwrap.dedent("""
    import json, os, sys, tempfile
    opened = []
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))
    sys.addaudithook(hook)
    import judge_adapter
    from tests.helpers.factory import build_mini
    with tempfile.TemporaryDirectory() as tmp:
        fixture = build_mini(os.path.join(tmp, "e"))
        opened.clear()
        env = judge_adapter.make_env()
        env.reset(0, {"mode": "closed", "episode_dir": str(fixture.path), "team_config": {}})
        for table in ("ledger", "invoices", "payments", "fx"):
            env.step({"name": "list_records", "args": {"table": table, "cursor": None, "limit": 50}})
        env.step({"name": "get_document", "args": {"doc_id": fixture.d1}})
        env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
        findings = {"episode_id": fixture.episode_id, "findings": []}
        env.step({"name": "submit", "args": {"findings": findings}})
        env.step({"name": "submit", "args": {"findings": findings}})
        env.reset(0, {"mode": "closed", "episode_dir": str(fixture.path), "team_config": {}})
        env.step({"name": "submit", "args": {"findings": None}})
    modules = sorted(m for m in sys.modules if m.startswith("cookedbooks"))
    print(json.dumps({"modules": modules, "opened": opened}))
""")


def test_C20_closed_episode_never_imports_grader_or_opens_truth():
    completed = subprocess.run([sys.executable, "-c", _SUBPROCESS_SCRIPT], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=60,
                               env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    imported = report["modules"]
    assert "cookedbooks.env.core" in imported
    assert not [m for m in imported if m == "cookedbooks.grader" or m.startswith("cookedbooks.grader.")]
    assert not [m for m in imported if m.startswith(("cookedbooks.world", "cookedbooks.verifier"))]
    names = {Path(p).name for p in report["opened"]}
    assert "truth.json" not in names and "provenance.json" not in names
    assert "manifest.json" in names   # the hook did see the episode being loaded

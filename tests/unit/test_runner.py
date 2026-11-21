"""The runner and its process boundary (docs/jury-checks.md, "Граница доступа").

Every run here starts a real child process. Misbehaving children are test-only
policy modules handed to the runner with `extra_policy_sources`; they are
written into the child's private copy of the package, never into the repo.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.runner.remote import available_policies
from cookedbooks.runner.run import run_episode

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("cookedbooks.grader", "cookedbooks.world", "cookedbooks.verifier", "cookedbooks.episode",
             "cookedbooks.env", "cookedbooks.pipeline", "cookedbooks.verify", "cookedbooks.runner",
             "judge_adapter")
SEED = 930001  # not a seed of the jury-recipe tests
NON_ORACLE = [p for p in available_policies() if p != "oracle"]


@pytest.fixture(scope="module", autouse=True)
def grader_log(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path_factory.mktemp("log") / "grader.jsonl"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def episode(tmp_path_factory) -> Path:
    """Books with one error of each type (and look-alikes)."""
    folder = tmp_path_factory.mktemp("episode") / "books"
    config = json.loads((ROOT / "configs" / "matrix" / "D11.json").read_text())
    config["errors"].update({"duplicate_invoice": 1, "fx_error": {"wrong_date": 1, "wrong_pair": 0},
                             "cutoff_error": {"late": 1, "early": 0}})
    judge_adapter.generate_episode(SEED, config, str(folder))
    return folder


def _secrets(folder: Path) -> list[str]:
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    return ([i["issue_id"] for i in truth["issues"]] + [i["description"] for i in provenance["injections"]]
            + ["truth", "provenance", str(folder), str(ROOT), os.path.realpath(folder)])


def _has_number(text: str, number: int) -> bool:
    return re.search(r"(?<![0-9])%d(?![0-9])" % number, text) is not None


# --- what the child can and cannot see ---------------------------------------------------


@pytest.mark.parametrize("policy", available_policies())
def test_policy_child_never_loads_judge_modules(policy, episode):
    run = run_episode(policy, episode, mode="closed", policy_seed=0, debug=True)
    assert run.failure is None and run.debug is not None
    loaded = run.debug["modules"]
    assert "cookedbooks.policies." + policy in loaded
    assert not [m for m in loaded if m.startswith(FORBIDDEN)]


def test_child_working_directory_is_empty_and_private(episode):
    run = run_episode("simple_recon", episode, mode="closed", policy_seed=0, debug=True)
    cwd = os.path.realpath(run.debug["cwd"])
    assert run.debug["cwd_entries"] == []
    for place in (ROOT, episode):
        place = os.path.realpath(place)
        assert cwd != place and not cwd.startswith(place + os.sep) and not place.startswith(cwd + os.sep)
    assert not os.path.exists(cwd), "the working directory is removed after the run"


@pytest.mark.parametrize("policy", NON_ORACLE)
def test_child_never_sees_truth_paths_seeds_or_rewards_in_closed_mode(policy, episode):
    run = run_episode(policy, episode, mode="closed", policy_seed=0, debug=True)
    received = run.debug["received"]
    text = "\n".join(received)
    for secret in _secrets(episode):
        assert secret not in text, secret
    assert not _has_number(text, SEED)
    messages = [json.loads(line) for line in received]
    assert messages[0]["type"] == "start" and messages[0]["options"] == {}
    assert sorted(messages[0]) == ["debug", "options", "policy", "seed", "type"]
    rewards = [m["reward"] for m in messages if "reward" in m]
    assert rewards and all(r == 0 for r in rewards)
    assert messages[-1]["type"] == "end"
    # It received exactly the observations the environment produced, and nothing else.
    sent = [m["observation"] for m in messages if m["type"] in ("observation", "end")]
    assert sent == [r["observation"] for r in run.records]


def test_train_mode_gives_the_terminal_reward_to_the_child(episode):
    run = run_episode("evidence_solver" if "evidence_solver" in available_policies() else "simple_recon",
                      episode, mode="train", policy_seed=0, debug=True)
    end = json.loads(run.debug["received"][-1])
    assert end["type"] == "end" and end["terminated"] is True
    assert end["reward"] == run.records[-1]["reward"] == run.metrics["return"]


def test_sandbox_refuses_files_imports_processes_and_network(episode):
    probe = f'''
import os
RESULTS = {{}}
def attempt(label, fn):
    try:
        fn()
        RESULTS[label] = "allowed"
    except BaseException as exc:
        RESULTS[label] = type(exc).__name__
class Probe:
    def act(self, observation):
        attempt("import_grader", lambda: __import__("cookedbooks.grader.score"))
        attempt("import_env", lambda: __import__("cookedbooks.env.core"))
        attempt("import_adapter", lambda: __import__("judge_adapter"))
        attempt("read_truth", lambda: open({str(episode / "truth.json")!r}).read())
        attempt("read_repo", lambda: open({str(ROOT / "AGENTS.md")!r}).read())
        attempt("list_episode", lambda: os.listdir({str(episode)!r}))
        attempt("write_outside", lambda: open({str(episode / "x.txt")!r}, "w"))
        attempt("subprocess", lambda: __import__("subprocess").run(["true"]))
        attempt("system", lambda: os.system("true"))
        attempt("socket", lambda: __import__("socket").socket())
        attempt("ctypes", lambda: __import__("ctypes"))
        attempt("chdir", lambda: os.chdir("/"))
        return {{"name": "submit", "args": {{"findings": {{"episode_id": "", "findings": []}}, "probe": RESULTS}}}}
def make_policy(seed, **options):
    return Probe()
'''
    run = run_episode("probe", episode, mode="closed", policy_seed=0, extra_policy_sources={"probe": probe})
    results = run.records[-1]["action"]["args"]["probe"]
    assert results and all(outcome == "Refused" for outcome in results.values()), results
    assert not (episode / "x.txt").exists()


# --- misbehaving children end the episode cleanly -------------------------------------------

BAD_CHILDREN = {
    "crash": "raise RuntimeError('boom')",
    "hard_exit": "os._exit(3)",
    "exits_early": "os._exit(0)",
    "garbage": "[_write(fd, b'this is not json\\n') for fd in range(3, 32)]\n        time.sleep(30)",
    "babble": "[_write(fd, b'x' * (5 << 20)) for fd in range(3, 32)]\n        time.sleep(30)",
    "sleeps": "time.sleep(30)",
    "unserialisable": "return {'name': 'list_records', 'args': {'table': {1, 2}}}",
    "not_an_action_message": "[_write(fd, b'{\"type\": \"hello\"}\\n') for fd in range(3, 32)]\n        time.sleep(30)",
    "nan_action": "[_write(fd, b'{\"type\": \"action\", \"action\": NaN}\\n') for fd in range(3, 32)]\n        time.sleep(30)",
}


EXPECTED_FAILURE = {"crash": "crash", "hard_exit": "exited", "exits_early": "exited", "garbage": "garbage",
                    "babble": "garbage", "sleeps": "timeout", "unserialisable": "crash",
                    "not_an_action_message": "garbage", "nan_action": "garbage"}


def _bad_policy(body: str) -> str:
    return f'''
import os, time
def _write(fd, data):
    try:
        os.write(fd, data)
    except OSError:
        pass
class Bad:
    def __init__(self):
        self.calls = 0
    def act(self, observation):
        self.calls += 1
        if self.calls < 3:
            return {{"name": "list_records", "args": {{"table": "ledger", "cursor": None, "limit": 2}}}}
        {body}
def make_policy(seed, **options):
    return Bad()
'''


@pytest.mark.parametrize("kind", sorted(BAD_CHILDREN))
def test_misbehaving_child_is_truncated_with_an_empty_answer(kind, episode):
    started = time.monotonic()
    run = run_episode("bad", episode, mode="train", policy_seed=0, time_limit_s=1.0,
                      extra_policy_sources={"bad": _bad_policy(BAD_CHILDREN[kind])})
    assert time.monotonic() - started < 10
    assert run.failure == EXPECTED_FAILURE[kind]
    assert run.truncated and not run.terminated
    manifest = json.loads((episode / "manifest.json").read_text())
    assert run.final_findings == {"episode_id": manifest["episode_id"], "findings": []}
    assert run.inspection_steps == 2
    expected = judge_adapter.score_episode(str(episode), run.final_findings, 2)
    assert run.metrics == expected and run.metrics["return"] < 0


def test_failing_make_policy_still_scores_an_empty_answer(episode):
    source = "def make_policy(seed, **options):\n    raise ValueError('no')\n"
    run = run_episode("broken", episode, mode="closed", policy_seed=0, extra_policy_sources={"broken": source})
    assert run.failure == "crash" and run.truncated and run.inspection_steps == 0
    assert run.records[0]["step"] == 0 and len(run.records) == 1


def test_slow_child_hits_the_time_limit_from_the_environment(episode):
    """A child that answers after the budget gets `time_limit` from the environment itself."""
    body = "time.sleep(1.2)\n        return {'name': 'submit', 'args': {'findings': {'episode_id': '', 'findings': []}}}"
    run = run_episode("slow", episode, mode="closed", policy_seed=0, time_limit_s=1.0,
                      extra_policy_sources={"slow": _bad_policy(body)})
    assert run.failure is None and run.truncated
    assert run.records[-1]["observation"]["error"]["code"] == "time_limit"


def test_unknown_policy_is_rejected(episode):
    with pytest.raises(ValueError):
        run_episode("no_such_policy", episode, mode="closed", policy_seed=0)
    with pytest.raises(ValueError):
        run_episode("simple_recon", episode, mode="open", policy_seed=0)


# --- rewards by mode ------------------------------------------------------------------------


def test_closed_rewards_are_zero_and_train_terminal_reward_is_the_score(episode):
    closed = run_episode("simple_recon", episode, mode="closed", policy_seed=0)
    train = run_episode("simple_recon", episode, mode="train", policy_seed=0)
    assert all(r["reward"] == 0.0 for r in closed.records)
    assert all(r["reward"] == 0.0 for r in train.records[:-1])
    expected = judge_adapter.score_episode(str(episode), closed.final_findings, closed.inspection_steps)
    assert closed.metrics == expected == train.metrics
    assert train.records[-1]["reward"] == expected["return"]
    assert [r["action"] for r in closed.records] == [r["action"] for r in train.records]
    assert closed.records[0] == {"step": 0, "action": None, "observation": closed.records[0]["observation"],
                                 "reward": 0.0, "terminated": False, "truncated": False}
    assert closed.records[-1]["terminated"] and closed.inspection_steps == len(closed.records) - 2


def test_oracle_answer_is_built_on_the_judge_side(episode):
    run = run_episode("oracle", episode, mode="closed", policy_seed=0)
    assert run.inspection_steps == 0 and run.metrics["return"] == 1.0 and run.metrics["fn"] == 0


def test_cli_prints_the_summary_and_writes_the_trajectory(episode, tmp_path):
    out = tmp_path / "run.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "cookedbooks.runner.run", "--policy", "oracle", "--episode", str(episode),
         "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=60,
        env={**os.environ, "COOKEDBOOKS_GRADER_LOG": str(tmp_path / "log.jsonl")})
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["return"] == 1.0 and summary["mode"] == "closed" and summary["failure"] is None
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["step"] for line in lines] == [0, 1] and lines[1]["action"]["name"] == "submit"

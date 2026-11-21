"""Train mode of AuditEnv: grader wiring (with a fake grader) and generation wiring."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from cookedbooks.env.core import AuditEnv
from cookedbooks.env.errors import error
from tests.helpers.factory import build_mini
from tests.unit.test_env import FakeClock, act, oracle

FAKE_METRICS = {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "return": 0.5}
INVALID_METRICS = {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None,
                   "f1": None, "return": -1}


@pytest.fixture
def fake_grader(monkeypatch):
    """Replace cookedbooks.grader.score with a recorder; returns the list of calls."""
    calls: list[tuple] = []
    module = types.ModuleType("cookedbooks.grader.score")

    def score_episode(episode_dir, findings, inspection_steps):
        calls.append((episode_dir, findings, inspection_steps))
        if not isinstance(findings, dict) or "findings" not in findings:
            return dict(INVALID_METRICS)
        return dict(FAKE_METRICS)

    module.score_episode = score_episode
    monkeypatch.setitem(sys.modules, "cookedbooks.grader.score", module)
    return calls


def train(fixture, clock=None, team_config=None) -> AuditEnv:
    env = AuditEnv(clock=clock)
    env.reset(0, {"mode": "train", "episode_dir": str(fixture.path), "team_config": team_config or {}})
    return env


def test_exploration_rewards_zero_then_metrics_on_submit(mini_episode, fake_grader):
    env = train(mini_episode)
    assert env.mode == "train"
    for _ in range(3):
        result = env.step(act("get_document", doc_id="7"))
        assert result[1:] == (0.0, False, False, {})
    env.step("junk")
    result = env.step(act("submit", findings=oracle(mini_episode)))
    assert result[0] == {"result": {"accepted": True}, "error": None, "remaining_steps": 95}
    assert result[1] == 0.5 and type(result[1]) is float
    assert result[2:4] == (True, False)
    assert result[4] == {"metrics": FAKE_METRICS}
    assert fake_grader == [(str(mini_episode.path), oracle(mini_episode), 4)]
    assert env.inspection_steps == 4


def test_invalid_submit_scored_as_structure_error(mini_episode, fake_grader):
    env = train(mini_episode)
    result = env.step(act("submit", findings=[1, 2]))
    assert result[0]["error"] == error("invalid_submission")
    assert result[1] == -1.0 and type(result[1]) is float
    assert result[4] == {"metrics": INVALID_METRICS}
    assert fake_grader == [(str(mini_episode.path), [1, 2], 0)]
    assert env.final_findings == [1, 2]


@pytest.mark.parametrize("action", [
    {"name": "submit"},
    {"name": "submit", "args": None},
    {"name": "submit", "args": [1]},
    {"name": "submit", "args": {}},
    {"name": "submit", "args": {"other": 1}},
])
def test_submit_without_findings_scored_as_structure_error(mini_episode, fake_grader, action):
    env = train(mini_episode)
    env.step(act("calculator", expression="1"))
    result = env.step(action)
    assert result[0] == {"result": None, "error": error("invalid_submission"), "remaining_steps": 98}
    assert result[1] == -1.0 and result[2:4] == (True, False)
    assert result[4] == {"metrics": INVALID_METRICS}
    assert fake_grader == [(str(mini_episode.path), None, 1)]
    assert env.final_findings is None and env.inspection_steps == 1


def test_step_truncation_scores_empty_answer(mini_episode, fake_grader):
    env = train(mini_episode)
    for _ in range(99):
        assert env.step(act("calculator", expression="1"))[1] == 0.0
    result = env.step(act("calculator", expression="1"))
    assert result[2:4] == (False, True)
    assert result[1] == 0.5 and result[4] == {"metrics": FAKE_METRICS}
    assert fake_grader == [(str(mini_episode.path), {"episode_id": "ep-mini", "findings": []}, 100)]


def test_time_truncation_scores_empty_answer(mini_episode, fake_grader):
    clock = FakeClock(0.0)
    env = train(mini_episode, clock=clock, team_config={"time_limit_s": 10})
    env.step(act("calculator", expression="1"))
    clock.now = 10.5
    result = env.step(act("submit", findings=oracle(mini_episode)))
    assert result[0]["error"] == error("time_limit")
    assert result[2:4] == (False, True)
    assert result[4] == {"metrics": FAKE_METRICS}
    assert fake_grader == [(str(mini_episode.path), {"episode_id": "ep-mini", "findings": []}, 2)]


def test_after_end_no_more_scoring(mini_episode, fake_grader):
    env = train(mini_episode)
    env.step(act("submit", findings=oracle(mini_episode)))
    for _ in range(3):
        result = env.step(act("submit", findings={"episode_id": "x", "findings": []}))
        assert result[1:] == (0.0, True, False, {})
        assert result[0]["error"] == error("episode_over")
    assert len(fake_grader) == 1


def test_closed_mode_never_calls_grader(mini_episode, monkeypatch):
    module = types.ModuleType("cookedbooks.grader.score")

    def explode(*args, **kwargs):
        raise AssertionError("grader called in closed mode")

    module.score_episode = explode
    monkeypatch.setitem(sys.modules, "cookedbooks.grader.score", module)
    env = AuditEnv()
    env.reset(0, {"mode": "closed", "episode_dir": str(mini_episode.path)})
    assert env.step(act("submit", findings=oracle(mini_episode)))[1:] == (0.0, True, False, {})


# --- generation wiring ------------------------------------------------------------


@pytest.fixture
def fake_generator(monkeypatch):
    calls: list[tuple] = []
    package = types.ModuleType("cookedbooks.world")
    module = types.ModuleType("cookedbooks.world.generate")

    def generate_episode(seed, team_config, output_dir, *, attempt=0):
        calls.append((seed, team_config, output_dir))
        build_mini(output_dir)

    module.generate_episode = generate_episode
    package.generate = module
    monkeypatch.setitem(sys.modules, "cookedbooks.world", package)
    monkeypatch.setitem(sys.modules, "cookedbooks.world.generate", module)
    # Train-mode generation goes through cookedbooks.pipeline, which runs the
    # verifier on every attempt; the hand-built mini episode is not a generator
    # output, so the verifier is faked to accept it.
    from cookedbooks import pipeline
    monkeypatch.setattr(pipeline, "verify_episode", lambda folder: types.SimpleNamespace(ok=True))
    monkeypatch.setattr(pipeline, "STATS", pipeline.GenerationStats())
    return calls


def test_train_without_episode_dir_generates(fake_generator, fake_grader):
    env = AuditEnv()
    observation, info = env.reset(7, {"mode": "train", "team_config": {"time_limit_s": 30}})
    assert info == {} and observation["manifest"]["episode_id"] == "ep-mini"
    (seed, team_config, output_dir), = fake_generator
    assert seed == 7 and team_config == {"time_limit_s": 30}
    assert Path(output_dir).is_dir()
    result = env.step(act("submit", findings=oracle_like()))
    assert fake_grader[0][0] == output_dir and result[1] == 0.5
    env.reset(8, {"mode": "train"})
    assert not Path(output_dir).exists()  # the previous temporary folder is removed
    second = fake_generator[1][2]
    env.close()
    assert not Path(second).exists()


def oracle_like() -> dict:
    return {"episode_id": "ep-mini", "findings": []}


def test_missing_generator_is_a_clear_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cookedbooks.world.generate", None)
    with pytest.raises(RuntimeError, match="generator"):
        AuditEnv().reset(1, {"mode": "train"})


def test_closed_without_episode_dir_never_generates(fake_generator):
    with pytest.raises(ValueError):
        AuditEnv().reset(1, {"mode": "closed", "team_config": {}})
    assert fake_generator == []


def test_real_generator(tmp_path, monkeypatch):
    pytest.importorskip("cookedbooks.world.generate")
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    env = AuditEnv()
    observation, _ = env.reset(20261120, {"mode": "train", "team_config": {}})
    assert observation["remaining_steps"] == 100
    result = env.step(act("submit", findings={"episode_id": observation["manifest"]["episode_id"],
                                              "findings": []}))
    assert result[2] is True and "metrics" in result[4]
    env.close()


# --- the real grader --------------------------------------------------------------


@pytest.fixture
def real_grader(tmp_path, monkeypatch):
    pytest.importorskip("cookedbooks.grader.score")
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader-log.jsonl"))


def test_real_oracle_after_ten_steps(mini_episode, real_grader):
    env = train(mini_episode)
    for _ in range(10):
        env.step(act("get_document", doc_id="7"))
    result = env.step(act("submit", findings=oracle(mini_episode)))
    assert result[1] == pytest.approx(0.98, abs=1e-9)
    metrics = result[4]["metrics"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)
    assert metrics["return"] == pytest.approx(0.98, abs=1e-9)


def test_real_truncation_on_clean(clean_episode, real_grader):
    env = train(clean_episode)
    for _ in range(100):
        result = env.step(act("calculator", expression="1"))
    assert result[2:4] == (False, True)
    assert result[1] == pytest.approx(0.8, abs=1e-9)
    assert result[4]["metrics"]["f1"] == 1


def test_real_invalid_submit(mini_episode, real_grader):
    env = train(mini_episode)
    result = env.step(act("submit", findings={"episode_id": "ep-mini", "findings": [{}]}))
    assert result[1] == -1.0
    assert result[4]["metrics"]["tp"] is None


@pytest.mark.parametrize("action", [{"name": "submit"}, {"name": "submit", "args": {}}])
def test_real_bare_submit_is_minus_one(mini_episode, real_grader, action):
    env = train(mini_episode)
    for _ in range(3):
        env.step(act("calculator", expression="1"))
    result = env.step(action)
    assert result[0]["error"] == error("invalid_submission")
    assert result[1] == -1.0 and result[2:4] == (True, False)
    assert result[4]["metrics"]["return"] == -1 and result[4]["metrics"]["tp"] is None
    assert env.inspection_steps == 3


def test_real_grader_log_not_in_episode_dir(mini_episode, real_grader):
    before = sorted(os.listdir(mini_episode.path))
    env = train(mini_episode)
    env.step(act("submit", findings=oracle(mini_episode)))
    assert sorted(os.listdir(mini_episode.path)) == before

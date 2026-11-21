"""trajectories.jsonl: format, rewards, exact replay and a leak scan."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.runner import trajectories
from cookedbooks.runner.episodes import episodes

PATH = trajectories.TRAJECTORIES_PATH
FIELDS = {"episode_id", "policy", "split", "step", "action", "observation", "reward", "terminated", "truncated"}
TRAIN = episodes("train")


@pytest.fixture(scope="module", autouse=True)
def grader_log(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path_factory.mktemp("log") / "grader.jsonl"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def lines() -> list[dict]:
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def runs(lines) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in lines:
        out[(line["policy"], line["episode_id"])].append(line)
    return dict(out)


@pytest.fixture(scope="module")
def folders() -> dict[str, Path]:
    return {json.loads((e.folder() / "manifest.json").read_text())["episode_id"]: e.folder() for e in TRAIN}


def test_committed_file_equals_a_fresh_build():
    assert PATH.read_text(encoding="utf-8") == trajectories.render()


def test_file_is_small_and_covers_every_baseline_on_every_train_episode(runs, folders):
    assert PATH.stat().st_size < 5 * 1024 * 1024
    assert len(folders) == 10
    assert set(runs) == {(p, e) for p in trajectories.policies() for e in folders}
    assert {"random_explorer", "simple_recon", "flag_all", "oracle"} <= set(trajectories.policies())


def test_every_line_has_exactly_the_documented_fields(lines):
    for line in lines:
        assert set(line) == FIELDS
        assert isinstance(line["episode_id"], str) and isinstance(line["policy"], str)
        assert line["split"] == "train"
        assert isinstance(line["step"], int) and not isinstance(line["step"], bool)
        # Step 0 has no action; later steps record whatever the policy sent (random_explorer
        # deliberately sends some invalid actions, such as a bare string).
        assert (line["action"] is None) == (line["step"] == 0)
        assert isinstance(line["observation"], dict)
        assert isinstance(line["reward"], float)
        assert isinstance(line["terminated"], bool) and isinstance(line["truncated"], bool)


def test_steps_are_consecutive_and_start_with_the_reset(runs):
    for records in runs.values():
        assert [r["step"] for r in records] == list(range(len(records)))
        first = records[0]
        assert first["action"] is None and first["reward"] == 0.0
        assert set(first["observation"]) == {"task", "manifest", "policy", "remaining_steps"}
        assert all(r["action"] is not None for r in records[1:])
        assert all(set(r["observation"]) == {"result", "error", "remaining_steps"} for r in records[1:])
        assert not any(r["terminated"] or r["truncated"] for r in records[:-1])
        assert records[-1]["terminated"] or records[-1]["truncated"]


def test_exploration_rewards_are_zero_and_the_terminal_reward_is_the_score(runs, folders):
    for (_, episode_id), records in runs.items():
        assert all(r["reward"] == 0.0 for r in records[:-1])
        last = records[-1]
        if last["terminated"]:
            findings, steps = last["action"]["args"]["findings"], last["step"] - 1
        else:
            findings, steps = {"episode_id": episode_id, "findings": []}, last["step"]
        expected = judge_adapter.score_episode(str(folders[episode_id]), findings, steps)
        assert last["reward"] == expected["return"]


def test_replaying_the_actions_reproduces_every_record(runs, folders):
    for (policy, episode_id), records in runs.items():
        env = judge_adapter.make_env()
        observation, _ = env.reset(None, {"mode": "train", "episode_dir": str(folders[episode_id])})
        assert json.loads(json.dumps(observation)) == records[0]["observation"], (policy, episode_id)
        for record in records[1:]:
            observation, reward, terminated, truncated, _ = env.step(copy.deepcopy(record["action"]))
            replay = {"observation": json.loads(json.dumps(observation)), "reward": reward,
                      "terminated": terminated, "truncated": truncated}
            assert replay == {k: record[k] for k in replay}, (policy, episode_id, record["step"])


def _secrets(folder: Path) -> list[str]:
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    texts = [i["issue_id"] for i in truth["issues"]] + [i["description"] for i in provenance["injections"]]
    texts += sorted({d["kind"] for d in provenance["distractors"]})
    texts += ["truth", "provenance", "issue_id", "expected_amount_minor", "evidence_sets", "injections",
              "distractors", "generator_version", "team_config"]
    return texts


def test_observations_leak_no_truth_paths_or_seeds(runs, folders):
    path_like = re.compile(r"(?:/Users/|/home/|/tmp/|/private/|/var/|[A-Za-z]:\\\\|episodes/|\.json\b|\.csv\b)")
    seeds = [e.seed for e in TRAIN]
    for (policy, episode_id), records in runs.items():
        secrets = _secrets(folders[episode_id])
        for record in records:
            text = json.dumps(record["observation"], ensure_ascii=False)
            for secret in secrets:
                assert secret not in text, (policy, episode_id, record["step"], secret)
            assert not path_like.search(text), (policy, episode_id, record["step"])
            for seed in seeds:
                assert not re.search(r"(?<![0-9])%d(?![0-9])" % seed, text), (policy, episode_id, seed)

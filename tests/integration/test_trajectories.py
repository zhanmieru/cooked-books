"""trajectories.jsonl: format, rewards, exact replay and a leak scan."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.env.core import task_text
from cookedbooks.episode.loader import load_public
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


# --- leak scan ------------------------------------------------------------------------------
#
# The task text is one constant string and the reset's `manifest` and `policy` are the public
# files verbatim (both checked first), so none of them can carry anything about an episode
# beyond the public files. Every observation, all of it, is scanned for the episode's
# closed-only values: every string or number in truth.json / provenance.json that occurs
# nowhere in the public files (issue IDs, descriptions, the seed, the generator version, values
# from before an injected change, ...), and for paths. Generic vocabulary (type, variant,
# scenario and look-alike names, field names of the closed files) may appear in the task text,
# which explains the rules, and as the keys of the public policy (amount tolerances per type),
# but nowhere else: in records, results or errors it could only have come from a closed file.

ERROR_TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")
VARIANTS = ("missing", "partial", "wrong_date", "wrong_pair", "late", "early")
SCENARIOS = ("S1", "S2")
LOOK_ALIKES = ("split_payment", "refund", "recurring_amount", "boundary", "advance", "overpayment")
CLOSED_FIELDS = ("truth", "provenance", "issue_id", "issue_ids", "expected_amount_minor", "evidence_sets",
                 "injections", "distractors", "generator_version", "team_config")
GENERIC_VOCABULARY = ERROR_TYPES + VARIANTS + SCENARIOS + LOOK_ALIKES + CLOSED_FIELDS
PATH_LIKE = re.compile(r"(?:/Users/|/home/|/tmp/|/private/|/var/|[A-Za-z]:\\\\|episodes/|\.jsonl?\b|\.csv\b)")
MIN_SECRET_CHARS = 4      # shorter closed-only strings ("add", "C5") are schema words, not data
MIN_SECRET_NUMBER = 10000  # smaller integers occur everywhere (step counts, limits, quantities)


@dataclass(frozen=True)
class Secrets:
    strings: frozenset[str]
    numbers: frozenset[int]


def _leaves(value, strings: set, numbers: set, *, keys: bool) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if keys:
                strings.add(key)
            _leaves(item, strings, numbers, keys=keys)
    elif isinstance(value, list):
        for item in value:
            _leaves(item, strings, numbers, keys=keys)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        numbers.add(value)
    elif isinstance(value, str):
        strings.add(value)


def episode_secrets(folder: Path) -> Secrets:
    """Values of truth.json and provenance.json that the public files never contain."""
    public = load_public(folder)
    public_strings: set = set()
    public_numbers: set = set()
    _leaves([public.manifest, public.policy, {t: list(r) for t, r in public.tables.items()}],
            public_strings, public_numbers, keys=True)
    closed_strings: set = set()
    closed_numbers: set = set()
    for name in ("truth.json", "provenance.json"):
        _leaves(json.loads((folder / name).read_text()), closed_strings, closed_numbers, keys=False)
    public_text = "\n".join(sorted(public_strings))  # a value inside a public string is public
    strings = {x for x in closed_strings - public_strings
               if len(x) >= MIN_SECRET_CHARS and x not in GENERIC_VOCABULARY and x not in public_text}
    numbers = {n for n in closed_numbers - public_numbers if abs(n) >= MIN_SECRET_NUMBER}
    return Secrets(frozenset(strings), frozenset(numbers))


def leak_problems(observation: dict, secrets: Secrets) -> list[str]:
    """Everything in `observation` that could only have come from the closed files or the judge."""
    task = observation.get("task") if isinstance(observation.get("task"), str) else ""
    rest = json.dumps({k: v for k, v in observation.items() if k != "task"}, ensure_ascii=False)
    # Checked verbatim against the public files elsewhere; their keys name the four types.
    data = json.dumps({k: v for k, v in observation.items() if k not in ("task", "manifest", "policy")},
                      ensure_ascii=False)
    problems = []
    for where, text in (("task", task), ("observation", rest)):
        problems += [f"{where}: closed value {x!r}" for x in sorted(secrets.strings) if x in text]
        problems += [f"{where}: closed number {n}" for n in sorted(secrets.numbers)
                     if re.search(r"(?<![0-9])%d(?![0-9])" % n, text)]
        if PATH_LIKE.search(text):
            problems.append(f"{where}: path")
    problems += [f"observation: closed vocabulary {w!r}" for w in GENERIC_VOCABULARY
                 if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(w), data)]
    return problems


def test_task_text_is_one_constant_string(runs):
    limits = {e.name: float(e.team_config.get("time_limit_s", 60)) for e in TRAIN}
    assert set(limits.values()) == {60.0}
    tasks = {records[0]["observation"]["task"] for records in runs.values()}
    assert tasks == {task_text(60.0)}


def test_reset_manifest_and_policy_are_the_public_files_verbatim(runs, folders):
    for (_, episode_id), records in runs.items():
        observation = records[0]["observation"]
        for name in ("manifest", "policy"):
            assert observation[name] == json.loads((folders[episode_id] / f"{name}.json").read_text())


def test_observations_leak_no_truth_paths_or_seeds(runs, folders):
    secrets = {episode_id: episode_secrets(folder) for episode_id, folder in folders.items()}
    for episode_id, folder in folders.items():
        seed = json.loads((folder / "provenance.json").read_text())["seed"]
        assert seed in secrets[episode_id].numbers  # the scan really looks for the seed
        assert all(i["issue_id"] in secrets[episode_id].strings
                   for i in json.loads((folder / "truth.json").read_text())["issues"])
    for (policy, episode_id), records in runs.items():
        for record in records:
            assert leak_problems(record["observation"], secrets[episode_id]) == [], \
                (policy, episode_id, record["step"])


def test_leak_scan_flags_closed_data_outside_the_task_text(runs, folders):
    """The scan bites: closed data planted in a real observation is flagged, the task text's words are not."""
    entry = next(e for e in TRAIN if e.name == "train-03")  # two duplicates next to recurring amounts
    folder = entry.folder()
    episode_id = json.loads((folder / "manifest.json").read_text())["episode_id"]
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    secrets = episode_secrets(folder)
    records = runs[("simple_recon", episode_id)]
    reset, page = records[0]["observation"], records[1]["observation"]
    assert leak_problems(reset, secrets) == [] and leak_problems(page, secrets) == []

    def planted(change) -> dict:
        observation = copy.deepcopy(page)
        change(observation)
        return observation

    issue_id = truth["issues"][0]["issue_id"]
    description = provenance["injections"][0]["description"]
    kind = provenance["distractors"][0]["kind"]
    cases = {
        "issue id in a record": planted(lambda o: o["result"]["records"][0].update(note=issue_id)),
        "description in an error": planted(lambda o: o.update(error={"code": "x", "message": description})),
        "look-alike kind in a record": planted(lambda o: o["result"]["records"][0].update(kind=kind)),
        "error type in a result": planted(lambda o: o.update(result={"type": "duplicate_invoice"})),
        "seed in remaining_steps": planted(lambda o: o.update(remaining_steps=provenance["seed"])),
        "path in a record": planted(lambda o: o["result"]["records"][0].update(src=str(folder))),
    }
    for label, observation in cases.items():
        assert leak_problems(observation, secrets), label
    # A closed value inside the task text is flagged too; generic words there are not.
    assert leak_problems({**reset, "task": reset["task"] + " " + issue_id}, secrets)
    assert leak_problems({**reset, "task": reset["task"] + " " + kind + " advance refund"}, secrets) == []

"""Run one policy on one episode folder across the process boundary.

    run = run_episode("simple_recon", "episodes/dev/dev-01", mode="closed", policy_seed=0)

The judge side (this process) holds the episode folder, the environment
(`judge_adapter.make_env()`) and the scorer. The policy runs in a child
process (`cookedbooks.runner.remote.PolicyProcess`) and only ever receives
observations and the reward the contract gives it: in `closed` mode always 0,
and after the episode the true metrics come from `judge_adapter.score_episode`
on this side; in `train` mode the terminal reward comes back from the
environment and the child sees it in the final `end` message.

The 60-second budget is enforced twice: the environment answers a late
action with `time_limit`, and the runner stops waiting for a child at the
same deadline plus a short grace. A child that crashes, hangs, exits early
or sends garbage ends the episode as truncated with the empty answer.

CLI:
    python -m cookedbooks.runner.run --policy NAME --episode DIR [--mode closed|train] [--out FILE]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from cookedbooks.runner.remote import PolicyFailure, PolicyProcess

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIME_LIMIT_S = 60.0
DEADLINE_GRACE_S = 1.0
MODES = ("closed", "train")


def _judge_adapter() -> Any:
    try:
        import judge_adapter
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT))
        import judge_adapter
    return judge_adapter


def _oracle_options(episode_dir: str) -> dict:
    """Judge side: the oracle's answer is built here, from the closed truth."""
    from cookedbooks.grader.oracle import oracle_findings

    return {"answer": oracle_findings(episode_dir)}


# Policies whose options are built on the judge side from the episode folder.
JUDGE_SIDE_OPTIONS: dict[str, Callable[[str], dict]] = {"oracle": _oracle_options}


@dataclass
class EpisodeRun:
    """Everything the judge side learned from one run."""

    policy: str
    episode_id: str
    mode: str
    records: list[dict]          # trajectory: step 0 is the reset (action None)
    final_findings: Any          # the submitted object, or the empty answer on truncation
    inspection_steps: int
    terminated: bool
    truncated: bool
    metrics: dict                # score_episode's result for final_findings and inspection_steps
    wall_time_s: float           # from reset to the end of the episode
    failure: str | None = None   # crash / timeout / garbage / exited, when the child failed
    debug: dict | None = field(default=None, repr=False)

    @property
    def episode_return(self) -> float:
        value = self.metrics.get("return")
        return float(value) if isinstance(value, (int, float)) else -1.0


def _record(step: int, action: Any, observation: Any, reward: float, terminated: bool, truncated: bool) -> dict:
    return {"step": step, "action": copy.deepcopy(action), "observation": copy.deepcopy(observation),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)}


def run_episode(policy_name: str, episode_dir: str | Path, *, mode: str, policy_seed: int,
                policy_options: Mapping[str, Any] | None = None, time_limit_s: float | None = None,
                debug: bool = False, extra_policy_sources: Mapping[str, str] | None = None) -> EpisodeRun:
    """Run `policy_name` on the episode in `episode_dir` and return what happened.

    `time_limit_s` lowers the episode budget through `team_config.time_limit_s`.
    `debug=True` asks the child for its debug report (modules loaded at exit,
    working directory, every line it received). `extra_policy_sources` adds
    test-only policy modules to the child's copy of the package.
    """
    if mode not in MODES:
        raise ValueError("mode must be 'closed' or 'train'")
    episode_dir = str(episode_dir)
    adapter = _judge_adapter()
    options = dict(policy_options or {})
    if policy_name in JUDGE_SIDE_OPTIONS:
        options = {**JUDGE_SIDE_OPTIONS[policy_name](episode_dir), **options}
    team_config: dict = {}
    limit = DEFAULT_TIME_LIMIT_S
    if time_limit_s is not None:
        team_config["time_limit_s"] = time_limit_s
        limit = float(time_limit_s)

    with PolicyProcess(policy_name, policy_seed, options, debug=debug, extra_sources=extra_policy_sources) as child:
        failure: str | None = None
        try:
            child.start()
        except PolicyFailure as exc:
            failure = exc.kind
        env = adapter.make_env()
        started = time.monotonic()
        observation, _ = env.reset(None, {"mode": mode, "episode_dir": episode_dir, "team_config": team_config})
        manifest = observation.get("manifest") or {}
        episode_id = manifest.get("episode_id") if isinstance(manifest.get("episode_id"), str) else ""
        deadline = started + limit + DEADLINE_GRACE_S
        records = [_record(0, None, observation, 0.0, False, False)]
        reward, terminated, truncated, info = 0.0, False, False, {}
        if failure is None:
            child.send({"type": "observation", "observation": observation, "reward": 0.0})
        while failure is None:
            try:
                reply = child.receive(deadline)
            except PolicyFailure as exc:
                failure = exc.kind
                break
            if reply.get("type") != "action" or "action" not in reply:
                child.kill()
                failure = "garbage"
                break
            action = reply["action"]
            observation, reward, terminated, truncated, info = env.step(action)
            records.append(_record(len(records), action, observation, reward, terminated, truncated))
            if terminated or truncated:
                break
            child.send({"type": "observation", "observation": observation, "reward": float(reward)})

        if failure is not None:
            from cookedbooks.env.core import empty_answer

            truncated, terminated = True, False
            findings: Any = empty_answer(manifest)
            steps = env.inspection_steps
        else:
            findings = env.final_findings
            steps = env.inspection_steps
        wall_time = time.monotonic() - started

        if mode == "train" and failure is None and isinstance(info.get("metrics"), dict):
            metrics = dict(info["metrics"])
        else:
            metrics = adapter.score_episode(episode_dir, copy.deepcopy(findings), steps)
        seen_reward = float(reward) if mode == "train" and failure is None else 0.0
        if mode == "train" and failure is not None:
            seen_reward = float(metrics["return"])

        debug_report = None
        if failure is None:
            child.send({"type": "end", "observation": observation, "reward": seen_reward,
                        "terminated": terminated, "truncated": truncated})
            if debug:
                try:
                    debug_report = child.receive(time.monotonic() + 5.0)
                except PolicyFailure:
                    debug_report = None
        env.close()

    return EpisodeRun(
        policy=policy_name, episode_id=episode_id, mode=mode, records=records,
        final_findings=findings, inspection_steps=steps, terminated=terminated, truncated=truncated,
        metrics=metrics, wall_time_s=wall_time, failure=failure, debug=debug_report,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.runner.run",
                                     description="Run one policy on one episode folder in a separate process.")
    parser.add_argument("--policy", required=True, help="module name in cookedbooks/policies")
    parser.add_argument("--episode", required=True, help="episode folder")
    parser.add_argument("--mode", choices=MODES, default="closed")
    parser.add_argument("--seed", type=int, default=0, help="the policy's own seed (default 0)")
    parser.add_argument("--out", help="write the trajectory here as JSON lines")
    args = parser.parse_args(argv)

    run = run_episode(args.policy, args.episode, mode=args.mode, policy_seed=args.seed)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for record in run.records:
                line = {"episode_id": run.episode_id, "policy": run.policy, **record}
                handle.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"policy": run.policy, "episode_id": run.episode_id, "mode": run.mode, **run.metrics,
               "inspection_steps": run.inspection_steps, "terminated": run.terminated,
               "truncated": run.truncated, "failure": run.failure, "wall_time_s": round(run.wall_time_s, 3)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

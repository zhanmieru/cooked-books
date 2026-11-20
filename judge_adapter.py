"""The jury's single entry point into CookedBooks: it imports this module and
calls these three functions (EVALUATION_CONTRACT.md, section 1). Imports are
lazy so that loading this file stays cheap and pulls in nothing else.
"""


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Write a full episode (eight files) for `seed` and `config` into `output_dir`.

    `config` is the team_config; a reset-style wrapper {"team_config": {...}, ...}
    is unwrapped (EVALUATION_CONTRACT.md, section 1).
    """
    from cookedbooks.world.generate import generate_episode as _generate_episode

    team_config = {} if config is None else config
    if isinstance(team_config, dict) and "team_config" in team_config:
        team_config = team_config["team_config"]
        if team_config is None:
            team_config = {}
    _generate_episode(seed, team_config, output_dir)


def make_env():
    """Return a fresh AuditEnv with `reset` and `step`."""
    from cookedbooks.env.core import AuditEnv

    return AuditEnv()


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Score `findings` against the episode's truth.json and return the metrics."""
    from cookedbooks.grader.score import score_episode as _score_episode

    return _score_episode(episode_dir, findings, inspection_steps)

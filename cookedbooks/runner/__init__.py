"""Judge-side runner: episodes, the policy process boundary, trajectories and baseline diagnostics.

- `remote`: starts one policy in a restricted child process (JSON lines over stdin/stdout).
- `run`: `run_episode(...)` drives the environment for that child; CLI `python -m cookedbooks.runner.run`.
- `episodes`: the open train/dev episodes from `episodes/index.json`; CLI regenerates or checks them.
- `trajectories`: builds `trajectories.jsonl` from the baselines on the train episodes.
- `diagnostics`: precision, recall, F1, return, steps and time per policy.

This package is the judge side: it may import the grader. Policies never import it.
"""

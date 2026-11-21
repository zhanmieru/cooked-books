"""Policies: the agents that audit an episode through observations and actions only.

The interface (fixed; every policy module follows it):

    make_policy(seed: int, **options) -> policy
    policy.act(observation: dict) -> {"name": ..., "args": {...}}

`act` is called with every observation in order: first the reset observation
`{"task", "manifest", "policy", "remaining_steps"}`, then each step
observation `{"result", "error", "remaining_steps"}`. It returns the next
action. The runner stops calling it when the episode ends.

Rules for every module in this package:
- deterministic for a given seed; standard library only;
- imports nothing from `cookedbooks` except other modules of
  `cookedbooks.policies` (never the grader, the world generator, the
  verifier, the loader or the environment);
- no file, network or process access: a policy runs in a separate child
  process (`cookedbooks.policies.worker`) whose audit hook refuses all of it.

The worker finds a policy by module name
(`importlib.import_module("cookedbooks.policies." + name)`), so a new policy
is a new module and nothing else. This file stays import-free on purpose.
"""

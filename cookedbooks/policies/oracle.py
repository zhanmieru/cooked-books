"""oracle: submits, as its first action, the answer the judge side handed it.

It computes nothing and never reads truth. The runner builds the perfect
answer on the judge side (`cookedbooks.grader.oracle.oracle_findings`) and
passes it in as `make_policy(seed, answer=...)`, the same way the jury's
oracle builds its answer from the closed truth and sends it with an ordinary
`submit`. Without an answer it submits an empty one.
"""

from __future__ import annotations

import copy
from typing import Any

from cookedbooks.policies.base import Action, Observation, episode_id_of


class Oracle:
    def __init__(self, answer: Any = None) -> None:
        self._answer = copy.deepcopy(answer)

    def act(self, observation: Observation) -> Action:
        answer = self._answer
        if not isinstance(answer, dict):
            answer = {"episode_id": episode_id_of(observation), "findings": []}
        return {"name": "submit", "args": {"findings": copy.deepcopy(answer)}}


def make_policy(seed: int, *, answer: Any = None, **options: Any) -> Oracle:
    """`answer` is the findings object built on the judge side; `seed` is unused."""
    return Oracle(answer)

"""Shared pytest fixtures (ready episodes built fresh in a temporary folder) and Hypothesis profiles.

Hypothesis runs the same examples on every machine by default (profile
"cookedbooks": derandomized, no example database, no deadline, no
reproduction blobs). For a random hunt:

    COOKEDBOOKS_HYPOTHESIS_RANDOM=1 COOKEDBOOKS_HYPOTHESIS_EXAMPLES=5000 \
        .venv/bin/python -m pytest -q tests/unit/test_score.py --hypothesis-seed=7

`--hypothesis-seed` is optional (without it every run draws a new seed, which
pytest prints in the header). A test's own `@settings(max_examples=...)`
overrides the profile's count.
"""

import os

import pytest
from hypothesis import settings

from tests.helpers.factory import EpisodeFixture, build_clean, build_large, build_mini

_COMMON = {"database": None, "deadline": None, "print_blob": False}
settings.register_profile("cookedbooks", derandomize=True, max_examples=200, **_COMMON)
settings.register_profile(
    "cookedbooks-random",
    derandomize=False,
    max_examples=int(os.environ.get("COOKEDBOOKS_HYPOTHESIS_EXAMPLES", "2000")),
    **_COMMON,
)
settings.load_profile(
    "cookedbooks-random" if os.environ.get("COOKEDBOOKS_HYPOTHESIS_RANDOM") == "1" else "cookedbooks"
)


@pytest.fixture
def mini_episode(tmp_path) -> EpisodeFixture:
    return build_mini(tmp_path / "mini")


@pytest.fixture
def clean_episode(tmp_path) -> EpisodeFixture:
    return build_clean(tmp_path / "clean")


@pytest.fixture
def large_episode(tmp_path) -> EpisodeFixture:
    return build_large(tmp_path / "large")

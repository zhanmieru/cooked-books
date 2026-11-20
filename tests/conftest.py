"""Shared pytest fixtures: ready episodes built fresh in a temporary folder."""

import pytest

from tests.helpers.factory import EpisodeFixture, build_clean, build_large, build_mini


@pytest.fixture
def mini_episode(tmp_path) -> EpisodeFixture:
    return build_mini(tmp_path / "mini")


@pytest.fixture
def clean_episode(tmp_path) -> EpisodeFixture:
    return build_clean(tmp_path / "clean")


@pytest.fixture
def large_episode(tmp_path) -> EpisodeFixture:
    return build_large(tmp_path / "large")

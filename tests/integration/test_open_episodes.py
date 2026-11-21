"""The open episodes in episodes/train and episodes/dev, and their index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cookedbooks.runner import episodes as open_episodes
from cookedbooks.verifier import verify_episode

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = open_episodes.load_index()
TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")
VARIANTS = {("unallocated_payment", "missing"), ("unallocated_payment", "partial"), ("fx_error", "wrong_date"),
            ("fx_error", "wrong_pair"), ("cutoff_error", "late"), ("cutoff_error", "early")}
# Seeds the jury-recipe and generator tests use (tests/jury/test_block2_data.py, tests/unit, tests/integration).
TEST_SEEDS = set(range(60)) | {77, 20261122}


def _json(entry: open_episodes.OpenEpisode, name: str) -> dict:
    return json.loads((entry.folder() / name).read_text(encoding="utf-8"))


def _issues(entry) -> list[dict]:
    return _json(entry, "truth.json")["issues"]


def _injections(entry) -> list[dict]:
    return _json(entry, "provenance.json")["injections"]


def test_committed_folders_equal_a_fresh_generation():
    assert open_episodes.check() == []


def test_index_lists_ten_train_and_ten_dev_with_exact_configs():
    data = json.loads(open_episodes.INDEX_PATH.read_text(encoding="utf-8"))
    assert [e["split"] for e in data["episodes"]] == ["train"] * 10 + ["dev"] * 10
    for raw, entry in zip(data["episodes"], ENTRIES):
        assert set(raw) == {"split", "name", "seed", "note", "team_config"}
        assert isinstance(raw["seed"], int) and isinstance(raw["team_config"], dict)
        assert entry.folder().is_dir()
        assert _json(entry, "provenance.json")["seed"] == entry.seed
    assert len({(e.split, e.name) for e in ENTRIES}) == 20


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_open_episode_passes_the_independent_verifier(entry):
    report = verify_episode(entry.folder())
    assert report.ok, report


def test_train_and_dev_seeds_are_disjoint_and_unused_by_tests():
    train = {e.seed for e in ENTRIES if e.split == "train"}
    dev = {e.seed for e in ENTRIES if e.split == "dev"}
    assert len(train) == len(dev) == 10 and not train & dev
    assert not (train | dev) & TEST_SEEDS


def test_train_and_dev_configs_differ_at_least_partly():
    def canon(e):
        return json.dumps(e.team_config, sort_keys=True)

    train = {canon(e) for e in ENTRIES if e.split == "train"}
    dev = {canon(e) for e in ENTRIES if e.split == "dev"}
    assert dev - train


@pytest.mark.parametrize("split", open_episodes.SPLITS)
def test_split_covers_the_space(split):
    entries = [e for e in ENTRIES if e.split == split]
    clean = [e for e in entries if not _issues(e)]
    assert len(clean) >= 2
    look_alikes = [frozenset(d["kind"] for d in _json(e, "provenance.json")["distractors"]) for e in clean]
    assert all(look_alikes) and len(set(look_alikes)) == len(look_alikes)
    assert {i["type"] for e in entries for i in _issues(e)} == set(TYPES)
    assert {i["scenario"] for e in entries for i in _injections(e)} >= {"S1", "S2"}
    assert len({tuple(sorted(e.team_config["currencies"])) for e in entries}) >= 4
    assert len({e.team_config["base_currency"] for e in entries}) >= 2
    assert len({e.team_config["timezone"] for e in entries}) >= 4
    assert len({e.team_config["n_ledger_rows"] for e in entries}) >= 4


def test_both_variants_of_every_type_appear_across_the_splits():
    seen = {(i["kind"], i["variant"]) for e in ENTRIES for i in _injections(e) if i["variant"] is not None}
    assert seen >= VARIANTS


def test_open_episodes_hold_all_eight_files():
    for entry in ENTRIES:
        assert sorted(p.name for p in entry.folder().iterdir()) == sorted(open_episodes.EPISODE_FILES)

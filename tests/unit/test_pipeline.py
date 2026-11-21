"""Verified generation with retries: `cookedbooks.pipeline.generate_verified`."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks import pipeline
from cookedbooks.pipeline import GenerationStats, generate_verified
from cookedbooks.verifier import verify_episode
from cookedbooks.world.generate import generate_episode

MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
CONFIG = {"n_ledger_rows": 200, "n_documents": 40, "errors": {"duplicate_invoice": 1}}


@pytest.fixture(autouse=True)
def fresh_stats(monkeypatch) -> GenerationStats:
    """Each test counts into its own STATS, so fake rejections never reach the report."""
    stats = GenerationStats()
    monkeypatch.setattr(pipeline, "STATS", stats)
    return stats


def reject_first(monkeypatch, k: int, seen: list[int] | None = None) -> None:
    """Make the verifier reject the first `k` attempts (read from provenance.json), then use the real one."""
    def fake(folder):
        attempt = json.loads((Path(folder) / "provenance.json").read_text())["attempt"]
        if seen is not None:
            seen.append(attempt)
        if attempt < k:
            report = types.SimpleNamespace(ok=False, codes=lambda: {"fake_reject", f"fake_{attempt}"})
            return report
        return verify_episode(folder)
    monkeypatch.setattr(pipeline, "verify_episode", fake)


def files_of(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


@pytest.mark.parametrize("k", [0, 1, 3])
def test_accepted_attempt_is_the_first_the_verifier_accepts(tmp_path, monkeypatch, fresh_stats, k):
    seen: list[int] = []
    reject_first(monkeypatch, k, seen)
    result = generate_verified(5, CONFIG, tmp_path / "ep")
    assert result.attempt == k and seen == list(range(k + 1))
    assert [r.attempt for r in result.rejected] == list(range(k))
    assert all(r.codes == (f"fake_{r.attempt}", "fake_reject") for r in result.rejected)
    provenance = json.loads((tmp_path / "ep" / "provenance.json").read_text())
    assert provenance["attempt"] == k
    assert fresh_stats.to_dict() == {
        "episodes": 1, "failed_episodes": 0, "attempts": k + 1, "rejected": k,
        "codes": {"fake_reject": k, **{f"fake_{n}": 1 for n in range(k)}} if k else {},
    }


def test_rejected_attempts_leave_no_stale_files(tmp_path, monkeypatch):
    folder = tmp_path / "ep"
    folder.mkdir()
    (folder / "notes.txt").write_text("not ours")
    reject_first(monkeypatch, 2)
    generate_verified(5, CONFIG, folder)
    direct = tmp_path / "direct"
    generate_episode(5, CONFIG, direct, attempt=2)
    produced = files_of(folder)
    assert produced.pop("notes.txt") == b"not ours"
    assert produced == files_of(direct)


def test_all_attempts_rejected_raises_with_codes(tmp_path, monkeypatch, fresh_stats):
    reject_first(monkeypatch, 99)
    with pytest.raises(RuntimeError) as caught:
        generate_verified(5, CONFIG, tmp_path / "ep", max_attempts=3)
    message = str(caught.value)
    for n in range(3):
        assert f"attempt {n}: fake_{n}, fake_reject" in message
    assert list((tmp_path / "ep").iterdir()) == []
    assert (fresh_stats.attempts, fresh_stats.rejected, fresh_stats.failed_episodes) == (3, 3, 1)


def test_same_seed_and_config_give_the_same_files(tmp_path, monkeypatch):
    reject_first(monkeypatch, 1)
    first = generate_verified(9, CONFIG, tmp_path / "a")
    second = generate_verified(9, CONFIG, tmp_path / "b")
    assert first.attempt == second.attempt == 1
    assert files_of(tmp_path / "a") == files_of(tmp_path / "b")


def test_config_does_not_leak_between_attempts(tmp_path, monkeypatch):
    config = json.loads(json.dumps(CONFIG))
    reject_first(monkeypatch, 1)
    generate_verified(9, config, tmp_path / "ep")
    assert config == CONFIG


@pytest.mark.parametrize("config, error", [
    ({"n_ledger_rows": "many"}, ValueError),
    ({"unknown_key": 1}, ValueError),
])
def test_config_errors_pass_through_without_retries(tmp_path, monkeypatch, fresh_stats, config, error):
    calls: list[int] = []
    monkeypatch.setattr(pipeline, "verify_episode", lambda folder: calls.append(1))
    with pytest.raises(error):
        generate_verified(1, config, tmp_path / "ep")
    assert calls == [] and fresh_stats.attempts == 0


def test_not_implemented_passes_through_without_retries(tmp_path, monkeypatch, fresh_stats):
    import cookedbooks.world.generate as generate_module

    def missing(seed, team_config, output_dir, *, attempt=0):
        calls.append(attempt)
        raise NotImplementedError("no injector is registered for fx_error")

    calls: list[int] = []
    monkeypatch.setattr(generate_module, "generate_episode", missing)
    with pytest.raises(NotImplementedError, match="no injector"):
        generate_verified(1, {}, tmp_path / "ep")
    assert calls == [0] and fresh_stats.attempts == 0


def test_bad_max_attempts(tmp_path):
    with pytest.raises(ValueError):
        generate_verified(1, {}, tmp_path / "ep", max_attempts=0)


def test_judge_adapter_goes_through_the_pipeline(tmp_path, monkeypatch, fresh_stats):
    reject_first(monkeypatch, 2)
    judge_adapter.generate_episode(4, {"mode": "closed", "team_config": CONFIG}, str(tmp_path / "ep"))
    assert json.loads((tmp_path / "ep" / "provenance.json").read_text())["attempt"] == 2
    assert (fresh_stats.episodes, fresh_stats.rejected) == (1, 2)


def test_train_reset_goes_through_the_pipeline(tmp_path, monkeypatch, fresh_stats):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    reject_first(monkeypatch, 1)
    env = judge_adapter.make_env()
    observation, _ = env.reset(4, {"mode": "train", "team_config": CONFIG})
    assert observation["remaining_steps"] == 100
    assert (fresh_stats.episodes, fresh_stats.rejected) == (1, 1)
    env.close()


def test_stats_summary_line():
    stats = GenerationStats()
    stats.record_attempt(("b", "a"))
    stats.record_attempt(None)
    stats.episodes += 1
    assert stats.summary() == ("generator attempts rejected by the verifier: 1 of 2 (1 episodes accepted); "
                               "codes: a x1, b x1")
    stats.reset()
    assert stats.to_dict()["attempts"] == 0


@pytest.mark.parametrize("recipe", [f"D{n:02d}" for n in range(1, 12)])
def test_real_pipeline_needs_no_retries_on_recipes(tmp_path, fresh_stats, recipe):
    config = json.loads((MATRIX / f"{recipe}.json").read_text(encoding="utf-8"))
    for seed in range(5):
        result = generate_verified(seed, config, tmp_path / f"s{seed}")
        assert (result.attempt, result.rejected) == (0, ()), (recipe, seed, result)
    assert fresh_stats.rejected == 0 and fresh_stats.episodes == 5

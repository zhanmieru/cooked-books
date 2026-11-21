"""Known, measured limits of the ID contract (docs/failures.md, F-05).

IDs in findings and in `get_record` are plain strings with no ID space
attached (the case's format). When a relabelled folder gives the same string
to records of different spaces, two things stop working: lookup by business
key, and telling evidence records apart. These tests pin the CURRENT
behaviour so the limits stay reproducible from the repo, and check the
property that keeps them out of our own generated data.

Counts for limit A: `pytest -s tests/integration/test_known_limits.py`
prints "wrong / total" per style.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.episode.loader import load_public
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.verifier import verify_episode
from cookedbooks.verify import transform

MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
LOOKUP_RECIPE = "D11"
LOOKUP_SEEDS = (0, 1, 2)
COLLISION_SEEDS = (0, 1)
BUSINESS_KEYS = (("invoices", "invoice_id"), ("payments", "payment_id"))


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader-log.jsonl"))


def generate(name: str, seed: int, folder: Path) -> Path:
    config = json.loads((MATRIX / f"{name}.json").read_text(encoding="utf-8"))
    judge_adapter.generate_episode(seed, config, str(folder))
    return folder


def business_key_counts(root: Path, style: str, seeds=LOOKUP_SEEDS, name: str = LOOKUP_RECIPE) -> tuple[int, int]:
    """(wrong, total) `get_record(table, business key)` lookups on copies relabelled with `style`."""
    wrong = total = 0
    for seed in seeds:
        src = generate(name, seed, root / f"{name}-{seed}")
        dst = root / f"{name}-{seed}-{style}"
        transform.transform_episode(src, dst, permute=True, relabel=True, seed=seed, style=style)
        public = load_public(dst)
        for table, field in BUSINESS_KEYS:
            for record in public.tables[table]:
                total += 1
                wrong += public.get_record(table, record[field]) != record
    return wrong, total


def test_limit_business_key_lookup_under_cross_space_collision(tmp_path):
    """Limit A (docs/failures.md F-05): `get_record` on invoices and payments tries `doc_id` before the
    business key, so when a relabel gives an invoice_id the same string as another document's doc_id, the
    lookup returns that other document. Without cross-space collisions (`hex`) every lookup is right."""
    counts = {style: business_key_counts(tmp_path / style, style) for style in ("int", "hex")}
    for style, (wrong, total) in counts.items():
        print(f"\nlimit A, {LOOKUP_RECIPE} seeds {LOOKUP_SEEDS}, style {style}: {wrong} / {total} business-key lookups wrong")
    assert counts["hex"][0] == 0
    assert counts["int"][0] > 0

    # One concrete case: the record returned is the one whose doc_id is the string asked for.
    dst = tmp_path / "int" / f"{LOOKUP_RECIPE}-{LOOKUP_SEEDS[0]}-int"
    public = load_public(dst)
    for record in public.tables["invoices"]:
        got = public.get_record("invoices", record["invoice_id"])
        if got != record:
            assert got["doc_id"] == record["invoice_id"] != record["doc_id"]
            break
    else:
        pytest.fail("expected at least one colliding invoice_id in the int copy")
    # get_document by doc_id stays exact: the limit is only in the business-key fallback.
    for table in ("invoices", "payments"):
        for record in public.tables[table]:
            assert public.get_document(record["doc_id"]) == record


def _fx_row_and_impostor(src: Path, root: Path) -> tuple[dict, str, str, dict]:
    """An fx_error finding, its evidence row, a doc that gets the same new string, and that mapping."""
    oracle = oracle_findings(src)
    finding = next(f for f in oracle["findings"] if f["type"] == "fx_error")
    for transform_seed in range(200):  # 7 for D12 seed 0 today
        dst = root / f"copy-{transform_seed}"
        mapping = transform.transform_episode(src, dst, permute=False, relabel=True, seed=transform_seed, style="int")
        row = next(e for e in finding["evidence_ids"] if e in mapping["row_id"])
        impostor = transform.invert(mapping)["doc_id"].get(mapping["row_id"][row])
        if impostor is not None and impostor not in finding["evidence_ids"]:
            return oracle, row, impostor, {"mapping": mapping, "dst": dst}
    pytest.fail("no transform seed made a doc_id collide with the fx_error evidence row")


def test_limit_evidence_strings_carry_no_space(tmp_path):
    """Limit B (docs/failures.md F-05): evidence IDs are matched as plain strings, so a relabel that maps a
    row and a different document to the same string makes them interchangeable as evidence, and an answer
    that is wrong on the original can match on the copy. The oracle and any answer mapped forward with the
    same mapping still score identically on both folders."""
    src = generate("D12", 0, tmp_path / "src")
    oracle, row, impostor, found = _fx_row_and_impostor(src, tmp_path)
    mapping, dst = found["mapping"], found["dst"]

    wrong = copy.deepcopy(oracle)
    finding = next(f for f in wrong["findings"] if f["type"] == "fx_error")
    finding["evidence_ids"] = [impostor if e == row else e for e in finding["evidence_ids"]]
    on_original = judge_adapter.score_episode(str(src), wrong, 0)
    on_copy = judge_adapter.score_episode(str(dst), transform.map_findings(wrong, mapping), 0)
    n = len(oracle["findings"])
    assert (on_original["tp"], on_original["fp"], on_original["fn"]) == (n - 1, 1, 1)  # evidence_insufficient
    assert (on_copy["tp"], on_copy["fp"], on_copy["fn"]) == (n, 0, 0)  # the impostor doc counts as the row

    # Control: the oracle and a forward-mapped answer without the impostor score the same on both.
    assert judge_adapter.score_episode(str(src), oracle, 0) == \
        judge_adapter.score_episode(str(dst), oracle_findings(dst), 0)
    degraded = copy.deepcopy(oracle)
    degraded["findings"][0]["claimed_amount_minor"] += 1
    for steps in (0, 37):
        assert judge_adapter.score_episode(str(src), degraded, steps) == \
            judge_adapter.score_episode(str(dst), transform.map_findings(degraded, mapping), steps)


def test_generated_ids_never_collide_across_spaces(tmp_path):
    """No string is an ID in two spaces (fx keys included) of a generated episode, and no fx key repeats,
    so neither limit of docs/failures.md F-05 can arise in our own data; it needs a colliding relabel."""
    for n in range(1, 21):
        name = f"D{n:02d}"
        for seed in COLLISION_SEEDS:
            src = generate(name, seed, tmp_path / f"{name}-{seed}")
            identity = transform.transform_episode(src, tmp_path / f"{name}-{seed}-ids", permute=False,
                                                   relabel=False, seed=0)
            fx = load_public(src).tables["fx"]
            keys = [(r["date"], r["source_currency"], r["target_currency"]) for r in fx]
            assert len(set(keys)) == len(keys), f"{name} seed {seed}: fx.csv repeats a key"
            spaces = {space: set(table) for space, table in identity.items()}
            spaces["fx"] = {r["id"] for r in fx}
            names = sorted(spaces)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    shared = spaces[a] & spaces[b]
                    assert not shared, f"{name} seed {seed}: {a} and {b} share {sorted(shared)[:3]}"


def test_fx_suffix_depends_on_order_only_for_repeated_keys(tmp_path):
    """A repeated fx key gets `#2` in file order, so reordering fx.csv changes which rate the plain key
    names (docs/failures.md F-05); unique keys do not depend on order. The verifier rejects a folder with
    a repeated fx key, so such a folder is outside what we generate or accept."""
    src = generate("D12", 0, tmp_path / "src")
    unique = {r["id"]: r for r in load_public(src).tables["fx"]}
    transform.transform_episode(src, tmp_path / "permuted", permute=True, relabel=False, seed=3)
    assert {r["id"]: r for r in load_public(tmp_path / "permuted").tables["fx"]} == unique

    header, rows, term = transform.read_csv(src / "fx.csv")
    repeat = list(rows[0])
    repeat[header.index("rate")] = "999.5"
    edited = tmp_path / "edited"
    transform.transform_episode(src, edited, permute=False, relabel=False, seed=0)
    manifest = transform.read_json(edited / "manifest.json")
    manifest["record_counts"]["fx"] += 1  # keep the counts right so only the repeat is wrong
    transform.write_json(edited / "manifest.json", manifest)

    key = "{}:{}:{}".format(*(rows[0][header.index(c)] for c in ("date", "source_currency", "target_currency")))
    rate = rows[0][header.index("rate")]
    transform.write_csv(edited / "fx.csv", header, rows + [repeat], term)  # the repeat last
    first = load_public(edited)
    assert first.get_record("fx", key)["rate"] == rate and first.get_record("fx", key + "#2")["rate"] == "999.5"
    transform.write_csv(edited / "fx.csv", header, [repeat] + rows, term)  # the repeat first
    second = load_public(edited)
    assert second.get_record("fx", key)["rate"] == "999.5" and second.get_record("fx", key + "#2")["rate"] == rate

    report = verify_episode(edited)
    assert not report.ok and report.conditions["formats"] is False
    assert any("repeats" in p.message for p in report.failures("formats")), report.render()

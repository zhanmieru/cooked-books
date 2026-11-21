"""The block 4 transformer, `cookedbooks.verify.transform` (docs/jury-checks.md, section 5)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.episode.loader import load_public
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.verifier import verify_episode
from cookedbooks.verify import transform
from cookedbooks.verify.transform import (
    SPACES, STYLES, AmbiguousIdError, invert, map_findings, restore_findings, restore_issue,
    transform_episode,
)

MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
MINI = Path(__file__).resolve().parents[1] / "fixtures" / "mini"
# Non-ID fields per file; they must never change.
ID_FIELDS = {name: set(fields) for name, fields in transform.RECORD_FIELDS.items()}


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader-log.jsonl"))


@pytest.fixture(scope="module", params=["D11", "D20"])
def episode(request, tmp_path_factory) -> Path:
    """A generated episode with modify and add changes (D11: unallocated payments, D20: scenario S2)."""
    folder = tmp_path_factory.mktemp(request.param) / "episode"
    config = json.loads((MATRIX / f"{request.param}.json").read_text(encoding="utf-8"))
    judge_adapter.generate_episode(3, config, str(folder))
    return folder


def files(folder: Path) -> dict:
    """Every file parsed by value: CSV as raw rows, JSON and JSONL decoded."""
    out = {}
    for name in transform.EPISODE_FILES:
        path = folder / name
        if not path.is_file():
            continue
        if name.endswith(".csv"):
            header, rows, _ = transform.read_csv(path)
            out[name] = [header] + rows
        elif name.endswith(".jsonl"):
            out[name] = transform.read_jsonl(path)
        else:
            out[name] = transform.read_json(path)
    return out


def snapshot(folder: Path) -> dict[str, tuple[str, int]]:
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.iterdir())}


def ledger_dicts(folder: Path) -> list[dict]:
    header, rows, _ = transform.read_csv(folder / "ledger.csv")
    return [dict(zip(header, row)) for row in rows]


def canon(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def all_ids(folder: Path) -> dict[str, set[str]]:
    """Every ID of every space as it appears in the files, collected independently of the transformer."""
    ids: dict[str, set[str]] = {space: set() for space in SPACES}
    for row in ledger_dicts(folder):
        for field in ("row_id", "entity_id", "invoice_id", "payment_id"):
            if row[field] not in ("", "null"):
                ids[field].add(row[field])
    records = [("invoices.jsonl", r) for r in transform.read_jsonl(folder / "invoices.jsonl")]
    records += [("payments.jsonl", r) for r in transform.read_jsonl(folder / "payments.jsonl")]
    provenance = transform.read_json(folder / "provenance.json")
    for injection in provenance["injections"]:
        ids["issue_id"].update(injection["issue_ids"])
        for change in injection["changes"]:
            ids[transform.KEY_SPACE[change["file"]]].add(change["key"])
            records += [(change["file"], change[side]) for side in ("before", "after") if change[side]]
    for file_name, record in records:
        for field, space in transform.RECORD_FIELDS[file_name].items():
            value = record.get(field)
            ids[space].update(v for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
    for issue in transform.read_json(folder / "truth.json")["issues"]:
        ids["issue_id"].add(issue["issue_id"])
        ids["doc_id"].update(issue["affected_ids"])
    for distractor in provenance["distractors"]:
        ids["doc_id"].update(distractor["ids"])
    return ids


# --- bijectivity and consistency -------------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
def test_mapping_is_a_bijection_per_space(episode, tmp_path, style):
    mapping = transform_episode(episode, tmp_path / "out", permute=False, relabel=True, seed=1, style=style)
    expected = all_ids(episode)
    new_ids = all_ids(tmp_path / "out")
    for space in SPACES:
        table = mapping[space]
        assert set(table) == expected[space], space
        assert len(set(table.values())) == len(table), f"{space} is not one-to-one"
        assert new_ids[space] == set(table.values()), f"{space}: the copy holds IDs outside the mapping"
        if style != "shuffle":
            assert not set(table) & set(table.values()), f"{space}: an old ID survived"
        elif len(table) > 1:
            assert all(old != new for old, new in table.items()), f"{space}: shuffle left an ID in place"
    assert invert(invert(mapping)) == mapping


def test_int_style_collides_across_spaces_without_false_links(episode, tmp_path):
    out = tmp_path / "out"
    mapping = transform_episode(episode, out, permute=True, relabel=True, seed=2, style="int")
    tables = load_public(out).tables
    rows = {r["row_id"] for r in tables["ledger"]}
    docs = {r["doc_id"] for r in tables["invoices"] + tables["payments"]}
    invoices = {r["invoice_id"] for r in tables["invoices"]}
    assert "7" in rows and "7" in docs and "7" in invoices, "the int style must collide across spaces"
    assert transform.is_ambiguous(mapping)

    # Every link (ledger -> invoice, ledger -> payment, payment -> invoice) is the mapped original link.
    def links(folder: Path, m: dict | None) -> Counter:
        public = load_public(folder).tables
        by_iid = {r["invoice_id"]: r["doc_id"] for r in public["invoices"]}
        by_pid = {r["payment_id"]: r["doc_id"] for r in public["payments"]}
        found = Counter()
        for row in public["ledger"]:
            found[("row-invoice", row["row_id"], by_iid.get(row["invoice_id"]))] += 1
            found[("row-payment", row["row_id"], by_pid.get(row["payment_id"]))] += 1
        for payment in public["payments"]:
            for ref in payment["invoice_refs"] or []:
                found[("payment-invoice", payment["doc_id"], by_iid.get(ref))] += 1
        if m is None:
            return found
        return Counter({(kind, (m["row_id"] if kind != "payment-invoice" else m["doc_id"])[a],
                         None if b is None else m["doc_id"][b]): n for (kind, a, b), n in found.items()})

    assert links(episode, mapping) == links(out, None)
    assert verify_episode(out).ok, verify_episode(out).render()


@pytest.mark.parametrize("style", ["hex", "awkward", "shuffle"])
@pytest.mark.parametrize("permute", [False, True])
def test_inverse_mapping_restores_the_original_files(episode, tmp_path, style, permute):
    mapping = transform_episode(episode, tmp_path / "out", permute=permute, relabel=True, seed=4, style=style)
    transform_episode(tmp_path / "out", tmp_path / "back", permute=False, relabel=True, seed=0,
                      mapping=invert(mapping))
    original, back = files(episode), files(tmp_path / "back")
    assert set(original) == set(back)
    for name in original:
        if permute and name in transform.TABLE_FILES:
            if name.endswith(".csv"):
                assert original[name][0] == back[name][0]
                assert Counter(map(tuple, original[name][1:])) == Counter(map(tuple, back[name][1:])), name
            else:
                assert Counter(map(canon, original[name])) == Counter(map(canon, back[name])), name
        else:
            assert original[name] == back[name], name


def test_inverse_of_int_style_is_ambiguous_without_context(episode, tmp_path):
    mapping = transform_episode(episode, tmp_path / "out", permute=False, relabel=True, seed=4, style="int")
    answer = oracle_findings(tmp_path / "out")
    fx = [f for f in answer["findings"] if f["type"] == "fx_error"]
    ambiguous = [e for f in answer["findings"] for e in f["evidence_ids"] if e not in f["affected_ids"]
                 and sum(e in mapping[s].values() for s in transform.EVIDENCE_SPACES) > 1]
    if ambiguous:
        with pytest.raises(AmbiguousIdError):
            restore_findings(answer, mapping)
    else:
        assert not fx  # without a row ID in the evidence there is nothing to confuse
    # The forward direction is always exact.
    expected = map_findings(oracle_findings(episode), mapping)
    assert sorted(map(canon, expected["findings"])) == sorted(map(canon, answer["findings"]))


def test_invert_rejects_a_mapping_that_is_not_one_to_one():
    with pytest.raises(ValueError):
        invert({"row_id": {"a": "x", "b": "x"}})


def test_restore_maps_findings_and_issues_back(episode, tmp_path):
    out = tmp_path / "out"
    mapping = transform_episode(episode, out, permute=True, relabel=True, seed=5, style="awkward")
    original = oracle_findings(episode)
    restored = restore_findings(oracle_findings(out), mapping)
    key = lambda f: (f["type"], frozenset(f["affected_ids"]), f["claimed_amount_minor"], f["currency"],
                     frozenset(f["evidence_ids"]))
    assert Counter(map(key, restored["findings"])) == Counter(map(key, original["findings"]))
    src_issues = transform.read_json(episode / "truth.json")["issues"]
    dst_issues = transform.read_json(out / "truth.json")["issues"]
    assert [restore_issue(i, mapping) for i in dst_issues] == src_issues


# --- permutation ---------------------------------------------------------------------------


def test_permutation_keeps_records_and_line_item_order(episode, tmp_path):
    out = tmp_path / "out"
    mapping = transform_episode(episode, out, permute=True, relabel=False, seed=6)
    assert all(old == new for table in mapping.values() for old, new in table.items())
    before, after = files(episode), files(out)
    for name in transform.TABLE_FILES:
        if name.endswith(".csv"):
            assert before[name][0] == after[name][0], f"{name}: header changed"
            rows_before, rows_after = before[name][1:], after[name][1:]
            assert Counter(map(tuple, rows_before)) == Counter(map(tuple, rows_after)), name
        else:
            rows_before, rows_after = before[name], after[name]
            assert Counter(map(canon, rows_before)) == Counter(map(canon, rows_after)), name
        assert rows_before != rows_after, f"{name} was not reordered"
    items = {r["doc_id"]: r["line_items"] for r in before["invoices.jsonl"]}
    assert all(r["line_items"] == items[r["doc_id"]] for r in after["invoices.jsonl"])
    for name in ("manifest.json", "policy.json", "truth.json", "provenance.json"):
        assert before[name] == after[name], f"{name} changed under permutation"


def test_relabel_changes_only_id_fields(episode, tmp_path):
    out = tmp_path / "out"
    mapping = transform_episode(episode, out, permute=False, relabel=True, seed=7, style="hex")
    before, after = files(episode), files(out)
    assert before["fx.csv"] == after["fx.csv"], "fx rows (and so fx keys) must not change"
    assert before["manifest.json"] == after["manifest.json"] and before["policy.json"] == after["policy.json"]

    def strip(name: str, record: dict) -> dict:
        return {k: v for k, v in record.items() if k not in ID_FIELDS[name]}

    for old, new in zip(ledger_dicts(episode), ledger_dicts(out)):
        assert strip("ledger.csv", old) == strip("ledger.csv", new)
        assert new["row_id"] == mapping["row_id"][old["row_id"]]
        for field in ("entity_id", "invoice_id", "payment_id"):
            assert (new[field] == "") == (old[field] == ""), "nulls stay null"
    for name in ("invoices.jsonl", "payments.jsonl"):
        for old, new in zip(before[name], after[name]):
            assert strip(name, old) == strip(name, new)
            assert new["doc_id"] == mapping["doc_id"][old["doc_id"]]
    for old, new in zip(before["invoices.jsonl"], after["invoices.jsonl"]):
        assert (new["po_ref"], new["service_period"]) == (old["po_ref"], old["service_period"])

    truth_before, truth_after = before["truth.json"], after["truth.json"]
    assert truth_before["episode_id"] == truth_after["episode_id"]
    for old, new in zip(truth_before["issues"], truth_after["issues"]):
        same = ("type", "expected_amount_minor", "currency")
        assert {k: old[k] for k in same} == {k: new[k] for k in same}
        assert new["issue_id"] == mapping["issue_id"][old["issue_id"]]
        assert new["affected_ids"] == [mapping["doc_id"][d] for d in old["affected_ids"]]
        for old_set, new_set in zip(old["evidence_sets"], new["evidence_sets"]):
            for old_id, new_id in zip(old_set, new_set):
                if old_id.count(":") == 2:  # an fx key
                    assert new_id == old_id
                else:
                    assert new_id in {mapping[s].get(old_id) for s in transform.EVIDENCE_SPACES}

    prov_before, prov_after = before["provenance.json"], after["provenance.json"]
    for field in ("episode_id", "seed", "attempt", "team_config", "generator_version", "schema_version"):
        assert prov_before[field] == prov_after[field], field
    for old, new in zip(prov_before["injections"], prov_after["injections"]):
        for field in ("kind", "variant", "scenario", "description", "effects"):
            assert old[field] == new[field], field
        assert new["issue_ids"] == [mapping["issue_id"][i] for i in old["issue_ids"]]
        for c_old, c_new in zip(old["changes"], new["changes"]):
            assert (c_old["file"], c_old["op"]) == (c_new["file"], c_new["op"])
            assert c_new["key"] == mapping[transform.KEY_SPACE[c_old["file"]]][c_old["key"]]
            for side in ("before", "after"):
                if c_old[side] is None:
                    assert c_new[side] is None
                else:
                    assert c_new[side] == transform.map_record(c_old["file"], c_old[side], mapping)
    for old, new in zip(prov_before["distractors"], prov_after["distractors"]):
        assert new == {"kind": old["kind"], "ids": [mapping["doc_id"][d] for d in old["ids"]]}


def test_removed_records_in_provenance_are_relabelled(episode, tmp_path):
    """A `remove` change keeps its record only in provenance; its IDs must still be mapped."""
    src = tmp_path / "src"
    transform_episode(episode, src, permute=False, relabel=False, seed=0)
    header, rows, term = transform.read_csv(src / "ledger.csv")
    gone = dict(zip(header, rows.pop(0)))
    transform.write_csv(src / "ledger.csv", header, rows, term)
    provenance = transform.read_json(src / "provenance.json")
    record = {k: (None if v == "" else int(v) if k.endswith("_minor") else v) for k, v in gone.items()}
    provenance["injections"].append({"issue_ids": [], "kind": "test", "variant": None, "scenario": None,
                                     "description": "", "effects": [],
                                     "changes": [{"file": "ledger.csv", "op": "remove", "key": gone["row_id"],
                                                  "before": record, "after": None}]})
    transform.write_json(src / "provenance.json", provenance)

    mapping = transform_episode(src, tmp_path / "out", permute=False, relabel=True, seed=8, style="awkward")
    change = transform.read_json(tmp_path / "out" / "provenance.json")["injections"][-1]["changes"][0]
    assert change["key"] == mapping["row_id"][gone["row_id"]] == change["before"]["row_id"]
    assert change["before"] == transform.map_record("ledger.csv", record, mapping)


# --- safety and determinism --------------------------------------------------------------


@pytest.mark.parametrize("style", STYLES)
def test_source_folder_is_never_modified(episode, tmp_path, style):
    before = snapshot(episode)
    transform_episode(episode, tmp_path / "out", permute=True, relabel=True, seed=9, style=style,
                      mapping_path=tmp_path / "mapping.json")
    assert snapshot(episode) == before


def test_deterministic_for_a_seed(episode, tmp_path):
    def run(name: str, seed: int) -> tuple[dict, dict]:
        mapping = transform_episode(episode, tmp_path / name, permute=True, relabel=True, seed=seed, style="hex")
        return mapping, {p.name: p.read_bytes() for p in (tmp_path / name).iterdir()}

    first, second, other = run("a", 11), run("b", 11), run("c", 12)
    assert first == second
    assert first[0] != other[0] and first[1]["ledger.csv"] != other[1]["ledger.csv"]


def test_refuses_unsafe_destinations(episode, tmp_path):
    with pytest.raises(ValueError):
        transform_episode(episode, episode, permute=True, relabel=False, seed=0)
    with pytest.raises(ValueError):
        transform_episode(episode, episode / "inner", permute=True, relabel=False, seed=0)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x")
    with pytest.raises(ValueError):
        transform_episode(episode, tmp_path / "busy", permute=True, relabel=False, seed=0)
    with pytest.raises(ValueError):
        transform_episode(episode, tmp_path / "out", permute=True, relabel=True, seed=0,
                          mapping_path=tmp_path / "out" / "mapping.json")
    with pytest.raises(ValueError):
        transform_episode(episode, tmp_path / "out2", permute=False, relabel=True, seed=0, style="emoji")
    with pytest.raises(ValueError):
        transform_episode(episode, tmp_path / "out3", permute=False, relabel=True, seed=0,
                          mapping={"row_id": {}})


# --- awkward IDs survive the files -------------------------------------------------------


def test_awkward_ids_round_trip_exactly(episode, tmp_path):
    out = tmp_path / "out"
    mapping = transform_episode(episode, out, permute=True, relabel=True, seed=10, style="awkward")
    new_ids = {space: set(table.values()) for space, table in mapping.items()}
    every = set().union(*new_ids.values())
    assert any(i != i.strip() for i in every), "leading or trailing spaces"
    assert any(" " in i.strip() or "\t" in i for i in every), "inner whitespace"
    assert any(not i.isascii() for i in every), "non-ASCII"
    assert any(i.isdigit() and i.startswith("0") for i in every), "leading zero"
    assert any(i.swapcase() in new_ids["row_id"] and i != i.swapcase() for i in new_ids["row_id"]), \
        "two row IDs that differ only by case"
    assert any("\u00e9" in i for i in every) and any("e\u0301" in i for i in every), "composed vs decomposed"

    public = load_public(out)
    tables = public.tables
    assert {r["row_id"] for r in tables["ledger"]} <= new_ids["row_id"]
    assert {r["doc_id"] for r in tables["invoices"] + tables["payments"]} <= new_ids["doc_id"]
    assert {r["entity_id"] for r in tables["invoices"]} <= new_ids["entity_id"]
    for row in tables["ledger"]:
        assert public.get_record("ledger", row["row_id"]) == row
    for invoice in tables["invoices"]:
        assert public.get_document(invoice["doc_id"]) == invoice
    assert all(all_ids(out)[space] == new_ids[space] for space in SPACES)
    assert verify_episode(out).ok, verify_episode(out).render()


def test_awkward_ids_score_like_the_original(episode, tmp_path):
    out = tmp_path / "out"
    transform_episode(episode, out, permute=True, relabel=True, seed=13, style="awkward")
    for steps in (0, 5):
        assert judge_adapter.score_episode(str(out), oracle_findings(out), steps) == \
            judge_adapter.score_episode(str(episode), oracle_findings(episode), steps)


# --- folders the jury may have trimmed ---------------------------------------------------


def test_folder_without_optional_parts_still_transforms(episode, tmp_path):
    src = tmp_path / "src"
    transform_episode(episode, src, permute=False, relabel=False, seed=0)
    invoices = transform.read_jsonl(src / "invoices.jsonl")
    for record in invoices:
        record.pop("po_ref")
        record.pop("service_period")
    transform.write_jsonl(src / "invoices.jsonl", invoices)
    truth = transform.read_json(src / "truth.json")
    transform.write_json(src / "truth.json", truth["issues"])  # a bare list of issues
    (src / "policy.json").unlink()
    (src / "fx.csv").unlink()
    (src / "provenance.json").unlink()

    out = tmp_path / "out"
    mapping = transform_episode(src, out, permute=True, relabel=True, seed=14, style="int")
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in src.iterdir())
    new_truth = transform.read_json(out / "truth.json")
    assert isinstance(new_truth, list) and len(new_truth) == len(truth["issues"])
    assert [i["issue_id"] for i in new_truth] == [mapping["issue_id"][i["issue_id"]] for i in truth["issues"]]
    public = load_public(out)
    assert all("po_ref" not in r for r in public.tables["invoices"]) and not public.tables["fx"]
    score = judge_adapter.score_episode(str(out), oracle_findings(out), 0)
    assert score == judge_adapter.score_episode(str(src), oracle_findings(src), 0)


def test_mini_fixture_with_colliding_source_ids(tmp_path):
    """The mini fixture reuses one string in several spaces; affected_ids settle which one evidence means."""
    for style in STYLES:
        out = tmp_path / style
        mapping = transform_episode(MINI, out, permute=True, relabel=True, seed=15, style=style)
        truth = transform.read_json(out / "truth.json")["issues"]
        old = transform.read_json(MINI / "truth.json")["issues"]
        assert truth[0]["evidence_sets"] == [[mapping["doc_id"][d] for d in old[0]["evidence_sets"][0]]]
        assert judge_adapter.score_episode(str(out), oracle_findings(out), 0)["f1"] == 1


# --- CLI -----------------------------------------------------------------------------------


def test_cli(episode, tmp_path):
    lines: list[str] = []
    code = transform.main([str(episode), str(tmp_path / "out"), "--permute", "--relabel", "--seed", "3",
                           "--style", "awkward", "--mapping", str(tmp_path / "map.json")], out=lines.append)
    assert code == 0 and "relabelled (awkward)" in lines[0]
    mapping = transform.read_mapping(tmp_path / "map.json")
    assert set(mapping) == set(SPACES)
    again = transform_episode(episode, tmp_path / "again", permute=True, relabel=True, seed=3, style="awkward")
    assert mapping == again
    assert files(tmp_path / "out") == files(tmp_path / "again")

    lines.clear()
    assert transform.main([str(episode), str(tmp_path / "out")], out=lines.append) == 2
    assert lines[0].startswith("transform failed")
    with pytest.raises(SystemExit):
        transform.main([str(episode), str(tmp_path / "x"), "--style", "nope"], out=lines.append)


def test_cli_module_runs(episode, tmp_path):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "cookedbooks.verify.transform", str(episode),
                             str(tmp_path / "out"), "--permute"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "permuted" in result.stdout
    assert files(tmp_path / "out")["truth.json"] == files(episode)["truth.json"]


def test_findings_mapping_leaves_fx_keys_and_unknown_ids(episode, tmp_path):
    mapping = transform_episode(episode, tmp_path / "out", permute=False, relabel=True, seed=16, style="hex")
    findings = {"episode_id": "e", "findings": [
        {"type": "fx_error", "affected_ids": ["nobody"], "claimed_amount_minor": 1, "currency": "KZT",
         "evidence_ids": ["2026-07-01:USD:KZT", "nobody"], "explanation": ""}]}
    assert map_findings(findings, mapping) == findings
    assert map_findings(copy.deepcopy({"episode_id": "e"}), mapping) == {"episode_id": "e"}

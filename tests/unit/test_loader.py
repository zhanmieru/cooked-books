"""Tests for cookedbooks.episode.loader."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cookedbooks.episode.loader import TABLES, PublicEpisode, load_public
from tests.helpers.factory import (
    FIXTURES_DIR, EPISODE_FILES, build_clean, build_large, build_mini, write_episode,
)


def _row(row_id, **extra):
    row = {"row_id": row_id, "timestamp": "2026-08-01T10:00:00+05:00", "account": "CASH",
           "debit_minor": 100, "credit_minor": 0, "currency": "KZT", "entity_id": "c",
           "invoice_id": None, "payment_id": None}
    row.update(extra)
    return row


def _invoice(doc_id, invoice_id, **extra):
    inv = {"doc_id": doc_id, "invoice_id": invoice_id, "entity_id": "c",
           "issued_at": "2026-08-01T10:00:00+05:00", "due_at": "2026-08-31T10:00:00+05:00",
           "currency": "KZT", "total_minor": 100,
           "line_items": [{"description": "x", "quantity": 1, "unit_price_minor": 100, "amount_minor": 100}]}
    inv.update(extra)
    return inv


def _payment(doc_id, payment_id, **extra):
    pay = {"doc_id": doc_id, "payment_id": payment_id, "timestamp": "2026-08-02T10:00:00+05:00",
           "currency": "KZT", "amount_minor": 100, "invoice_refs": None}
    pay.update(extra)
    return pay


def _fx(date, src="USD", tgt="KZT", rate="470.25"):
    return {"date": date, "source_currency": src, "target_currency": tgt, "rate": rate}


# --- the three fixtures ---------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["mini_episode", "clean_episode", "large_episode"])
def test_fixtures_load(fixture_name, request):
    fixture = request.getfixturevalue(fixture_name)
    episode = load_public(fixture.path)
    assert isinstance(episode, PublicEpisode)
    tables = episode.tables
    assert tuple(tables) == TABLES
    assert tuple(r["row_id"] for r in tables["ledger"]) == fixture.row_ids
    assert tuple(r["doc_id"] for r in tables["invoices"]) == fixture.invoice_doc_ids
    assert tuple(r["doc_id"] for r in tables["payments"]) == fixture.payment_doc_ids
    assert tuple(r["id"] for r in tables["fx"]) == fixture.fx_ids
    assert episode.manifest["episode_id"] == fixture.episode_id
    counts = episode.manifest["record_counts"]
    assert counts == {name: len(tables[name]) for name in TABLES}
    assert fixture.u not in episode.existing_ids()


def test_mini_shape(mini_episode):
    episode = load_public(mini_episode.path)
    tables = episode.tables
    assert [len(tables[name]) for name in TABLES] == [5, 2, 1, 1]
    assert episode.manifest["base_currency"] == "KZT"
    assert episode.policy["amount_tolerance_minor"]["duplicate_invoice"] == 0
    assert (mini_episode.d1, mini_episode.d2, mini_episode.d3) == ("7", "kX", "5")
    [issue] = mini_episode.issues
    assert issue["affected_ids"] == [mini_episode.d1, mini_episode.d2]
    assert issue["evidence_sets"] == [[mini_episode.d1, mini_episode.d2]]
    debit = sum(r["debit_minor"] for r in tables["ledger"])
    credit = sum(r["credit_minor"] for r in tables["ledger"])
    assert debit == credit


def test_large_has_three_pages(large_episode):
    rows = load_public(large_episode.path).tables["ledger"]
    assert len(rows) >= 101
    assert len(set(large_episode.row_ids)) == len(large_episode.row_ids)
    assert sum(r["debit_minor"] for r in rows) == sum(r["credit_minor"] for r in rows)


def test_build_large_odd_row_count_balances(tmp_path):
    fixture = build_large(tmp_path / "odd", n_rows=101)
    rows = load_public(fixture.path).tables["ledger"]
    assert len(rows) == 101
    assert sum(r["debit_minor"] for r in rows) == sum(r["credit_minor"] for r in rows)


def test_committed_fixture_folders_match_factory(tmp_path):
    for name, build in (("mini", build_mini), ("clean", build_clean), ("large", build_large)):
        fresh = build(tmp_path / name).path
        for file_name in EPISODE_FILES:
            committed = FIXTURES_DIR / name / file_name
            assert committed.read_bytes() == (fresh / file_name).read_bytes(), (
                f"tests/fixtures/{name}/{file_name} is stale: run python -m tests.helpers.factory")


def test_factory_is_deterministic(tmp_path):
    a = build_mini(tmp_path / "a").path
    b = build_mini(tmp_path / "b").path
    for file_name in EPISODE_FILES:
        assert (a / file_name).read_bytes() == (b / file_name).read_bytes()


# --- leniency --------------------------------------------------------------------


def test_folder_without_our_fields_and_without_policy_loads(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], invoices=[_invoice("d", "i")],
                         payments=[_payment("p", "q")])
    (path / "policy.json").unlink()
    episode = load_public(path)
    assert episode.policy == {}
    invoice = episode.get_document("d")
    assert "po_ref" not in invoice and "service_period" not in invoice


@pytest.mark.parametrize("content", ["{}", "", "  \n"])
def test_empty_policy_loads(tmp_path, content):
    path = write_episode(tmp_path / "e", ledger=[_row("r")])
    (path / "policy.json").write_text(content, encoding="utf-8")
    assert load_public(path).policy == {}


def test_partial_policy_is_kept_as_is(tmp_path):
    path = write_episode(tmp_path / "e", policy={"rounding": "half_up"}, ledger=[_row("r")])
    assert load_public(path).policy == {"rounding": "half_up"}


def test_missing_optional_tables_are_empty(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")])
    for name in ("invoices.jsonl", "payments.jsonl", "fx.csv"):
        (path / name).unlink()
    tables = load_public(path).tables
    assert tables["invoices"] == tables["payments"] == tables["fx"] == ()
    assert len(tables["ledger"]) == 1


def test_extra_columns_and_fields_are_kept(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r", memo="hello")],
                         invoices=[_invoice("d", "i", note={"k": [1, 2]})],
                         fx=[{**_fx("2026-08-01"), "source": "bank"}])
    episode = load_public(path)
    assert episode.get_record("ledger", "r")["memo"] == "hello"
    assert episode.get_document("d")["note"] == {"k": [1, 2]}
    assert episode.get_record("fx", "2026-08-01:USD:KZT")["source"] == "bank"


def test_manifest_counts_are_not_checked(tmp_path):
    manifest = {"episode_id": "x", "record_counts": {"ledger": 999}, "document_count": 7}
    path = write_episode(tmp_path / "e", manifest=manifest, ledger=[_row("r")])
    assert load_public(path).manifest == manifest


def test_five_record_episode_loads(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("1"), _row("2", debit_minor=0, credit_minor=100)],
                         invoices=[_invoice("3", "4")], payments=[_payment("5", "6")],
                         fx=[_fx("2026-08-01")])
    episode = load_public(path)
    assert sum(len(t) for t in episode.tables.values()) == 5
    assert episode.get_record("ledger", "2")["credit_minor"] == 100


def test_header_only_ledger_loads(tmp_path):
    path = write_episode(tmp_path / "e")
    assert load_public(path).tables["ledger"] == ()


def test_bom_and_crlf_are_accepted(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], invoices=[_invoice("d", "i")])
    ledger = (path / "ledger.csv").read_text(encoding="utf-8")
    (path / "ledger.csv").write_bytes(b"\xef\xbb\xbf" + ledger.replace("\n", "\r\n").encode())
    invoices = (path / "invoices.jsonl").read_text(encoding="utf-8")
    (path / "invoices.jsonl").write_bytes(invoices.replace("\n", "\r\n\r\n").encode())
    episode = load_public(path)
    assert episode.get_record("ledger", "r")["debit_minor"] == 100
    assert episode.get_document("d")["invoice_id"] == "i"


# --- value parsing ---------------------------------------------------------------


def test_empty_field_and_literal_null_are_none(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r1", invoice_id=None), _row("r2", invoice_id="null"),
                                                 _row("r3", invoice_id="Null")])
    episode = load_public(path)
    assert episode.get_record("ledger", "r1")["invoice_id"] is None
    assert episode.get_record("ledger", "r2")["invoice_id"] is None
    assert episode.get_record("ledger", "r3")["invoice_id"] == "Null"  # exact word only


def test_money_is_int_and_rate_is_exact_string(mini_episode):
    episode = load_public(mini_episode.path)
    for row in episode.tables["ledger"]:
        assert type(row["debit_minor"]) is int and type(row["credit_minor"]) is int
    assert episode.get_record("fx", "2026-08-03:USD:KZT")["rate"] == "470.25"


def test_rate_keeps_trailing_zeros(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], fx=[_fx("2026-08-01", rate="470.100")])
    assert load_public(path).tables["fx"][0]["rate"] == "470.100"


def test_negative_money_parses(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r", debit_minor=-5)])
    assert load_public(path).get_record("ledger", "r")["debit_minor"] == -5


@pytest.mark.parametrize("bad", ["1.5", "abc", "1e3", " 5", "1_000", "+5", "٣"])
def test_non_integer_ledger_money_is_rejected(tmp_path, bad):
    path = write_episode(tmp_path / "e", ledger=[_row("r", debit_minor=bad)])
    with pytest.raises(ValueError, match="debit_minor"):
        load_public(path)


@pytest.mark.parametrize("bad", [100.0, "100", True, None])
def test_non_integer_document_money(tmp_path, bad):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], invoices=[_invoice("d", "i", total_minor=bad)])
    if bad is None:
        assert load_public(path).get_document("d")["total_minor"] is None
    else:
        with pytest.raises(ValueError, match="total_minor"):
            load_public(path)


def test_fx_ids_and_duplicate_suffixes(tmp_path):
    rows = [_fx("2026-08-01"), _fx("2026-08-02"), _fx("2026-08-01", rate="1"),
            _fx("2026-08-01", src="EUR"), _fx("2026-08-01", rate="2")]
    path = write_episode(tmp_path / "e", ledger=[_row("r")], fx=rows)
    episode = load_public(path)
    assert [r["id"] for r in episode.tables["fx"]] == [
        "2026-08-01:USD:KZT", "2026-08-02:USD:KZT", "2026-08-01:USD:KZT#2",
        "2026-08-01:EUR:KZT", "2026-08-01:USD:KZT#3",
    ]
    assert episode.get_record("fx", "2026-08-01:USD:KZT#3")["rate"] == "2"
    assert episode.get_record("fx", "2026-08-01:USD:KZT")["rate"] == "470.25"
    assert episode.get_record("fx", "2026-08-01:usd:KZT") is None


def test_large_fixture_fx_duplicate(large_episode):
    episode = load_public(large_episode.path)
    assert episode.get_record("fx", "2026-07-01:USD:KZT#2")["rate"] == "471.15"


# --- lookup ------------------------------------------------------------------------


def test_get_record_fallback_for_documents(mini_episode):
    episode = load_public(mini_episode.path)
    # invoices: doc_id first, then invoice_id.
    assert episode.get_record("invoices", "7")["invoice_id"] == "Kx"
    assert episode.get_record("invoices", "Kx")["doc_id"] == "7"
    assert episode.get_record("invoices", "5")["doc_id"] == "kX"  # by invoice_id
    # payments: doc_id first, then payment_id.
    assert episode.get_record("payments", "5")["payment_id"] == "7"
    assert episode.get_record("payments", "7")["doc_id"] == "5"  # by payment_id


def test_doc_id_wins_over_business_key(tmp_path):
    invoices = [_invoice("A", "B"), _invoice("B", "C")]
    path = write_episode(tmp_path / "e", ledger=[_row("r")], invoices=invoices)
    episode = load_public(path)
    assert episode.get_record("invoices", "B")["doc_id"] == "B"
    assert episode.get_record("invoices", "C")["doc_id"] == "B"


def test_no_fallback_for_ledger_and_fx(mini_episode):
    episode = load_public(mini_episode.path)
    assert episode.get_record("ledger", "Kx") is None  # an invoice_id referenced by a row
    assert episode.get_record("ledger", "kX") is None  # a doc_id
    assert episode.get_record("fx", "7") is None


def test_colliding_ids_resolve_per_table(mini_episode):
    episode = load_public(mini_episode.path)
    assert episode.get_record("ledger", "7")["account"] == "AR"
    assert episode.get_record("invoices", "7")["total_minor"] == 10000
    assert "invoice_id" in episode.get_record("invoices", "7")
    assert "payment_id" in episode.get_record("payments", "7")
    assert episode.get_document("7")["invoice_id"] == "Kx"
    assert episode.get_document("5")["payment_id"] == "7"  # doc_id only, not invoice_id "5"


def test_get_document_uses_doc_id_only(mini_episode):
    episode = load_public(mini_episode.path)
    assert episode.get_document("Kx") is None  # invoice_id, not doc_id
    assert episode.get_document("kx") is None  # case differs
    assert episode.get_document("B7") is None  # row_id
    assert episode.get_document(mini_episode.u) is None


def test_ids_are_case_sensitive(mini_episode):
    episode = load_public(mini_episode.path)
    assert episode.get_record("ledger", "B7")["row_id"] == "B7"
    assert episode.get_record("ledger", "b7")["row_id"] == "b7"
    assert episode.get_record("invoices", "kX")["doc_id"] == "kX"
    assert episode.get_record("invoices", "KX") is None


def test_unknown_table_raises_key_error(mini_episode):
    episode = load_public(mini_episode.path)
    for table in ("Ledger", "documents", "", None, ["ledger"]):
        with pytest.raises(KeyError):
            episode.get_record(table, "7")


def test_unknown_or_non_string_id_returns_none(mini_episode):
    episode = load_public(mini_episode.path)
    for bad in (mini_episode.u, "", 7, None, ["7"], {"7": 1}):
        assert episode.get_record("ledger", bad) is None
        assert episode.get_document(bad) is None


def test_ids_with_spaces_are_not_stripped(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row(" r1"), _row("r1 ")],
                         invoices=[_invoice(" d ", "i ")], payments=[_payment("p", " q")])
    episode = load_public(path)
    assert [r["row_id"] for r in episode.tables["ledger"]] == [" r1", "r1 "]
    assert episode.get_record("ledger", "r1") is None
    assert episode.get_record("ledger", " r1")["row_id"] == " r1"
    assert episode.get_document("d") is None
    assert episode.get_document(" d ")["invoice_id"] == "i "
    assert episode.get_record("payments", " q")["doc_id"] == "p"
    assert episode.existing_ids() >= {" r1", "r1 ", " d ", "i ", " q"}
    assert "r1" not in episode.existing_ids()


def test_large_fixture_space_and_case_traps(large_episode):
    episode = load_public(large_episode.path)
    assert episode.get_document("q ")["payment_id"] == "Q"
    assert episode.get_document("q") is None
    assert episode.get_record("invoices", "q")["doc_id"] == "Q"
    assert episode.get_record("payments", "Q")["doc_id"] == "q "


def test_duplicate_id_first_occurrence_wins(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r", account="A"), _row("r", account="B")])
    episode = load_public(path)
    assert episode.get_record("ledger", "r")["account"] == "A"
    assert len(episode.tables["ledger"]) == 2


# --- order -------------------------------------------------------------------------


def _shuffle_lines(path: Path, order: list[int], header: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    head, body = (lines[:1], lines[1:]) if header else ([], lines)
    body = [body[i] for i in order]
    path.write_text("".join(head + body), encoding="utf-8")
    return body


def test_record_order_is_file_order(large_episode):
    path = large_episode.path
    n = len(large_episode.row_ids)
    order = [(i * 53 + 11) % n for i in range(n)]  # 53 is coprime with 120
    assert sorted(order) == list(range(n))
    _shuffle_lines(path / "ledger.csv", order, header=True)
    _shuffle_lines(path / "invoices.jsonl", [2, 0, 1], header=False)
    _shuffle_lines(path / "fx.csv", [2, 1, 0], header=True)
    tables = load_public(path).tables
    assert [r["row_id"] for r in tables["ledger"]] == [large_episode.row_ids[i] for i in order]
    assert [r["doc_id"] for r in tables["invoices"]] == [large_episode.invoice_doc_ids[i] for i in (2, 0, 1)]
    # The duplicate fx key now comes first in the file, so it is the unsuffixed one.
    assert [(r["id"], r["rate"]) for r in tables["fx"]] == [
        ("2026-07-01:USD:KZT", "471.15"), ("2026-07-02:USD:KZT", "470.80"),
        ("2026-07-01:USD:KZT#2", "471.10"),
    ]


# --- existing_ids ------------------------------------------------------------------


def test_existing_ids_mini(mini_episode):
    ids = load_public(mini_episode.path).existing_ids()
    assert isinstance(ids, frozenset)
    assert ids == frozenset({
        "7", "B7", "a", "0042", "b7",  # row_id
        "kX", "5",                     # doc_id ("7" too)
        "Kx",                          # invoice_id ("5" too)
                                       # payment_id "7" is already in
        "2026-08-03:USD:KZT",          # fx key
    })


def test_existing_ids_excludes_references_and_nulls(tmp_path):
    rows = [_row("r", invoice_id="ref-only", payment_id="pay-ref", entity_id="ent")]
    path = write_episode(tmp_path / "e", ledger=rows, invoices=[_invoice("d", None)],
                         payments=[_payment("p", "q", invoice_refs=["ghost"])])
    ids = load_public(path).existing_ids()
    assert ids == frozenset({"r", "d", "p", "q"})


# --- errors ------------------------------------------------------------------------


def test_missing_manifest_is_value_error(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")])
    (path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        load_public(path)


def test_missing_ledger_is_value_error(tmp_path):
    path = write_episode(tmp_path / "e")
    (path / "ledger.csv").unlink()
    with pytest.raises(ValueError, match="ledger.csv"):
        load_public(path)


def test_missing_folder_is_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_public(tmp_path / "nope")


@pytest.mark.parametrize("name,content", [
    ("manifest.json", "{not json"),
    ("manifest.json", "[1, 2]"),
    ("policy.json", "{\"a\": NaN}"),
    ("invoices.jsonl", "{\"doc_id\": \n"),
    ("payments.jsonl", "[]\n"),
])
def test_broken_json_is_value_error(tmp_path, name, content):
    path = write_episode(tmp_path / "e", ledger=[_row("r")])
    (path / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        load_public(path)


def test_missing_required_column_or_field(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], payments=[_payment("p", "q")])
    (path / "fx.csv").write_text("date,source_currency,rate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target_currency"):
        load_public(path)
    (path / "fx.csv").unlink()
    bad = _payment("p", "q")
    del bad["invoice_refs"]
    (path / "payments.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invoice_refs"):
        load_public(path)


def test_non_string_document_id_is_value_error(tmp_path):
    path = write_episode(tmp_path / "e", ledger=[_row("r")], invoices=[_invoice(5, "i")])
    with pytest.raises(ValueError, match="doc_id"):
        load_public(path)


def test_error_messages_do_not_leak_paths(tmp_path):
    path = write_episode(tmp_path / "secret-dir", ledger=[_row("r", debit_minor="x")])
    with pytest.raises(ValueError) as info:
        load_public(path)
    assert "secret-dir" not in str(info.value)


# --- isolation ----------------------------------------------------------------------

_opened: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if _opened is not None and event in {"open", "os.listdir", "os.scandir"} and args:
        _opened.append(os.fsdecode(args[0]) if isinstance(args[0], (str, bytes, os.PathLike)) else str(args[0]))


sys.addaudithook(_audit)


@pytest.fixture
def file_access(monkeypatch):
    """Record every path opened, listed or stat'ed while the test runs."""
    global _opened
    _opened = []
    real_stat = os.stat

    def recording_stat(path, *args, **kwargs):
        if _opened is not None and isinstance(path, (str, bytes, os.PathLike)):
            _opened.append(os.fsdecode(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", recording_stat)
    yield _opened
    _opened = None


def test_truth_and_provenance_are_never_touched(mini_episode, file_access):
    path = mini_episode.path
    (path / "truth.json").write_bytes(b"\x00\xff garbage {{{")
    (path / "provenance.json").unlink()
    (path / "provenance.json").mkdir()
    file_access.clear()  # forget the setup above; record only the loader
    episode = load_public(path)
    episode.get_record("invoices", "7")
    episode.get_document("kX")
    episode.existing_ids()
    touched = {Path(p).name for p in file_access}
    assert "truth.json" not in touched and "provenance.json" not in touched
    assert touched <= {"manifest.json", "policy.json", "ledger.csv", "invoices.jsonl",
                       "payments.jsonl", "fx.csv"}
    assert {"manifest.json", "ledger.csv"} <= touched


def test_loads_when_truth_is_a_directory(mini_episode):
    (mini_episode.path / "truth.json").unlink()
    (mini_episode.path / "truth.json").mkdir()
    assert len(load_public(mini_episode.path).tables["ledger"]) == 5


def test_loader_source_never_names_closed_files():
    import cookedbooks.episode.loader as loader

    source = Path(loader.__file__).read_text(encoding="utf-8")
    code = source.split('"""', 2)[2]  # skip the module docstring, which names them
    assert "truth" not in code and "provenance" not in code


# --- copies ------------------------------------------------------------------------


def test_returned_records_are_copies(mini_episode):
    episode = load_public(mini_episode.path)
    record = episode.get_record("invoices", "7")
    record["total_minor"] = 1
    record["line_items"][0]["amount_minor"] = 1
    episode.get_document("7")["doc_id"] = "zzz"
    tables = episode.tables
    tables["ledger"][0]["row_id"] = "hacked"
    episode.manifest["episode_id"] = "hacked"
    episode.policy["amount_tolerance_minor"]["duplicate_invoice"] = 99
    fresh = episode.get_record("invoices", "7")
    assert fresh["total_minor"] == 10000 and fresh["line_items"][0]["amount_minor"] == 10000
    assert episode.get_document("7")["doc_id"] == "7"
    assert episode.tables["ledger"][0]["row_id"] == "7"
    assert episode.manifest["episode_id"] == "ep-mini"
    assert episode.policy["amount_tolerance_minor"]["duplicate_invoice"] == 0


# --- opaque IDs, property-based ------------------------------------------------------

_id_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1, max_size=12,
).filter(lambda s: s != "null")


@settings(max_examples=60, deadline=None)
@given(row_ids=st.lists(_id_text, min_size=1, max_size=6, unique=True),
       doc_ids=st.lists(_id_text, min_size=2, max_size=2, unique=True))
def test_arbitrary_ids_round_trip(row_ids, doc_ids):
    with tempfile.TemporaryDirectory() as tmp:
        path = write_episode(Path(tmp) / "e", ledger=[_row(r) for r in row_ids],
                             invoices=[_invoice(doc_ids[0], doc_ids[1])])
        episode = load_public(path)
        assert [r["row_id"] for r in episode.tables["ledger"]] == row_ids
        for r in row_ids:
            assert episode.get_record("ledger", r)["row_id"] == r
        assert episode.get_document(doc_ids[0])["invoice_id"] == doc_ids[1]
        assert episode.get_record("invoices", doc_ids[1])["doc_id"] == doc_ids[0]

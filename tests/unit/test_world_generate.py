"""generate_episode: determinism, exact sizes, on-disk formats, IDs, order, adapter and env wiring."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.episode.loader import load_public
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import books_from_folder, check_books, instant, pattern_counts

FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
         "fx.csv", "policy.json", "truth.json", "provenance.json")
MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
RICH = {
    "currencies": ["KZT", "USD", "EUR"],
    "distractors": {"split_payment": 2, "refund": 1, "recurring_amount": 1, "boundary": 3, "advance": 1},
}
ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


def gen(tmp_path: Path, seed: int = 7, config: dict | None = None, name: str = "ep") -> Path:
    folder = tmp_path / name
    generate_episode(seed, {} if config is None else config, folder)
    return folder


def file_bytes(folder: Path) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in FILES}


# --- determinism -----------------------------------------------------------------


def test_same_seed_and_config_give_identical_bytes(tmp_path):
    a = gen(tmp_path, 11, RICH, "a")
    b = gen(tmp_path, 11, json.loads(json.dumps(RICH)), "b")
    assert file_bytes(a) == file_bytes(b)


def test_different_seeds_differ(tmp_path):
    a, b = gen(tmp_path, 1, RICH, "a"), gen(tmp_path, 2, RICH, "b")
    for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "manifest.json"):
        assert (a / name).read_bytes() != (b / name).read_bytes(), name


def test_attempt_changes_the_episode(tmp_path):
    generate_episode(5, RICH, tmp_path / "a", attempt=0)
    generate_episode(5, RICH, tmp_path / "b", attempt=1)
    assert (tmp_path / "a" / "ledger.csv").read_bytes() != (tmp_path / "b" / "ledger.csv").read_bytes()
    assert json.loads((tmp_path / "b" / "provenance.json").read_text())["attempt"] == 1


@pytest.mark.parametrize("seed", [True, 1.0, "1", None])
def test_seed_must_be_an_int(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        generate_episode(seed, {}, tmp_path / "x")


def test_output_dir_is_created(tmp_path):
    folder = tmp_path / "deep" / "er" / "ep"
    generate_episode(3, {}, str(folder))
    assert sorted(p.name for p in folder.iterdir()) == sorted(FILES)


# --- sizes ---------------------------------------------------------------------


@pytest.mark.parametrize("rows, docs, config", [
    (200, 20, {}),
    (200, 20, {"currencies": ["KZT", "USD", "JPY"]}),
    (2000, 200, {"currencies": ["KZT", "USD"]}),
    (2000, 20, {}),
    (200, 99, {"currencies": ["KZT", "EUR"]}),
    (201, 20, {}),
    (2000, 200, {"distractors": RICH["distractors"]}),
    (2000, 200, {"currencies": ["USD", "KWD", "JPY"], "base_currency": "USD"}),
    (200, 20, {"distractors": {"refund": 1, "split_payment": 1, "advance": 1, "boundary": 1,
                               "recurring_amount": 1}}),
])
def test_exact_sizes(tmp_path, rows, docs, config):
    for seed in range(5):
        folder = gen(tmp_path, seed, {**config, "n_ledger_rows": rows, "n_documents": docs}, f"s{seed}")
        tables = load_public(folder).tables
        manifest = json.loads((folder / "manifest.json").read_text())
        assert len(tables["ledger"]) == rows
        assert len(tables["invoices"]) + len(tables["payments"]) == docs
        assert manifest["record_counts"] == {name: len(records) for name, records in tables.items()}
        assert manifest["document_count"] == docs
        assert check_books(books_from_folder(folder)) == []


def test_impossible_sizes_raise_value_error(tmp_path):
    with pytest.raises(ValueError, match="n_documents|row"):
        generate_episode(1, {"n_ledger_rows": 200, "n_documents": 150}, tmp_path / "x")
    with pytest.raises(ValueError):
        generate_episode(1, {"n_documents": 20, "distractors": {"refund": 6}}, tmp_path / "y")


# --- formats -------------------------------------------------------------------


def test_on_disk_formats(tmp_path):
    folder = gen(tmp_path, 4, {**RICH, "timezone": "Asia/Kolkata"})
    manifest = json.loads((folder / "manifest.json").read_text())
    assert list(manifest) == ["schema_version", "episode_id", "base_currency", "period_start", "period_end",
                              "timezone", "utc_offset", "record_counts", "document_count",
                              "currency_exponents", "rounding"]
    assert manifest["utc_offset"] == "+05:30" and manifest["timezone"] == "Asia/Kolkata"
    assert manifest["period_start"] == "2026-07-01T00:00:00+05:30"
    assert manifest["period_end"] == "2026-09-30T23:59:59+05:30"
    assert manifest["rounding"] == {"mode": "half_up", "unit": "minor"}
    assert set(manifest["currency_exponents"]) >= {"KZT", "USD", "EUR"}
    assert len(manifest["currency_exponents"]) == 4  # plus the reference pair currency

    with open(folder / "ledger.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
                       "currency", "entity_id", "invoice_id", "payment_id"]
    for row in rows[1:]:
        assert ISO.fullmatch(row[1]) and re.fullmatch(r"\d+", row[3]) and re.fullmatch(r"\d+", row[4])
        assert "null" not in row and "None" not in row
    assert any(row[6] == "" for row in rows[1:])  # empty field = null

    with open(folder / "fx.csv", newline="", encoding="utf-8") as handle:
        fx = list(csv.reader(handle))
    assert fx[0] == ["date", "source_currency", "target_currency", "rate"]
    for row in fx[1:]:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0]) and re.fullmatch(r"\d+(\.\d+)?", row[3])

    invoice_keys = ["doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor",
                    "line_items", "po_ref", "service_period"]
    for line in (folder / "invoices.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert list(record) == invoice_keys
        assert ISO.fullmatch(record["issued_at"]) and ISO.fullmatch(record["due_at"])
        assert type(record["total_minor"]) is int
        for item in record["line_items"]:
            assert list(item) == ["description", "quantity", "unit_price_minor", "amount_minor"]
            assert all(type(item[k]) is int for k in ("quantity", "unit_price_minor", "amount_minor"))
        assert record["po_ref"] is None or isinstance(record["po_ref"], str)
        assert re.fullmatch(r"\d{4}-\d{2}", record["service_period"])
    payment_lines = (folder / "payments.jsonl").read_text().splitlines()
    assert any('"invoice_refs": null' in line for line in payment_lines)  # explicit null
    for line in payment_lines:
        record = json.loads(line)
        assert list(record) == ["doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"]
        assert ISO.fullmatch(record["timestamp"]) and type(record["amount_minor"]) is int

    policy = json.loads((folder / "policy.json").read_text())
    assert policy["duplicate_key"] == ["entity_id", "po_ref", "service_period", "currency", "total_minor"]
    assert policy["fx_materiality"] == {"min_relative": "0.005", "min_minor": 100}
    assert policy["amount_tolerance_minor"] == {"duplicate_invoice": 0, "unallocated_payment": 0,
                                                "fx_error": 0, "cutoff_error": 0}
    truth = json.loads((folder / "truth.json").read_text())
    assert truth == {"schema_version": "1.0", "episode_id": manifest["episode_id"], "issues": []}
    provenance = json.loads((folder / "provenance.json").read_text())
    assert provenance["seed"] == 4 and provenance["attempt"] == 0 and provenance["injections"] == []
    assert provenance["episode_id"] == manifest["episode_id"] and provenance["generator_version"]
    assert provenance["team_config"]["currencies"] == ["KZT", "USD", "EUR"]
    assert provenance["team_config"]["timezone"] == "Asia/Kolkata"
    assert all(set(d) == {"kind", "ids"} for d in provenance["distractors"])


def test_offsets_written_in_files(tmp_path):
    folder = gen(tmp_path, 2, {"timezone": "America/Bogota", "distractors": {"boundary": 4}})
    text = (folder / "invoices.jsonl").read_text() + (folder / "ledger.csv").read_text()
    assert "-05:00" in text and "+00:00" in text


def test_export_window_is_about_ten_days_wider(tmp_path):
    folder = gen(tmp_path, 9, RICH)
    books = books_from_folder(folder)
    moments = [instant(r["timestamp"]) for r in books.ledger] + [instant(i["issued_at"]) for i in books.invoices]
    day = 86_400
    assert books.start - 11 * day < min(moments) < books.start - 5 * day
    assert books.end + 5 * day < max(moments) <= books.end + 10 * day
    fx_dates = sorted(r["date"] for r in books.fx)
    assert fx_dates[0] < "2026-06-21"  # earlier days so that "previous available" always resolves
    assert all(__import__("datetime").date.fromisoformat(d).weekday() < 5 for d in fx_dates)


def test_reference_pair_only_with_foreign_currency(tmp_path):
    single = books_from_folder(gen(tmp_path, 1, {}, "single"))
    assert single.fx == [] and set(single.exponents) == {"KZT"}
    multi = books_from_folder(gen(tmp_path, 1, {"currencies": ["KZT", "USD"]}, "multi"))
    pairs = {(r["source_currency"], r["target_currency"]) for r in multi.fx}
    assert ("USD", "KZT") in pairs and len(pairs) == 2
    assert all(target == "KZT" for _, target in pairs)
    used = {i["currency"] for i in multi.invoices}
    assert used <= {"KZT", "USD"}


# --- IDs and order -------------------------------------------------------------


def test_ids_unique_within_and_across_spaces(tmp_path):
    for seed in range(5):
        books = books_from_folder(gen(tmp_path, seed, {**RICH, "n_ledger_rows": 2000, "n_documents": 200},
                                      f"s{seed}"))
        spaces = {
            "row": [r["row_id"] for r in books.ledger],
            "doc": [i["doc_id"] for i in books.invoices] + [p["doc_id"] for p in books.payments],
            "invoice": [i["invoice_id"] for i in books.invoices],
            "payment": [p["payment_id"] for p in books.payments],
            "entity": sorted({i["entity_id"] for i in books.invoices}
                             | {r["entity_id"] for r in books.ledger if r["entity_id"]}),
        }
        everything = [value for values in spaces.values() for value in values]
        assert len(everything) == len(set(everything))
        hex_parts = [value.split("-", 1)[1] for value in everything]
        assert len(hex_parts) == len(set(hex_parts))
        # invoices and payments share one doc_id prefix, so the ID does not tell the table
        assert {d.split("-")[0] for d in spaces["doc"]} == {"doc"}


def test_file_order_is_chronological(tmp_path):
    books = books_from_folder(gen(tmp_path, 3, {**RICH, "timezone": "America/Bogota"}))
    for records, field in ((books.ledger, "timestamp"), (books.invoices, "issued_at"),
                           (books.payments, "timestamp")):
        moments = [instant(r[field]) for r in records]
        assert moments == sorted(moments)
    assert [r["date"] for r in books.fx] == sorted(r["date"] for r in books.fx)


# --- environment and adapter ---------------------------------------------------


def run_closed(folder: Path) -> None:
    env = judge_adapter.make_env()
    observation, info = env.reset(0, {"mode": "closed", "episode_dir": str(folder), "team_config": {}})
    assert info == {} and observation["remaining_steps"] == 100
    episode = load_public(folder)
    first_doc = episode.tables["invoices"][0]["doc_id"]
    for action in (
        {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}},
        {"name": "get_document", "args": {"doc_id": first_doc}},
        {"name": "list_records", "args": {"table": "fx", "cursor": None, "limit": 5}},
    ):
        result = env.step(action)
        assert result[0]["error"] is None and result[1] == 0.0
    result = env.step({"name": "submit", "args": {"findings": {
        "episode_id": observation["manifest"]["episode_id"], "findings": []}}})
    assert result[0]["result"] == {"accepted": True} and result[1:4] == (0.0, True, False)


def test_generated_folder_runs_through_env_and_scores_one(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    folder = gen(tmp_path, 8, RICH)
    run_closed(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    metrics = judge_adapter.score_episode(str(folder), {"episode_id": manifest["episode_id"], "findings": []}, 0)
    assert metrics["return"] == 1.0 and (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 0, 0)


def test_adapter_generates_and_unwraps_reset_config(tmp_path):
    judge_adapter.generate_episode(5, RICH, str(tmp_path / "flat"))
    judge_adapter.generate_episode(5, {"mode": "closed", "episode_dir": "x", "team_config": RICH},
                                   str(tmp_path / "wrapped"))
    assert file_bytes(tmp_path / "flat") == file_bytes(tmp_path / "wrapped")
    with pytest.raises(ValueError, match="unknown key"):
        judge_adapter.generate_episode(5, {"mode": "closed"}, str(tmp_path / "bad"))


def test_train_reset_without_episode_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    env = judge_adapter.make_env()
    observation, _ = env.reset(42, {"mode": "train", "team_config": {**RICH, "time_limit_s": 30}})
    assert observation["manifest"]["record_counts"]["ledger"] == 600
    result = env.step({"name": "submit", "args": {"findings": {
        "episode_id": observation["manifest"]["episode_id"], "findings": []}}})
    assert result[1] == 1.0 and result[4]["metrics"]["f1"] == 1
    env.close()


def test_generation_is_fast(tmp_path):
    config = {**RICH, "n_ledger_rows": 2000, "n_documents": 200, "currencies": ["KZT", "USD", "JPY"]}
    started = time.perf_counter()
    gen(tmp_path, 1, config)
    assert time.perf_counter() - started < 1.0


# --- recipes -------------------------------------------------------------------


RECIPE_KINDS = {"D01": set(), "D02": {"split_payment"}, "D03": {"refund"},
                "D04": {"recurring_amount"}, "D05": {"boundary"}}


@pytest.mark.parametrize("recipe", sorted(RECIPE_KINDS))
def test_recipes_generate_clean_with_only_their_pattern(tmp_path, recipe, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    config = json.loads((MATRIX / f"{recipe}.json").read_text())
    for seed in range(10):
        folder = gen(tmp_path, seed, config, f"s{seed}")
        books = books_from_folder(folder)
        assert check_books(books) == []
        counts = pattern_counts(books)
        assert {k for k, v in counts.items() if v} == RECIPE_KINDS[recipe]
        provenance = json.loads((folder / "provenance.json").read_text())
        assert {d["kind"] for d in provenance["distractors"]} == RECIPE_KINDS[recipe]
        assert json.loads((folder / "truth.json").read_text())["issues"] == []
    if recipe == "D05":
        assert "USD" in config["currencies"] and any(i["currency"] == "USD" for i in books.invoices)
    run_closed(folder)


def test_world_never_imports_the_verifier():
    import ast

    root = Path(__file__).resolve().parents[2] / "cookedbooks" / "world"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.startswith("cookedbooks.verif") for m in modules), path

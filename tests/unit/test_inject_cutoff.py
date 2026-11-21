"""The production cutoff_error injectors (late, early), checked on generated files only.

Also checks the clean-world change that keeps an early error from being the
only posting before its invoice: ordinary invoices are pre-posted 1-2 days
now and then, never across a bound.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import (
    Books, books_from_folder, check_books, instant, local_day, parse_offset, revert_tables,
)

SETUPS = {
    "kzt": {"currencies": ["KZT"]},
    "kzt_usd_bogota": {"currencies": ["KZT", "USD"], "timezone": "America/Bogota"},
    "jpy_base_tokyo": {"base_currency": "JPY", "currencies": ["JPY", "KWD"], "timezone": "Asia/Tokyo"},
    "usd_base_kolkata_even": {"base_currency": "USD", "currencies": ["USD", "EUR"], "timezone": "Asia/Kolkata",
                              "rounding": "half_even"},
    "utc_kwd": {"base_currency": "KWD", "currencies": ["KWD", "KZT"], "timezone": "UTC"},
}


def gen(tmp_path: Path, seed: int, config: dict, name: str | None = None) -> Path:
    folder = tmp_path / (name or f"s{seed}")
    generate_episode(seed, config, folder)
    return folder


def load(folder: Path) -> tuple[Books, dict, dict, dict]:
    episode = load_public(folder)
    books = Books(episode.manifest, episode.policy, episode.tables)
    truth = json.loads((folder / "truth.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    return books, truth, provenance, episode.tables


def local_seconds(books: Books, text: str) -> int:
    return (instant(text) + books.offset * 60) % 86_400


def check_cutoff_issue(books: Books, issue: dict, injection: dict) -> dict:
    """The exact truth issue and change set of one cutoff_error; returns facts for the caller."""
    by_doc = {i["doc_id"]: i for i in books.invoices}
    assert issue["type"] == "cutoff_error" and len(issue["affected_ids"]) == 1
    invoice = by_doc[issue["affected_ids"][0]]
    assert issue["expected_amount_minor"] == invoice["total_minor"] and issue["currency"] == invoice["currency"]
    rows = books.posting[invoice["invoice_id"]]
    ar = [r for r in rows if r["account"] == "AR" and r["debit_minor"] > 0]
    revenue = [r for r in rows if r["account"] == "REVENUE"]
    assert len(ar) == 1 and len(revenue) == 1
    assert issue["evidence_sets"] == [[invoice["doc_id"], ar[0]["row_id"]], [invoice["doc_id"], revenue[0]["row_id"]]]
    issued, posted = instant(invoice["issued_at"]), instant(rows[0]["timestamp"])
    variant = injection["variant"]
    if variant == "late":
        assert books.in_period(issued) and posted > books.end
    else:
        assert issued > books.end and books.in_period(posted)
    # the move looks ordinary: 1-2 local days, at business hours, written with the company offset
    delay = (local_day(rows[0]["timestamp"], books.offset) - local_day(invoice["issued_at"], books.offset)).days
    assert delay in ((1, 2) if variant == "late" else (-1, -2))
    assert 8 * 3600 <= local_seconds(books, rows[0]["timestamp"]) < 20 * 3600
    assert parse_offset(rows[0]["timestamp"][-6:]) == books.offset
    # only the posting rows' timestamps changed; the invoice (issued_at) is untouched
    assert {c["key"] for c in injection["changes"]} == {r["row_id"] for r in rows}
    for change in injection["changes"]:
        assert change["file"] == "ledger.csv" and change["op"] == "modify"
        before, after = change["before"], change["after"]
        assert {k for k in before if before[k] != after[k]} == {"timestamp"}
        assert instant(before["timestamp"]) == issued  # clean: posted at the issue instant
    assert injection["effects"] == [{"control": "C6", "currency": invoice["currency"],
                                     "delta_minor": invoice["total_minor"]}]
    # every payment of the invoice still follows its posting
    for payment in books.payments:
        if invoice["invoice_id"] in (payment["invoice_refs"] or []):
            assert instant(payment["timestamp"]) > posted
    return {"invoice": invoice, "delay": delay, "variant": variant,
            "paid": any(invoice["invoice_id"] in (p["invoice_refs"] or []) for p in books.payments)}


def check_episode(folder: Path, n: int) -> list[dict]:
    books, truth, provenance, tables = load(folder)
    issues = {i["issue_id"]: i for i in truth["issues"]}
    facts = []
    for injection in provenance["injections"]:
        if injection["kind"] == "cutoff_error":
            assert len(injection["issue_ids"]) == 1 and injection["scenario"] is None
            facts.append(check_cutoff_issue(books, issues[injection["issue_ids"][0]], injection))
    assert len(facts) == n
    violations = check_books(books)
    assert len(violations) == n and all(v.startswith("C6") for v in violations), violations
    assert check_books(Books(books.manifest, books.policy, revert_tables(tables, provenance))) == []
    return facts


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("variant", ["late", "early"])
def test_variant_truth_changes_and_revert(tmp_path, setup, variant):
    config = {**SETUPS[setup], "errors": {"cutoff_error": {variant: 2}}, "distractors": {"boundary": 3}}
    currencies, paid = Counter(), Counter()
    for seed in range(12):
        folder = gen(tmp_path, seed, config)
        books, _, _, _ = load(folder)
        for fact in check_episode(folder, 2):
            assert fact["variant"] == variant
            currencies["base" if fact["invoice"]["currency"] == books.base else "foreign"] += 1
            paid[fact["paid"]] += 1
    assert paid[True] and paid[False]
    if len(SETUPS[setup]["currencies"]) > 1:
        assert currencies["base"] and currencies["foreign"]


def test_bare_number_mixes_variants(tmp_path):
    seen, delays = Counter(), Counter()
    for seed in range(12):
        for fact in check_episode(gen(tmp_path, seed, {"errors": {"cutoff_error": 3}}), 3):
            seen[fact["variant"]] += 1
            delays[fact["delay"]] += 1
    assert seen["late"] and seen["early"] and sum(seen.values()) == 36
    assert set(delays) == {-2, -1, 1, 2}


def test_clean_world_pre_posts_some_invoices_without_crossing(tmp_path):
    """Ordinary invoices are posted the same instant, 1-2 days later or 1-2 days earlier; never across a bound."""
    delays, earlier, total = Counter(), 0, 0
    for seed in range(10):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "distractors": {"boundary": 3}})
        books = books_from_folder(folder)
        assert check_books(books) == []
        for invoice in books.invoices:
            row = books.posting[invoice["invoice_id"]][0]
            delay = (local_day(row["timestamp"], books.offset) - local_day(invoice["issued_at"], books.offset)).days
            delays[delay] += 1
            earlier += instant(row["timestamp"]) < instant(invoice["issued_at"])
            total += 1
    assert set(delays) == {-2, -1, 0, 1, 2}, delays
    assert total * 5 < earlier * 100 < total * 25  # 5-25 % of invoices are posted before their document date


def test_error_delays_are_ordinary_delays(tmp_path):
    """Late and early errors use the same delays as clean postings; clean postings cover both directions."""
    clean = Counter()
    errors = Counter()
    config = {"currencies": ["KZT", "USD"], "errors": {"cutoff_error": {"late": 2, "early": 2}},
              "distractors": {"boundary": 4}}
    for seed in range(15):
        folder = gen(tmp_path, seed, config)
        books, truth, _, _ = load(folder)
        affected = {d for i in truth["issues"] for d in i["affected_ids"]}
        for invoice in books.invoices:
            row = books.posting[invoice["invoice_id"]][0]
            delay = (local_day(row["timestamp"], books.offset) - local_day(invoice["issued_at"], books.offset)).days
            (errors if invoice["doc_id"] in affected else clean)[delay] += 1
    assert set(errors) <= set(clean) and set(errors) == {-2, -1, 1, 2}


def test_footprint_keeps_exact_sizes(tmp_path):
    for rows, docs in ((200, 30), (2000, 200)):
        config = {"n_ledger_rows": rows, "n_documents": docs, "currencies": ["KZT", "EUR"],
                  "errors": {"cutoff_error": {"late": 3, "early": 3}}}
        for seed in range(3):
            books, truth, _, _ = load(gen(tmp_path, seed, config, f"{rows}_{seed}"))
            assert len(books.ledger) == rows and len(books.invoices) + len(books.payments) == docs
            assert len(truth["issues"]) == 6


def test_cutoff_is_deterministic(tmp_path):
    config = {"currencies": ["KZT", "USD"], "errors": {"cutoff_error": {"late": 2, "early": 2}},
              "distractors": {"boundary": 2, "split_payment": 1}}
    a, b = gen(tmp_path, 9, config, "a"), gen(tmp_path, 9, config, "b")
    for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "truth.json", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_lookalikes_near_the_bound_stay_correct(tmp_path):
    """Boundary look-alikes (delayed or pre-posted without crossing, next-period, midnight) sit beside the errors."""
    config = {"currencies": ["KZT", "USD"], "timezone": "Asia/Tokyo", "n_ledger_rows": 900, "n_documents": 120,
              "errors": {"cutoff_error": {"late": 2, "early": 2}}, "distractors": {"boundary": 8}}
    near_lookalikes = 0
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, _ = load(folder)
        check_episode(folder, 4)
        affected = {d for i in truth["issues"] for d in i["affected_ids"]}
        distractor_docs = {d for item in provenance["distractors"] for d in item["ids"]}
        assert not affected & distractor_docs and len(affected) == 4
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for item in provenance["distractors"]:
            invoice = by_doc[item["ids"][0]]
            near_lookalikes += books.distance(local_day(invoice["issued_at"], books.offset)) <= 2
    assert near_lookalikes >= 40


def test_errors_are_not_fx_errors(tmp_path):
    """A foreign cutoff target keeps the conversion of its issue date (only C6 breaks, see check_episode)."""
    config = {"base_currency": "EUR", "currencies": ["EUR", "JPY", "KZT"], "timezone": "America/Lima",
              "errors": {"cutoff_error": {"late": 3, "early": 3}}}
    foreign = 0
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, truth, _, _ = load(folder)
        check_episode(folder, 6)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            invoice = by_doc[issue["affected_ids"][0]]
            if invoice["currency"] != books.base:
                foreign += 1
                revenue = next(r for r in books.posting[invoice["invoice_id"]] if r["account"] == "REVENUE")
                assert revenue["credit_minor"] == books.correct_base(invoice)[0]
    assert foreign > 0


def test_targets_sit_on_ordinary_customers(tmp_path):
    sizes = Counter()
    for seed in range(15):
        folder = gen(tmp_path, seed, {"errors": {"cutoff_error": 3}})
        books, truth, _, _ = load(folder)
        per_customer = Counter(i["entity_id"] for i in books.invoices)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            sizes[min(per_customer[by_doc[issue["affected_ids"][0]]["entity_id"]], 3)] += 1
    assert sizes[3] > sizes[1] + sizes[2]


def test_issue_days_follow_the_delay_weights(tmp_path):
    """Late errors issue on the last day more often than the day before (two crossing delays vs one)."""
    days = Counter()
    for seed in range(30):
        folder = gen(tmp_path, seed, {"errors": {"cutoff_error": {"late": 2}}})
        books, truth, _, _ = load(folder)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            days[(books.last_day - local_day(by_doc[issue["affected_ids"][0]]["issued_at"], books.offset)).days] += 1
    assert set(days) == {0, 1} and days[0] > days[1]  # every late error is issued in the last two days
    assert sum(days.values()) == 60


@pytest.mark.parametrize("zone", ["America/Bogota", "Asia/Kolkata", "Asia/Tokyo"])
def test_date_line_lookalikes_cross_only_by_utc_date(tmp_path, zone):
    """Look-alikes whose issue and posting straddle the period end by UTC date but not by local date
    stand beside the cut-off errors, so reading timestamps as UTC dates finds look-alikes too."""
    config = {"currencies": ["KZT", "USD"], "timezone": zone, "errors": {"cutoff_error": {"late": 1, "early": 1}},
              "distractors": {"boundary": 10}}
    utc_only = 0
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, _ = load(folder)
        check_episode(folder, 2)
        lookalikes = {d for item in provenance["distractors"] for d in item["ids"]}

        def side(moment: int) -> int:  # -1 before the period, 0 inside, 1 after, by the UTC calendar date
            day = date(1970, 1, 1) + timedelta(days=moment // 86_400)
            return -1 if day < books.first_day else (1 if day > books.last_day else 0)
        for invoice in books.invoices:
            issued = instant(invoice["issued_at"])
            posted = instant(books.posting[invoice["invoice_id"]][0]["timestamp"])
            if side(issued) != side(posted) and books.in_period(issued) == books.in_period(posted):
                assert invoice["doc_id"] in lookalikes
                utc_only += 1
    assert utc_only >= 10

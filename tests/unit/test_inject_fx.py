"""The production fx_error injectors (wrong_date, wrong_pair), checked on generated files only.

Every check reads the public files, truth.json and provenance.json; conversions
and the clean-state checks come from tests/helpers/world_checks.py (written
from DESIGN.md, independent of the generator).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.world.fx import FX_MIN_MINOR, FX_MIN_RELATIVE, MIN_MOVE_BP, is_material
from cookedbooks.world.generate import generate_episode
from tests.helpers.world_checks import (
    Books, check_books, convert, instant, local_day, revert_tables, written_day,
)

SETUPS = {
    "kzt_usd": {"currencies": ["KZT", "USD"]},
    "kzt_usd_eur_tokyo": {"currencies": ["KZT", "USD", "EUR"], "timezone": "Asia/Tokyo"},
    "jpy_base_lima_even": {"base_currency": "JPY", "currencies": ["JPY", "USD"], "timezone": "America/Lima",
                           "rounding": "half_even"},
    "kwd_base_kolkata": {"base_currency": "KWD", "currencies": ["KWD", "EUR", "JPY"], "timezone": "Asia/Kolkata"},
    "usd_base_utc": {"base_currency": "USD", "currencies": ["USD", "KZT"], "timezone": "UTC"},
    "eur_base_bogota_kwd": {"base_currency": "EUR", "currencies": ["EUR", "KWD"], "timezone": "America/Bogota",
                            "rounding": "half_even"},
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


def fx_rows(books: Books) -> dict[tuple[str, str], list[tuple[str, Decimal]]]:
    """(source, target) -> [(date, rate)] in date order."""
    pairs: dict[tuple[str, str], list[tuple[str, Decimal]]] = {}
    for row in books.fx:
        pairs.setdefault((row["source_currency"], row["target_currency"]), []).append(
            (row["date"], Decimal(row["rate"])))
    return {k: sorted(v) for k, v in pairs.items()}


def converted_by(books: Books, invoice: dict, posted: int) -> list[tuple[str, str]]:
    """Every fx row (date, source) whose rate, applied to the invoice total, gives `posted`."""
    exps = books.exponents
    return [(row["date"], row["source_currency"]) for row in books.fx
            if row["target_currency"] == books.base
            and convert(invoice["total_minor"], Decimal(row["rate"]), exps[invoice["currency"]], exps[books.base],
                        books.policy["rounding"]) == posted]


def check_fx_issue(books: Books, issue: dict, changes: list[dict]) -> dict:
    """The exact truth issue of one fx_error and its two row changes; returns facts for the caller."""
    by_doc = {i["doc_id"]: i for i in books.invoices}
    assert issue["type"] == "fx_error" and issue["currency"] == books.base
    assert len(issue["affected_ids"]) == 1
    invoice = by_doc[issue["affected_ids"][0]]
    assert invoice["currency"] != books.base
    rows = books.posting[invoice["invoice_id"]]
    revenue = [r for r in rows if r["account"] == "REVENUE"]
    clearing = [r for r in rows if r["account"] == "FX_CLEARING" and r["currency"] == books.base]
    assert len(rows) == 4 and len(revenue) == 1 and len(clearing) == 1
    posted = revenue[0]["credit_minor"]
    assert clearing[0]["debit_minor"] == posted
    correct, key = books.correct_base(invoice)
    assert issue["expected_amount_minor"] == abs(posted - correct) > 0
    assert issue["evidence_sets"] == [[invoice["doc_id"], revenue[0]["row_id"], key]]
    # material by policy, computed from policy.json
    materiality = books.policy["fx_materiality"]
    amount = issue["expected_amount_minor"]
    assert amount >= materiality["min_minor"] and Decimal(amount) >= Decimal(materiality["min_relative"]) * correct
    # only the two base-currency rows changed, and only in their amount
    mine = [c for c in changes if c["key"] in {revenue[0]["row_id"], clearing[0]["row_id"]}]
    assert len(mine) == 2 and all(c["file"] == "ledger.csv" and c["op"] == "modify" for c in mine)
    for change in mine:
        before, after = change["before"], change["after"]
        side = "credit_minor" if after["account"] == "REVENUE" else "debit_minor"
        assert before[side] == correct and after[side] == posted
        assert {k for k in before if before[k] != after[k]} == {side}
    return {"invoice": invoice, "posted": posted, "correct": correct, "key": key,
            "used": converted_by(books, invoice, posted)}


def check_episode(folder: Path, n_fx: int) -> list[dict]:
    """Every fx injection of the episode is exact; the books break only C3 (n times); revert is clean."""
    books, truth, provenance, tables = load(folder)
    facts = []
    fx_injections = [i for i in provenance["injections"] if i["kind"] == "fx_error"]
    assert len(fx_injections) == n_fx
    issues = {i["issue_id"]: i for i in truth["issues"]}
    for injection in fx_injections:
        assert len(injection["issue_ids"]) == 1 and injection["scenario"] is None
        issue = issues[injection["issue_ids"][0]]
        fact = check_fx_issue(books, issue, injection["changes"])
        assert len(injection["changes"]) == 2
        assert injection["effects"] == [{"control": "C3", "currency": books.base,
                                         "delta_minor": issue["expected_amount_minor"]}]
        fact["variant"] = injection["variant"]
        key_day, source, _ = fact["key"].split(":")
        if fact["variant"] == "wrong_date":
            assert any(s == source and d != key_day for d, s in fact["used"]), fact
        else:
            assert any(s != source and d == key_day for d, s in fact["used"]), fact
        facts.append(fact)
    violations = check_books(books)
    assert len(violations) == n_fx and all(v.startswith("C3") for v in violations), violations
    assert check_books(Books(books.manifest, books.policy, revert_tables(tables, provenance))) == []
    return facts


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("variant", ["wrong_date", "wrong_pair"])
def test_variant_truth_changes_and_revert(tmp_path, setup, variant):
    config = {**SETUPS[setup], "errors": {"fx_error": {variant: 2}}, "distractors": {"boundary": 3}}
    for seed in range(10):
        facts = check_episode(gen(tmp_path, seed, config), 2)
        assert {f["variant"] for f in facts} == {variant}


def test_bare_number_mixes_variants(tmp_path):
    seen = Counter()
    for seed in range(10):
        for fact in check_episode(gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "errors": {"fx_error": 3}}), 3):
            seen[fact["variant"]] += 1
    assert seen["wrong_date"] and seen["wrong_pair"] and sum(seen.values()) == 30


def where(books: Books, invoice: dict) -> str:
    """Where an invoice sits: 'midnight' (written date != local date), 'weekend', 'near' a bound, 'ordinary'."""
    local = local_day(invoice["issued_at"], books.offset)
    if written_day(invoice["issued_at"]) != local:
        return "midnight"
    if local.weekday() >= 5:
        return "weekend"
    return "near" if books.distance(local) <= 2 else "ordinary"


def test_wrong_date_flavours_sit_everywhere(tmp_path):
    """Written-date errors near midnight, neighbour-day and payment-date errors mid-period, weekend errors."""
    places, written_used, payment_used = Counter(), 0, 0
    config = {"currencies": ["KZT", "USD"], "errors": {"fx_error": {"wrong_date": 4}}}
    for seed in range(15):
        folder = gen(tmp_path, seed, config)
        books, _, _, _ = load(folder)
        for fact in check_episode(folder, 4):
            invoice = fact["invoice"]
            place = where(books, invoice)
            places[place] += 1
            used_days = {d for d, s in fact["used"] if s == invoice["currency"]}
            if place == "midnight":
                # converted by the written date: the row on or before the written calendar date
                day = written_day(invoice["issued_at"])
                rows = fx_rows(books)[(invoice["currency"], books.base)]
                assert max(d for d, _ in rows if d <= day.isoformat()) in used_days
                written_used += 1
            if place == "weekend":  # Monday's rate instead of Friday's
                assert any(date.fromisoformat(d).weekday() == 0 for d in used_days)
            payers = [p for p in books.payments if invoice["invoice_id"] in (p["invoice_refs"] or [])]
            if place == "ordinary" and payers:
                pay_day = local_day(payers[0]["timestamp"], books.offset)
                rows = fx_rows(books)[(invoice["currency"], books.base)]
                payment_used += max(d for d, _ in rows if d <= pay_day.isoformat()) in used_days
    assert places["midnight"] and places["ordinary"] and places["weekend"], places
    assert places["ordinary"] >= places["midnight"]
    assert written_used == places["midnight"] and payment_used > 0


def test_wrong_pair_takes_the_closest_material_pair(tmp_path):
    config = {"currencies": ["KZT", "USD", "EUR"], "errors": {"fx_error": {"wrong_pair": 3}}}
    places = Counter()
    for seed in range(12):
        folder = gen(tmp_path, seed, config)
        books, _, _, _ = load(folder)
        exps = books.exponents
        for fact in check_episode(folder, 3):
            invoice, correct = fact["invoice"], fact["correct"]
            places[where(books, invoice)] += 1
            key_day = fact["key"].split(":")[0]
            options = []
            for row in books.fx:
                if row["date"] == key_day and row["source_currency"] != invoice["currency"]:
                    value = convert(invoice["total_minor"], Decimal(row["rate"]), exps[invoice["currency"]],
                                    exps[books.base], books.policy["rounding"])
                    if is_material(value, correct):
                        options.append(abs(value - correct))
            assert abs(fact["posted"] - correct) == min(options)
    assert places["ordinary"] and (places["midnight"] or places["weekend"]), places


def test_materiality_rule_at_the_edge():
    """policy.fx_materiality: both thresholds must hold; exactly 100 and exactly 0.5 % are material."""
    assert (FX_MIN_MINOR, FX_MIN_RELATIVE) == (100, Decimal("0.005"))
    assert is_material(20_100, 20_000) and is_material(19_900, 20_000)  # 100 = 0.5 % of 20 000
    assert not is_material(20_099, 20_000) and not is_material(20_000, 20_000)
    assert is_material(10_100, 10_000) and not is_material(10_099, 10_000)  # 100 minor units bind
    assert is_material(100_500, 100_000) and not is_material(100_499, 100_000)  # 0.5 % binds


def test_policy_states_the_materiality_used(tmp_path):
    folder = gen(tmp_path, 1, {"currencies": ["KZT", "USD"], "errors": {"fx_error": 1}})
    policy = json.loads((folder / "policy.json").read_text())
    assert policy["fx_materiality"] == {"min_relative": "0.005", "min_minor": 100}


def test_errors_are_material_with_margin_over_many_seeds(tmp_path):
    """No error is close to the threshold: every one exceeds 0.6 % of the correct amount and 200 minor units."""
    config = {"currencies": ["KZT", "USD", "JPY"], "errors": {"fx_error": {"wrong_date": 3, "wrong_pair": 2}}}
    for seed in range(20):
        for fact in check_episode(gen(tmp_path, seed, config), 5):
            diff = abs(fact["posted"] - fact["correct"])
            assert diff > 200 and diff * 1000 > 6 * fact["correct"], fact


@pytest.mark.parametrize("setup", ["kzt_usd_eur_tokyo", "jpy_base_lima_even", "kwd_base_kolkata"])
def test_rate_series_moves_at_least_the_minimum(tmp_path, setup):
    """Consecutive published rates of a pair differ by at least MIN_MOVE_BP (so a neighbour day is material)."""
    for seed in range(5):
        books, _, _, _ = load(gen(tmp_path, seed, SETUPS[setup]))
        for (source, target), rows in fx_rows(books).items():
            for (_, a), (_, b) in zip(rows, rows[1:]):
                assert abs(b - a) * 10000 >= a * MIN_MOVE_BP, (source, target, a, b)
            assert len({d for d, _ in rows}) == len(rows)


def test_footprint_keeps_exact_sizes(tmp_path):
    for rows, docs in ((200, 30), (2000, 200)):
        config = {"n_ledger_rows": rows, "n_documents": docs, "currencies": ["KZT", "USD"],
                  "errors": {"fx_error": {"wrong_date": 2, "wrong_pair": 2}}}
        for seed in range(3):
            books, truth, _, _ = load(gen(tmp_path, seed, config, f"{rows}_{seed}"))
            assert len(books.ledger) == rows and len(books.invoices) + len(books.payments) == docs
            assert len(truth["issues"]) == 4


def test_fx_error_is_deterministic(tmp_path):
    config = {"currencies": ["KZT", "USD", "EUR"], "errors": {"fx_error": {"wrong_date": 2, "wrong_pair": 2}},
              "distractors": {"boundary": 3}}
    a, b = gen(tmp_path, 7, config, "a"), gen(tmp_path, 7, config, "b")
    for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv",
                 "policy.json", "truth.json", "provenance.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_needs_a_foreign_currency(tmp_path):
    for config in ({"errors": {"fx_error": 1}}, {"currencies": ["USD"], "base_currency": "USD",
                                                 "errors": {"fx_error": {"wrong_pair": 1}}}):
        with pytest.raises(ValueError, match="foreign currency"):
            generate_episode(1, config, tmp_path / "x")


def test_lookalikes_stay_correct_next_to_fx_errors(tmp_path):
    """Correct near-midnight and weekend invoices exist beside the errors; no issue touches a look-alike."""
    config = {"currencies": ["KZT", "USD"], "n_ledger_rows": 900, "n_documents": 120,
              "errors": {"fx_error": {"wrong_date": 3, "wrong_pair": 2}},
              "distractors": {"boundary": 8, "split_payment": 1, "recurring_amount": 1}}
    lookalike_places = Counter()
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, truth, provenance, _ = load(folder)
        check_episode(folder, 5)
        affected = [d for issue in truth["issues"] for d in issue["affected_ids"]]
        assert len(affected) == len(set(affected)) == 5
        distractor_docs = {d for item in provenance["distractors"] for d in item["ids"]}
        assert not set(affected) & distractor_docs
        changed = {c["key"] for inj in provenance["injections"] for c in inj["changes"]}
        rows_of = {r["row_id"]: r for r in books.ledger}
        by_invoice_id = {i["invoice_id"]: i["doc_id"] for i in books.invoices}
        assert not {by_invoice_id.get(rows_of[k]["invoice_id"]) for k in changed if k in rows_of} & distractor_docs
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for item in provenance["distractors"]:
            if item["kind"] == "boundary":
                invoice = by_doc[item["ids"][0]]
                if invoice["currency"] != books.base:
                    lookalike_places[where(books, invoice)] += 1
    assert lookalike_places["midnight"] and lookalike_places["weekend"], lookalike_places


def test_targets_sit_on_ordinary_customers(tmp_path):
    sizes = Counter()
    for seed in range(15):
        folder = gen(tmp_path, seed, {"currencies": ["KZT", "USD"], "errors": {"fx_error": 3}})
        books, truth, _, _ = load(folder)
        per_customer = Counter(i["entity_id"] for i in books.invoices)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            sizes[min(per_customer[by_doc[issue["affected_ids"][0]]["entity_id"]], 3)] += 1
    assert sizes[3] > sizes[1] + sizes[2]


def test_errors_are_not_also_cutoff_or_duplicates(tmp_path):
    """An fx target posts on the side of its issue date and has no twin key: only C3 breaks (check_episode)."""
    config = {"currencies": ["KZT", "EUR"], "timezone": "America/Bogota",
              "errors": {"fx_error": {"wrong_date": 3, "wrong_pair": 3}}}
    for seed in range(10):
        folder = gen(tmp_path, seed, config)
        books, truth, _, _ = load(folder)
        check_episode(folder, 6)
        by_doc = {i["doc_id"]: i for i in books.invoices}
        for issue in truth["issues"]:
            invoice = by_doc[issue["affected_ids"][0]]
            posted = instant(books.posting[invoice["invoice_id"]][0]["timestamp"])
            assert books.in_period(posted) == books.in_period(instant(invoice["issued_at"]))
            assert abs((local_day(books.posting[invoice["invoice_id"]][0]["timestamp"], books.offset)
                        - local_day(invoice["issued_at"], books.offset))) <= timedelta(days=2)


def test_written_date_errors_and_lookalikes_share_their_places(tmp_path):
    """Near-midnight invoices converted by the written date (errors) and by the local date (look-alikes)
    both sit next to a period bound and on ordinary mid-period weekdays."""
    seen = Counter()
    config = {"currencies": ["KZT", "USD"], "timezone": "Asia/Dubai", "n_ledger_rows": 900, "n_documents": 120,
              "errors": {"fx_error": {"wrong_date": 4}}, "distractors": {"boundary": 10}}
    for seed in range(20):
        folder = gen(tmp_path, seed, config)
        books, truth, _, _ = load(folder)
        affected = {d for i in truth["issues"] for d in i["affected_ids"]}
        for invoice in books.invoices:
            if invoice["currency"] == books.base or where(books, invoice) != "midnight":
                continue
            near = books.distance(local_day(invoice["issued_at"], books.offset)) <= 2
            seen[("error" if invoice["doc_id"] in affected else "lookalike", "near" if near else "mid")] += 1
    assert set(seen) == {(k, p) for k in ("error", "lookalike") for p in ("near", "mid")}, seen

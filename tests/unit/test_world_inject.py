"""The injection pipeline, proved with small injectors registered only inside these tests.

CopyInvoice adds records (a duplicate), DetachPayment modifies and removes
and adds (an unallocated payment). Neither is a production injector.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta

import pytest

import judge_adapter
from cookedbooks.episode.loader import load_public
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.world.generate import generate_episode
from cookedbooks.world.inject import (
    INJECTORS, ApplyContext, Footprint, InjectionOutcome, InjectionPlan, IssueDraft, PlanContext, ReserveSpec,
)
from cookedbooks.world.postings import invoice_posting
from cookedbooks.world.timeutil import DAY
from tests.helpers.world_checks import Books, books_from_folder, check_books, revert_tables


class CopyInvoice:
    """Test-only duplicate: a copy of a reserved invoice with new refs and its own posting."""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        currency = ctx.rng.choice([ctx.base_currency, *ctx.foreign_currencies])
        rows = 2 if currency == ctx.base_currency else 4
        return InjectionPlan(footprint=Footprint(documents=1, ledger_rows=rows),
                             reserve=(ReserveSpec(currency=currency, paid="none"),))

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        world, tl = ctx.world, ctx.world.timeline
        original = world.invoices[ctx.targets[0].invoice]
        issued = tl.regular_invoice_time(ctx.rng)
        copy = dataclasses.replace(original, ref=world.new_ref("I"), issued_utc=issued,
                                   issued_offset=tl.written_offset(ctx.rng, issued),
                                   due_utc=issued + 30 * DAY)
        ctx.recorder.add("invoices", copy)
        ctx.recorder.add_all("ledger", invoice_posting(world, copy, tl.regular_posting_time(ctx.rng, issued)))
        pair = [("invoice", original.ref), ("invoice", copy.ref)]
        return InjectionOutcome(
            issues=[IssueDraft("duplicate_invoice", pair, copy.total_minor, copy.currency, [pair])],
            effects=[{"control": "C5", "currency": copy.currency, "delta_minor": copy.total_minor}],
            description="test copy",
        )


class DetachPayment:
    """Test-only unallocated payment: refs removed, the Cr AR row replaced by Cr CUSTOMER_ADVANCES."""

    def plan(self, ctx: PlanContext) -> InjectionPlan:
        return InjectionPlan(reserve=(ReserveSpec(currency=ctx.base_currency, paid="full"),))

    def apply(self, ctx: ApplyContext) -> InjectionOutcome:
        world = ctx.world
        target = ctx.targets[0]
        payment = world.payments[target.payment]
        ar_row = next(r for r in world.payment_rows(payment.ref) if r.account == "AR")
        ctx.recorder.update("payments", payment.ref, invoice_refs=None)
        ctx.recorder.remove("ledger", ar_row.ref)
        ctx.recorder.add("ledger", dataclasses.replace(ar_row, ref=world.new_ref("L"),
                                                       account="CUSTOMER_ADVANCES", invoice=None))
        pair = [("payment", payment.ref), ("invoice", target.invoice)]
        return InjectionOutcome(
            issues=[IssueDraft("unallocated_payment", pair, payment.amount_minor, payment.currency, [pair])],
            effects=[{"control": "C4", "currency": payment.currency, "delta_minor": payment.amount_minor}],
            description="test detach",
        )


class WrongFootprint(CopyInvoice):
    def plan(self, ctx):
        plan = super().plan(ctx)
        return dataclasses.replace(plan, footprint=Footprint(documents=1, ledger_rows=0))


@pytest.fixture
def test_injectors(monkeypatch):
    monkeypatch.setitem(INJECTORS, ("duplicate_invoice", None), CopyInvoice())
    monkeypatch.setitem(INJECTORS, ("unallocated_payment", "missing"), DetachPayment())
    # the production `partial` injector is hidden so a bare number resolves to the test one
    monkeypatch.delitem(INJECTORS, ("unallocated_payment", "partial"), raising=False)


CONFIG = {
    "n_ledger_rows": 400, "n_documents": 60, "currencies": ["KZT", "USD"],
    "errors": {"duplicate_invoice": 2, "unallocated_payment": 1},
    "distractors": {"refund": 1, "boundary": 2},
}


def test_requested_kind_without_injector_is_not_implemented(tmp_path, monkeypatch):
    """Every kind is registered now; with its injectors hidden, a request fails before any file is written."""
    for key in [("fx_error", "wrong_date"), ("fx_error", "wrong_pair"), ("cutoff_error", "late"), ("S2", None)]:
        monkeypatch.delitem(INJECTORS, key)
    config = {"currencies": ["KZT", "USD"]}
    for extra, name in (({"errors": {"fx_error": 1}}, "fx_error"),
                        ({"errors": {"cutoff_error": {"late": 1}}}, "cutoff_error.late"),
                        ({"scenarios": {"S2": 1}}, "S2")):
        with pytest.raises(NotImplementedError, match=name.replace(".", r"\.")):
            generate_episode(1, {**config, **extra}, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_every_kind_and_variant_is_registered():
    from cookedbooks.world.config import ERROR_VARIANTS, SCENARIOS

    expected = {(k, v) for k, vs in ERROR_VARIANTS.items() for v in (vs or (None,))} | {(s, None) for s in SCENARIOS}
    assert expected <= set(INJECTORS)


def test_injected_episode(tmp_path, test_injectors, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "log.jsonl"))
    positions = []
    for seed in range(20):
        folder = tmp_path / str(seed)
        generate_episode(seed, CONFIG, folder)
        episode = load_public(folder)
        tables = episode.tables
        # exact sizes despite the footprint
        assert len(tables["ledger"]) == 400
        assert len(tables["invoices"]) + len(tables["payments"]) == 60

        truth = json.loads((folder / "truth.json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        assert [i["type"] for i in truth["issues"]] == ["duplicate_invoice"] * 2 + ["unallocated_payment"]
        assert [i["issue_id"] for i in truth["issues"]] == ["ISS-0001", "ISS-0002", "ISS-0003"]
        assert [p["issue_ids"] for p in provenance["injections"]] == [["ISS-0001"], ["ISS-0002"], ["ISS-0003"]]
        assert provenance["injections"][2]["variant"] == "missing"  # a bare number picks a registered variant
        existing = episode.existing_ids()
        for issue in truth["issues"]:
            assert all(i in existing for i in issue["affected_ids"])
            assert all(i in existing for s in issue["evidence_sets"] for i in s)

        # injected books break exactly the controls the injections claim
        violations = check_books(books_from_folder(folder))
        assert sum(v.startswith("C5") for v in violations) == 2
        assert sum(v.startswith("C4") for v in violations) == 1
        assert len(violations) == 3, violations

        # before/after are records in final on-disk form with final IDs
        by_key = {r["row_id"]: r for r in tables["ledger"]}
        by_key.update({r["doc_id"]: r for r in tables["invoices"] + tables["payments"]})
        for injection in provenance["injections"]:
            for change in injection["changes"]:
                if change["op"] in ("add", "modify") and change["key"] in by_key \
                        and change is _last_change_of(provenance, change["key"]):
                    assert change["after"] == by_key[change["key"]]
                if change["op"] == "remove":
                    assert change["key"] not in by_key and change["before"]["row_id"] == change["key"]

        # reverting the recorded changes gives clean books
        reverted = revert_tables(tables, provenance)
        clean = Books(episode.manifest, episode.policy, reverted)
        assert check_books(clean) == []
        assert len(reverted["ledger"]) == 400 - sum(
            1 for inj in provenance["injections"] for c in inj["changes"]
            if c["file"] == "ledger.csv" and c["op"] == "add") + sum(
            1 for inj in provenance["injections"] for c in inj["changes"]
            if c["file"] == "ledger.csv" and c["op"] == "remove")

        # the oracle scores 1.0 at 0 steps
        metrics = judge_adapter.score_episode(str(folder), oracle_findings(folder), 0)
        assert metrics["return"] == 1.0 and metrics["tp"] == 3

        copy_key = provenance["injections"][0]["changes"][0]["key"]
        positions.append([r["doc_id"] for r in tables["invoices"]].index(copy_key) / len(tables["invoices"]))
    assert min(positions) < 0.5 < max(positions)  # injected records are not at the end


def _last_change_of(provenance: dict, key: str) -> dict:
    return [c for inj in provenance["injections"] for c in inj["changes"] if c["key"] == key][-1]


def test_injection_is_deterministic(tmp_path, test_injectors):
    generate_episode(3, CONFIG, tmp_path / "a")
    generate_episode(3, CONFIG, tmp_path / "b")
    for name in ("ledger.csv", "invoices.jsonl", "payments.jsonl", "truth.json", "provenance.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_record_counts_do_not_depend_on_injections(tmp_path, test_injectors):
    for errors in ({}, {"duplicate_invoice": 3}, {"duplicate_invoice": 1, "unallocated_payment": 2}):
        folder = tmp_path / json.dumps(errors).replace('"', "").replace(" ", "")
        generate_episode(1, {"n_ledger_rows": 300, "n_documents": 40, "errors": errors}, folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["record_counts"]["ledger"] == 300 and manifest["document_count"] == 40


def test_footprint_mismatch_is_caught(tmp_path, monkeypatch):
    monkeypatch.setitem(INJECTORS, ("duplicate_invoice", None), WrongFootprint())
    with pytest.raises(RuntimeError, match="footprint"):
        generate_episode(1, {"errors": {"duplicate_invoice": 1}}, tmp_path / "x")


def test_reserved_targets_use_their_own_customer(tmp_path, monkeypatch):
    seen = []

    class Spy(CopyInvoice):
        def apply(self, ctx):
            target = ctx.targets[0]
            others = [i for i in ctx.world.invoices.values() if i.customer == target.customer]
            seen.append(len(others))
            assert target.invoice not in {ref for d in ctx.world.distractors for _, ref in d.docs}
            return super().apply(ctx)

    monkeypatch.setitem(INJECTORS, ("duplicate_invoice", None), Spy())
    generate_episode(2, {"errors": {"duplicate_invoice": 2}, "distractors": {"split_payment": 2}}, tmp_path / "x")
    assert seen == [1, 1]


def test_timings_for_reserved_targets(tmp_path, monkeypatch):
    """Every ReserveSpec timing builds a clean target on the intended side of the period end."""
    for timing in ("regular", "midnight_offset", "weekend", "near_end_inside", "near_end_outside"):
        captured = {}

        class Probe:
            def plan(self, ctx):
                return InjectionPlan(reserve=(ReserveSpec(currency=ctx.foreign_currencies[0], paid="full",
                                                          timing=timing),))

            def apply(self, ctx):
                captured["invoice"] = ctx.world.invoices[ctx.targets[0].invoice]
                captured["tl"] = ctx.world.timeline
                return InjectionOutcome(issues=[])

        monkeypatch.setitem(INJECTORS, ("fx_error", None), Probe())
        folder = tmp_path / timing
        generate_episode(4, {"currencies": ["KZT", "USD"], "errors": {"fx_error": 1}}, folder)
        assert check_books(books_from_folder(folder)) == []
        invoice, tl = captured["invoice"], captured["tl"]
        day = tl.local_date(invoice.issued_utc)
        if timing == "near_end_inside":
            assert day in (tl.last, tl.last - timedelta(days=1))
        if timing == "near_end_outside":
            assert day in (tl.last + timedelta(days=1), tl.last + timedelta(days=2))
        if timing == "weekend":
            assert day.weekday() >= 5
        if timing == "midnight_offset":
            assert invoice.issued_offset != tl.offset


# --- the config salt: configs that mean the same thing give the same books ----------

BOOK_FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json",
              "truth.json")
SAME_MEANING = [
    ({"currencies": ["KZT", "USD"]}, {"currencies": ["KZT", "USD"], "time_limit_s": 30, "config_version": "1"}),
    ({}, {"n_ledger_rows": 600}),
    ({}, {"n_documents": 90, "base_currency": "KZT", "currencies": ["KZT"], "timezone": "Asia/Almaty",
          "rounding": "half_up", "fx_rate_date": "document_date", "period_start": "2026-07-01", "period_months": 3}),
    ({}, {"errors": {"fx_error": 0, "cutoff_error": {}}, "scenarios": {"S1": 0}, "distractors": {"boundary": 0}}),
    ({"currencies": ["KZT", "USD"], "errors": {"fx_error": {"wrong_date": 1}}},
     {"currencies": ["KZT", "USD"], "errors": {"fx_error": {"wrong_date": 1, "wrong_pair": 0}}, "time_limit_s": 60}),
]


def _folder_bytes(folder) -> dict[str, bytes]:
    return {name: (folder / name).read_bytes() for name in BOOK_FILES}


@pytest.mark.parametrize("pair", range(len(SAME_MEANING)))
def test_configs_with_the_same_meaning_give_identical_folders(tmp_path, pair):
    a, b = SAME_MEANING[pair]
    generate_episode(7, a, tmp_path / "a")
    generate_episode(7, b, tmp_path / "b")
    assert _folder_bytes(tmp_path / "a") == _folder_bytes(tmp_path / "b")
    pa, pb = (json.loads((tmp_path / x / "provenance.json").read_text()) for x in ("a", "b"))
    assert {k: v for k, v in pa.items() if k != "team_config"} == {k: v for k, v in pb.items() if k != "team_config"}


BASE = {"currencies": ["KZT", "USD"], "errors": {"duplicate_invoice": 1}, "distractors": {"boundary": 2}}
REAL_CHANGES = {
    "a count": {"errors": {"duplicate_invoice": 2}},
    "a look-alike count": {"distractors": {"boundary": 3}},
    "a currency": {"currencies": ["KZT", "EUR"]},
    "the base currency": {"base_currency": "USD", "currencies": ["USD", "KZT"]},
    "the time zone": {"timezone": "Asia/Tokyo"},
    "the period start": {"period_start": "2026-04-01"},
    "the period length": {"period_months": 2},
    "the ledger size": {"n_ledger_rows": 640},
    "the document count": {"n_documents": 92},
    "the rounding": {"rounding": "half_even"},
}


def test_bare_number_and_explicit_split_stay_distinct():
    from cookedbooks.world.config import parse_config

    from cookedbooks.world.rng import config_salt

    bare = parse_config({"currencies": ["KZT", "USD"], "errors": {"fx_error": 1}}).generator_config
    split = parse_config({"currencies": ["KZT", "USD"], "errors": {"fx_error": {"wrong_date": 1}}}).generator_config
    assert bare != split and config_salt(bare) != config_salt(split)


@pytest.mark.parametrize("change", sorted(REAL_CHANGES))
def test_a_real_setting_changes_the_stream(tmp_path, change):
    """Changing a generator setting changes the salt, so the books are drawn anew."""
    from cookedbooks.world.config import parse_config
    from cookedbooks.world.rng import config_salt

    other = {**BASE, **REAL_CHANGES[change]}
    assert config_salt(parse_config(BASE).generator_config) != config_salt(parse_config(other).generator_config)
    generate_episode(7, BASE, tmp_path / "a")
    generate_episode(7, other, tmp_path / "b")
    assert (tmp_path / "a" / "ledger.csv").read_bytes() != (tmp_path / "b" / "ledger.csv").read_bytes()


def test_same_seed_in_different_recipes_draws_different_numbers(tmp_path):
    """Recipes run at the same seed are independent worlds: the S1 advances share of D16, D17 and D18
    is not one number per seed (it was before the config salt, see the stream docstring in rng.py)."""
    from pathlib import Path

    matrix = Path(__file__).resolve().parents[2] / "configs" / "matrix"
    differing = 0
    for seed in range(6):
        shares = set()
        for recipe in ("D16", "D17", "D18"):
            folder = tmp_path / f"{recipe}_{seed}"
            generate_episode(seed, json.loads((matrix / f"{recipe}.json").read_text()), folder)
            books = books_from_folder(folder)
            truth = json.loads((folder / "truth.json").read_text())
            issue = next(i for i in truth["issues"] if i["type"] == "unallocated_payment")
            payment = next(p for p in books.payments if p["doc_id"] == issue["affected_ids"][0])
            shares.add(issue["expected_amount_minor"] * 100 // payment["amount_minor"])
        differing += len(shares) > 1
    assert differing >= 4

"""verify_episode on hand-injected episodes: truth, provenance, evidence."""

from __future__ import annotations

import copy
import json

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.verifier import Books, config_from, find_issues, revert, verify_episode
from cookedbooks.verifier.evidence import check_evidence
from cookedbooks.verifier.revert import _canon
from tests.helpers.books_builder import (
    Injector, inject_cutoff, inject_fx, inject_missing, inject_partial,
)
from tests.helpers.verifier_cases import lookalike_book


def issue(issue_id, type_, affected, amount, currency, sets):
    return {"issue_id": issue_id, "type": type_, "affected_ids": list(affected),
            "expected_amount_minor": amount, "currency": currency, "evidence_sets": [list(s) for s in sets]}


def effect(control, currency, delta):
    return {"control": control, "currency": currency, "delta_minor": delta}


def assert_ok(path):
    report = verify_episode(path)
    assert report.ok, report.render(50)
    return report


def assert_fails(path, *codes, condition=None):
    report = verify_episode(path)
    assert not report.ok
    for code in codes:
        assert code in report.codes(), report.render(50)
    if condition:
        assert not report.conditions[condition], report.render(50)
    return report


def found(path):
    episode = load_public(path)
    config, _ = config_from(episode.manifest, episode.policy)
    return find_issues(Books(config, {k: list(v) for k, v in episode.tables.items()}))


# --- one episode per error type -------------------------------------------------


def dup_case():
    book, _ = lookalike_book()
    book.invoice("50", "150", "c9", "2026-08-03T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08")
    book.payment("52", "152", "c9", "2026-08-08T10:00:00+05:00", 250000, allocations={"150": 250000})
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    book.invoice("51", "151", "c9", "2026-08-18T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08",
                 inj=inj)
    inj.commit(["ISS-1"], "duplicate_invoice", [effect("C5", "KZT", 250000)])
    truth = [issue("ISS-1", "duplicate_invoice", ["51", "50"], 250000, "KZT", [["50", "51"]])]
    return book, inj, truth, clean


def unallocated_case(variant):
    book, _ = lookalike_book()
    book.invoice("60", "160", "c10", "2026-08-02T10:00:00+05:00", 80000)
    pay = book.payment("61", "161", "c10", "2026-08-09T10:00:00+05:00", 80000, allocations={"160": 80000})
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    if variant == "missing":
        inject_missing(inj, pay)
        amount = 80000
    else:
        amount = inject_partial(inj, pay, "160", keep=50000)
    inj.commit(["U"], "unallocated_payment", [effect("C4", "KZT", amount)])
    truth = [issue("U", "unallocated_payment", ["61", "60"], amount, "KZT", [["61", "60"]])]
    return book, inj, truth, clean


def fx_case(variant):
    book, _ = lookalike_book()
    inv = book.invoice("70", "170", "c11", "2026-08-14T10:00:00+05:00", 100000, currency="USD")
    assert inv.base_amount == 47550000
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    wrong = 47025000 if variant == "wrong_date" else 52000000  # 13 Aug USD rate / 14 Aug EUR rate
    diff = inject_fx(inj, inv, wrong)
    inj.commit(["F"], "fx_error", [effect("C3", "KZT", diff)])
    truth = [issue("F", "fx_error", ["70"], diff, "KZT", [["70", inv.rows["revenue"], "2026-08-14:USD:KZT"]])]
    return book, inj, truth, clean


def cutoff_case(variant, currency="KZT"):
    book, _ = lookalike_book()
    if variant == "late":
        inv = book.invoice("80", "180", "c12", "2026-09-29T10:00:00+05:00", 66000, currency=currency)
        moved = "2026-10-01T10:00:00+05:00"
    else:
        inv = book.invoice("80", "180", "c12", "2026-10-02T10:00:00+05:00", 66000, currency=currency)
        moved = "2026-09-30T12:00:00+05:00"
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    inject_cutoff(inj, inv, moved)
    inj.commit(["K"], "cutoff_error", [effect("C6", currency, 66000)])
    truth = [issue("K", "cutoff_error", ["80"], 66000, currency,
                   [["80", inv.rows["ar"]], ["80", inv.rows["revenue"]]])]
    return book, inj, truth, clean


def s1_case():
    """Foreign invoice at the period end, posted at the UTC date's rate, payment partly allocated."""
    book, _ = lookalike_book()
    inv = book.invoice("90", "190", "c13", "2026-09-30T20:00:00+00:00", 200000, currency="USD")
    assert inv.fx_key == "2026-10-01:USD:KZT" and inv.base_amount == 98000000
    pay = book.payment("91", "191", "c13", "2026-10-05T10:00:00+05:00", 200000, currency="USD",
                       allocations={"190": 200000})
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    diff = inject_fx(inj, inv, 96400000)                  # 30 Sep rate
    moved = inject_partial(inj, pay, "190", keep=150000)
    inj.commit(["S1-fx", "S1-pay"], "S1", [effect("C3", "KZT", diff), effect("C4", "USD", moved)], scenario="S1")
    truth = [
        issue("S1-fx", "fx_error", ["90"], 1600000, "KZT", [["90", inv.rows["revenue"], "2026-10-01:USD:KZT"]]),
        issue("S1-pay", "unallocated_payment", ["90", "91"], 50000, "USD", [["90", "91"]]),
    ]
    return book, inj, truth, clean


def s2_case():
    """Duplicate of a foreign invoice whose copy is posted at another date's rate."""
    book, _ = lookalike_book()
    book.invoice("95", "195", "c14", "2026-08-14T10:00:00+05:00", 100000, currency="USD", po_ref="PO-S2",
                 service_period="2026-08")
    clean = copy.deepcopy(book.tables)
    inj = Injector(book)
    copy_inv = book.invoice("96", "196", "c14", "2026-08-17T10:00:00+05:00", 100000, currency="USD",
                            po_ref="PO-S2", service_period="2026-08", base_amount=47550000, inj=inj)
    inj.commit(["S2-dup", "S2-fx"], "S2", [effect("C5", "USD", 100000), effect("C3", "KZT", 460000)],
               scenario="S2")
    truth = [
        issue("S2-dup", "duplicate_invoice", ["95", "96"], 100000, "USD", [["95", "96"]]),
        issue("S2-fx", "fx_error", ["96"], 48010000 - 47550000, "KZT",
              [["96", copy_inv.rows["revenue"], "2026-08-17:USD:KZT"]]),
    ]
    return book, inj, truth, clean


CASES = {
    "duplicate": dup_case,
    "missing": lambda: unallocated_case("missing"),
    "partial": lambda: unallocated_case("partial"),
    "wrong_date": lambda: fx_case("wrong_date"),
    "wrong_pair": lambda: fx_case("wrong_pair"),
    "late": lambda: cutoff_case("late"),
    "early": lambda: cutoff_case("early"),
    "late_usd": lambda: cutoff_case("late", "USD"),
    "S1": s1_case,
    "S2": s2_case,
}


def canon_tables(tables):
    return {name: sorted(_canon(name, r) for r in tables[name]) for name in ("ledger", "invoices", "payments")}


@pytest.mark.parametrize("name", sorted(CASES))
def test_injected_episode_verifies(tmp_path, name):
    book, inj, truth, clean = CASES[name]()
    path = book.write(tmp_path / name, issues=truth, injector=inj)
    report = assert_ok(path)
    result = found(path)
    assert result.anomalies == []
    expected = sorted((i["type"], sorted(i["affected_ids"]), i["expected_amount_minor"], i["currency"],
                       sorted(sorted(s) for s in i["evidence_sets"])) for i in truth)
    assert sorted((i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"], i["evidence_sets"])
                  for i in result.issues) == expected
    assert len(report.found_issues) == len(truth)
    # Revert gives exactly the clean book.
    episode = load_public(path)
    reverted = revert({k: list(v) for k, v in episode.tables.items()}, inj.provenance())
    assert reverted.ok
    assert canon_tables(reverted.tables) == canon_tables(clean)


@pytest.mark.parametrize("name", ["S1", "S2"])
def test_scenarios_give_two_issues(tmp_path, name):
    book, inj, truth, _ = CASES[name]()
    path = book.write(tmp_path / name, issues=truth, injector=inj)
    assert len(found(path).issues) == 2


def test_s1_split_into_two_injections_also_verifies(tmp_path):
    book, _ = lookalike_book()
    inv = book.invoice("90", "190", "c13", "2026-09-30T20:00:00+00:00", 200000, currency="USD")
    pay = book.payment("91", "191", "c13", "2026-10-05T10:00:00+05:00", 200000, currency="USD",
                       allocations={"190": 200000})
    inj = Injector(book)
    inject_fx(inj, inv, 96400000)
    inj.commit(["a"], "fx_error", [effect("C3", "KZT", 1600000)], scenario="S1")
    inject_partial(inj, pay, "190", keep=150000)
    inj.commit(["b"], "unallocated_payment", [effect("C4", "USD", 50000)], scenario="S1")
    truth = [issue("a", "fx_error", ["90"], 1600000, "KZT", [["90", inv.rows["revenue"], "2026-10-01:USD:KZT"]]),
             issue("b", "unallocated_payment", ["90", "91"], 50000, "USD", [["90", "91"]])]
    assert_ok(book.write(tmp_path / "ep", issues=truth, injector=inj))


def test_all_types_in_one_episode(tmp_path):
    book, _ = lookalike_book()
    book.invoice("50", "150", "c9", "2026-08-03T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08")
    book.invoice("60", "160", "c10", "2026-08-02T10:00:00+05:00", 80000)
    pay = book.payment("61", "161", "c10", "2026-08-09T10:00:00+05:00", 80000, allocations={"160": 80000})
    fx_inv = book.invoice("70", "170", "c11", "2026-08-14T10:00:00+05:00", 100000, currency="USD")
    cut = book.invoice("80", "180", "c12", "2026-09-29T10:00:00+05:00", 66000)
    inj = Injector(book)
    inject_cutoff(inj, cut, "2026-10-01T10:00:00+05:00")
    inj.commit(["4"], "cutoff_error", [effect("C6", "KZT", 66000)])
    book.invoice("51", "151", "c9", "2026-08-18T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08",
                 inj=inj)
    inj.commit(["1"], "duplicate_invoice", [effect("C5", "KZT", 250000)])
    inject_fx(inj, fx_inv, 47025000)
    inj.commit(["3"], "fx_error", [effect("C3", "KZT", 525000)])
    inject_missing(inj, pay)
    inj.commit(["2"], "unallocated_payment", [effect("C4", "KZT", 80000)])
    truth = [
        issue("1", "duplicate_invoice", ["50", "51"], 250000, "KZT", [["50", "51"]]),
        issue("2", "unallocated_payment", ["60", "61"], 80000, "KZT", [["60", "61"]]),
        issue("3", "fx_error", ["70"], 525000, "KZT", [["70", fx_inv.rows["revenue"], "2026-08-14:USD:KZT"]]),
        issue("4", "cutoff_error", ["80"], 66000, "KZT", [["80", cut.rows["ar"]], ["80", cut.rows["revenue"]]]),
    ]
    path = book.write(tmp_path / "ep", issues=truth, injector=inj)
    assert_ok(path)
    # Row order and file order do not matter.
    for table in book.tables.values():
        table.reverse()
    assert_ok(book.write(tmp_path / "reversed", issues=list(reversed(truth)), injector=inj))


def test_clean_lookalike_episode_ok(tmp_path):
    book, _ = lookalike_book()
    report = assert_ok(book.write(tmp_path / "clean"))
    assert report.found_issues == [] and report.truth_issue_count == 0 and report.injection_count == 0


# --- truth that disagrees with the books -----------------------------------------


@pytest.mark.parametrize("mutate, code", [
    (lambda t: t[0].update(expected_amount_minor=250001), "wrong_amount"),
    (lambda t: t[0].update(currency="kzt"), "wrong_currency"),
    (lambda t: t[0].update(affected_ids=["50", "1"]), "wrong_affected_ids"),
    (lambda t: t[0].update(type="cutoff_error"), "extra_truth_issue"),
    (lambda t: t[0].update(evidence_sets=[["50"]]), "evidence_sets_differ"),
    (lambda t: t[0].update(evidence_sets=[["50", "51", "no-such-id"]]), "evidence_sets_differ"),
    (lambda t: t[0].update(evidence_sets=[["50", "51"], ["50", "51", "1"]]), "evidence_sets_differ"),
    (lambda t: t.append(issue("ISS-2", "fx_error", ["10"], 5, "KZT", [["10"]])), "extra_truth_issue"),
    (lambda t: t.clear(), "missing_truth_issue"),
    (lambda t: t[0].update(expected_amount_minor="250000"), "truth_issue_malformed"),
    (lambda t: t[0].update(expected_amount_minor=True), "truth_issue_malformed"),
    (lambda t: t.append(dict(t[0])), "truth_issue_id_repeats"),
])
def test_truth_disagreement_fails(tmp_path, mutate, code):
    book, inj, truth, _ = dup_case()
    mutate(truth)
    assert_fails(book.write(tmp_path / "ep", issues=truth, injector=inj), code, condition="truth_matches")


def test_insufficient_or_unknown_evidence_fails_evidence_condition(tmp_path):
    book, inj, truth, _ = dup_case()
    truth[0]["evidence_sets"] = [["50"]]
    report = assert_fails(book.write(tmp_path / "a", issues=truth, injector=inj), "evidence_insufficient")
    assert not report.conditions["evidence"]
    truth[0]["evidence_sets"] = [["50", "51", "no-such-id"]]
    report = assert_fails(book.write(tmp_path / "b", issues=truth, injector=inj), "evidence_insufficient")
    assert any("not retrievable" in p.message for p in report.failures("evidence"))


def _sub(path):
    episode = load_public(path)
    config, _ = config_from(episode.manifest, episode.policy)
    return config, {k: list(v) for k, v in episode.tables.items()}, episode.existing_ids()


def test_evidence_check_per_type(tmp_path):
    book, inj, truth, _ = fx_case("wrong_date")
    config, tables, existing = _sub(book.write(tmp_path / "fx", issues=truth, injector=inj))
    issue_ = truth[0]
    rev = issue_["evidence_sets"][0][1]
    assert check_evidence(config, tables, existing, issue_, ["70", rev, "2026-08-14:USD:KZT"]) == []
    assert check_evidence(config, tables, existing, issue_, ["70", rev, "2026-08-14:USD:KZT", "1"]) == []
    assert check_evidence(config, tables, existing, issue_, ["70", rev])          # no rate key
    assert check_evidence(config, tables, existing, issue_, ["70", rev, "2026-08-13:USD:KZT"])  # wrong rate
    assert check_evidence(config, tables, existing, issue_, ["70", "2026-08-14:USD:KZT"])        # no REVENUE row
    assert check_evidence(config, tables, existing, issue_, ["70"] * 11)

    book, inj, truth, _ = cutoff_case("early")
    config, tables, existing = _sub(book.write(tmp_path / "cut", issues=truth, injector=inj))
    for evidence_set in truth[0]["evidence_sets"]:
        assert check_evidence(config, tables, existing, truth[0], evidence_set) == []
    assert check_evidence(config, tables, existing, truth[0], ["80"])

    book, inj, truth, _ = unallocated_case("partial")
    config, tables, existing = _sub(book.write(tmp_path / "pay", issues=truth, injector=inj))
    assert check_evidence(config, tables, existing, truth[0], ["60", "61"]) == []
    assert check_evidence(config, tables, existing, truth[0], ["61"])
    wrong = dict(truth[0], expected_amount_minor=80000)
    assert check_evidence(config, tables, existing, wrong, ["60", "61"])


# --- provenance that does not revert, injections that do not explain ------------------


def _write(tmp_path, book, truth, provenance, name="ep"):
    return book.write(tmp_path / name, issues=truth, provenance=provenance)


def test_remove_whose_key_is_still_present(tmp_path):
    book, inj, truth, _ = dup_case()
    prov = inj.provenance()
    change = prov["injections"][0]["changes"][0]
    change.update(op="remove", before=change["after"], after=None)
    assert_fails(_write(tmp_path, book, truth, prov), "does_not_revert", "revert_failed", condition="clean_books")


def test_modify_whose_after_differs(tmp_path):
    book, inj, truth, _ = fx_case("wrong_date")
    prov = inj.provenance()
    prov["injections"][0]["changes"][1]["after"]["credit_minor"] = 1
    assert_fails(_write(tmp_path, book, truth, prov), "does_not_revert", condition="clean_books")


def test_add_whose_key_is_missing(tmp_path):
    book, inj, truth, _ = dup_case()
    prov = inj.provenance()
    ghost = dict(prov["injections"][0]["changes"][0]["after"], doc_id="ghost")
    prov["injections"][0]["changes"].append({"file": "invoices.jsonl", "op": "add", "key": "ghost",
                                             "before": None, "after": ghost})
    assert_fails(_write(tmp_path, book, truth, prov), "does_not_revert", condition="clean_books")


def test_published_file_edited_after_generation(tmp_path):
    book, inj, truth, _ = fx_case("wrong_date")
    revenue = truth[0]["evidence_sets"][0][1]
    book.record("ledger", revenue)["timestamp"] = "2026-08-14T10:00:01+05:00"
    assert_fails(_write(tmp_path, book, truth, inj.provenance()), "does_not_revert")


def test_bad_change_records(tmp_path):
    book, inj, truth, _ = dup_case()
    for mutate in (
        lambda c: c.update(file="truth.json"),
        lambda c: c.update(op="upsert"),
        lambda c: c.update(key="other"),
        lambda c: c.update(before={"doc_id": "51"}),
    ):
        prov = inj.provenance()
        mutate(prov["injections"][0]["changes"][0])
        assert_fails(_write(tmp_path, book, truth, prov), "does_not_revert")


def test_empty_provenance_but_dirty_books(tmp_path):
    book, inj, truth, _ = dup_case()
    prov = inj.provenance()
    prov["injections"] = []
    report = assert_fails(_write(tmp_path, book, truth, prov), "clean_has_issue", "clean_C5", "clean_mismatch")
    assert not report.conditions["clean_books"] and not report.conditions["clean_episode"]


def test_truth_lists_issue_for_clean_books(tmp_path):
    book, _ = lookalike_book()
    truth = [issue("x", "duplicate_invoice", ["1", "2"], 150000, "KZT", [["1", "2"]])]
    report = assert_fails(book.write(tmp_path / "ep", issues=truth), "extra_truth_issue", "clean_mismatch")
    assert not report.conditions["clean_episode"]


def test_wrong_effects_fail(tmp_path):
    book, inj, truth, _ = dup_case()
    inj.injections[0]["effects"] = [effect("C3", "KZT", 250000)]
    assert_fails(book.write(tmp_path / "a", issues=truth, injector=inj), "injection_effects_mismatch")
    inj.injections[0]["effects"] = [effect("C5", "KZT", 1)]
    assert_fails(book.write(tmp_path / "b", issues=truth, injector=inj), "injection_effects_mismatch")


def test_wrong_kind_fails(tmp_path):
    book, inj, truth, _ = dup_case()
    inj.injections[0]["kind"] = "fx_error"
    assert_fails(book.write(tmp_path / "a", issues=truth, injector=inj), "injection_kind_mismatch")


def test_issue_ids_swapped_between_injections(tmp_path):
    book, _ = lookalike_book()
    book.invoice("50", "150", "c9", "2026-08-03T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08")
    cut = book.invoice("80", "180", "c12", "2026-09-29T10:00:00+05:00", 66000)
    inj = Injector(book)
    book.invoice("51", "151", "c9", "2026-08-18T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-08",
                 inj=inj)
    inj.commit(["K"], "duplicate_invoice", [effect("C5", "KZT", 250000)])
    inject_cutoff(inj, cut, "2026-10-01T10:00:00+05:00")
    inj.commit(["D"], "cutoff_error", [effect("C6", "KZT", 66000)])
    truth = [issue("D", "duplicate_invoice", ["50", "51"], 250000, "KZT", [["50", "51"]]),
             issue("K", "cutoff_error", ["80"], 66000, "KZT", [["80", cut.rows["ar"]], ["80", cut.rows["revenue"]]])]
    report = assert_fails(book.write(tmp_path / "ep", issues=truth, injector=inj), "injection_issue_mismatch")
    assert report.conditions["truth_matches"] and not report.conditions["injections"]


def test_issue_not_claimed_by_any_injection(tmp_path):
    book, inj, truth, _ = dup_case()
    inj.injections[0]["issue_ids"] = ["other"]
    assert_fails(book.write(tmp_path / "ep", issues=truth, injector=inj),
                 "issue_not_claimed_once", "injection_unknown_issue")


def test_injection_that_breaks_another_control(tmp_path):
    """A duplicate whose copy is also posted after the period end breaks C5 and C6."""
    book, _ = lookalike_book()
    book.invoice("50", "150", "c9", "2026-09-28T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-09")
    inj = Injector(book)
    book.invoice("51", "151", "c9", "2026-09-29T10:00:00+05:00", 250000, po_ref="PO-9", service_period="2026-09",
                 posted_at="2026-10-01T10:00:00+05:00", inj=inj)
    inj.commit(["1"], "duplicate_invoice", [effect("C5", "KZT", 250000)])
    truth = [issue("1", "duplicate_invoice", ["50", "51"], 250000, "KZT", [["50", "51"]])]
    assert_fails(book.write(tmp_path / "ep", issues=truth, injector=inj),
                 "injection_breaks_other_controls", "injection_issue_mismatch", "missing_truth_issue")


def test_injected_anomaly_is_rejected(tmp_path):
    """An 'fx_error' below materiality is an anomaly, whatever truth says."""
    book, _ = lookalike_book()
    inv = book.invoice("70", "170", "c11", "2026-08-14T10:00:00+05:00", 100000, currency="USD")
    inj = Injector(book)
    inject_fx(inj, inv, 47550000 + 50)
    inj.commit(["F"], "fx_error", [effect("C3", "KZT", 50)])
    truth = [issue("F", "fx_error", ["70"], 50, "KZT", [["70", inv.rows["revenue"], "2026-08-14:USD:KZT"]])]
    assert_fails(book.write(tmp_path / "ep", issues=truth, injector=inj), "anomaly", "injection_creates_anomaly")


# --- strange folders ---------------------------------------------------------------


def test_missing_folder_is_a_failed_report(tmp_path):
    report = verify_episode(tmp_path / "nope")
    assert not report.ok and not report.conditions["readable"]
    assert "unreadable" in report.codes()


@pytest.mark.parametrize("file, text", [
    ("manifest.json", "{not json"), ("manifest.json", "[]"), ("ledger.csv", "row_id\n1\n"),
    ("truth.json", "{"), ("truth.json", "[]"), ("provenance.json", "null"), ("policy.json", '"x"'),
    ("invoices.jsonl", "[1]\n"), ("truth.json", '{"issues": [1, {"type": 3}]}'),
])
def test_malformed_files_never_crash(tmp_path, file, text):
    book, inj, truth, _ = dup_case()
    path = book.write(tmp_path / "ep", issues=truth, injector=inj)
    (path / file).write_text(text, encoding="utf-8")
    report = verify_episode(path)
    assert not report.ok
    assert "internal_error" not in report.codes(), report.render(50)
    json.dumps(report.to_dict(), default=str)


def test_missing_truth_and_provenance(tmp_path):
    book, _ = lookalike_book()
    path = book.write(tmp_path / "ep")
    (path / "truth.json").unlink()
    (path / "provenance.json").unlink()
    assert_fails(path, "truth_unreadable", "provenance_unreadable")


@pytest.mark.parametrize("mutate, condition", [
    (lambda b, m: m.update(record_counts={"ledger": 1, "invoices": 0, "payments": 0, "fx": 0}), "formats"),
    (lambda b, m: m.update(document_count=999), "formats"),
    (lambda b, m: m.pop("utc_offset"), "formats"),
    (lambda b, m: b.tables["ledger"][0].update(timestamp="2026-07-10T10:00:00"), "formats"),
    (lambda b, m: b.tables["invoices"][0].update(issued_at="2026-07-10"), "formats"),
    (lambda b, m: b.tables["invoices"][0].pop("po_ref"), "formats"),
    (lambda b, m: b.tables["invoices"][0].update(service_period="July"), "formats"),
    (lambda b, m: b.tables["invoices"][0]["line_items"][0].update(amount_minor=1), "formats"),
    (lambda b, m: b.tables["fx"][0].update(rate="4.7e2"), "formats"),
    (lambda b, m: b.tables["fx"].append(dict(b.tables["fx"][0], rate="1.00")), "formats"),
    (lambda b, m: b.tables["payments"][0].update(invoice_refs=[]), "formats"),
    (lambda b, m: m.update(rounding={"mode": "half_even", "unit": "minor"}), "formats"),
])
def test_format_problems(tmp_path, mutate, condition):
    book, _ = lookalike_book()
    manifest = book.manifest()
    mutate(book, manifest)
    report = assert_fails(book.write(tmp_path / "ep", manifest=manifest))
    assert not report.conditions[condition], report.render(50)


def test_unsupported_policy_rule_fails(tmp_path):
    book, _ = lookalike_book()
    policy = book.policy()
    policy["fx_rate_date"] = "payment_date"
    assert_fails(book.write(tmp_path / "ep", policy=policy), "config", condition="formats")


def test_policy_without_optional_keys_uses_defaults(tmp_path):
    book, _ = lookalike_book()
    assert_ok(book.write(tmp_path / "ep", policy={}))


def test_distractors(tmp_path):
    book, inj, truth, _ = dup_case()
    prov = inj.provenance()
    prov["distractors"] = [{"kind": "recurring_amount", "ids": ["1", "2"]}, {"kind": "refund", "ids": ["38", "39"]}]
    assert_ok(_write(tmp_path, book, truth, prov, "a"))
    prov["distractors"] = [{"kind": "recurring_amount", "ids": ["1", "nope"]}]
    assert_fails(_write(tmp_path, book, truth, prov, "b"), "distractor_unknown_id")
    prov["distractors"] = [{"kind": "recurring_amount", "ids": ["50", "1"]}]
    assert_fails(_write(tmp_path, book, truth, prov, "c"), "distractor_overlaps_issue")

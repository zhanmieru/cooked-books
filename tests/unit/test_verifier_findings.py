"""Controls C1-C6 and the issue finder on hand-built books (public data only)."""

import copy

import pytest

from cookedbooks.episode.loader import load_public
from cookedbooks.verifier import Books, check_controls, config_from, find_issues, verify_episode
from tests.helpers.books_builder import Book
from tests.helpers.verifier_cases import lookalike_book, with_fx


def books_of(directory) -> Books:
    episode = load_public(directory)
    config, problems = config_from(episode.manifest, episode.policy)
    assert not problems, problems
    return Books(config, {k: list(v) for k, v in episode.tables.items()})


def run(book: Book, tmp_path, name="ep"):
    path = book.write(tmp_path / name)
    books = books_of(path)
    return books, check_controls(books), find_issues(books), path


def kinds(result):
    return sorted({a["kind"] for a in result.anomalies})


def test_lookalikes_are_clean(tmp_path):
    book, _ = lookalike_book()
    _, violations, result, path = run(book, tmp_path)
    assert violations == []
    assert result.issues == [] and result.anomalies == []
    assert verify_episode(path).ok


def test_lookalikes_survive_row_reordering(tmp_path):
    book, _ = lookalike_book()
    for table in book.tables.values():
        table.reverse()
    _, violations, result, path = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == []
    assert verify_episode(path).ok


@pytest.mark.parametrize("offset, issued, rate_day", [
    ("+05:30", "2026-08-13T18:30:00+00:00", "2026-08-14"),   # 00:00 local next day
    ("+05:30", "2026-08-13T18:29:59+00:00", "2026-08-13"),
    ("-05:00", "2026-08-14T03:00:00+00:00", "2026-08-13"),   # still the 13th locally
    ("-05:00", "2026-08-14T05:00:00+00:00", "2026-08-14"),
])
def test_rate_date_follows_company_offset(tmp_path, offset, issued, rate_day):
    sign = offset[0]
    book = with_fx(Book(utc_offset=offset, period_start=f"2026-07-01T00:00:00{offset}",
                        period_end=f"2026-09-30T23:59:59{offset}"))
    inv = book.invoice("1", "1", "c", issued, 100000, currency="USD")
    assert inv.fx_key == f"{rate_day}:USD:KZT"
    _, violations, result, _ = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == [], sign


def test_wrong_offset_is_an_fx_error(tmp_path):
    """Posting by the UTC date when the local date differs is a material fx_error."""
    book = with_fx(Book())
    inv = book.invoice("1", "1", "c", "2026-08-13T21:00:00+00:00", 100000, currency="USD",
                       base_amount=47025000)  # 13 Aug rate; local date is the 14th
    _, violations, result, _ = run(book, tmp_path)
    assert [v.control for v in violations] == ["C3"]
    assert result.anomalies == []
    assert result.issues == [{
        "type": "fx_error", "affected_ids": ["1"], "expected_amount_minor": 47550000 - 47025000,
        "currency": "KZT", "evidence_sets": [sorted(["1", inv.rows["revenue"], "2026-08-14:USD:KZT"])],
    }]


def test_currency_exponents_0_2_3(tmp_path):
    book = Book(exponents={"KZT": 2, "JPY": 0, "KWD": 3, "USD": 2})
    book.add_fx("2026-08-14", "JPY", "3.4567")
    book.add_fx("2026-08-14", "KWD", "1650.125")
    jpy = book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 10001, currency="JPY")
    kwd = book.invoice("2", "2", "c", "2026-08-14T10:00:00+05:00", 1234, currency="KWD")
    assert jpy.base_amount == 3457046 and kwd.base_amount == 203625
    _, violations, result, _ = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == []


def test_base_currency_with_exponent_0(tmp_path):
    book = Book(base="JPY", exponents={"JPY": 0, "USD": 2})
    book.add_fx("2026-08-14", "USD", "150.5")
    inv = book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 100, currency="USD")
    assert inv.base_amount == 151
    _, violations, result, _ = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == []


@pytest.mark.parametrize("posted_mode, policy_mode, clean", [
    ("half_up", "half_up", True), ("half_even", "half_even", True),
    ("half_even", "half_up", False), ("half_up", "half_even", False),
])
def test_rounding_mode_from_policy(tmp_path, posted_mode, policy_mode, clean):
    book = Book(rounding=posted_mode)
    book.add_fx("2026-08-14", "USD", "0.5")
    book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 5, currency="USD")   # 2.5 minor: a tie
    book.rounding = policy_mode
    _, violations, result, _ = run(book, tmp_path)
    assert result.issues == []
    if clean:
        assert violations == [] and result.anomalies == []
    else:  # off by one minor unit: above zero, below materiality
        assert kinds(result) == ["fx_below_materiality"]


@pytest.mark.parametrize("posted, outcome", [
    (20100, "issue"), (19900, "issue"), (20099, "anomaly"), (20000, "clean"),
])
def test_materiality_at_and_below_min_minor(tmp_path, posted, outcome):
    book = Book()
    book.add_fx("2026-08-14", "USD", "200.00")
    book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 100, currency="USD", base_amount=posted)
    _, _, result, _ = run(book, tmp_path)
    if outcome == "issue":
        assert [i["expected_amount_minor"] for i in result.issues] == [abs(posted - 20000)]
        assert result.anomalies == []
    elif outcome == "anomaly":
        assert result.issues == [] and kinds(result) == ["fx_below_materiality"]
    else:
        assert result.issues == [] and result.anomalies == []


@pytest.mark.parametrize("posted, outcome", [(40200, "issue"), (40199, "anomaly")])
def test_materiality_relative_threshold(tmp_path, posted, outcome):
    book = Book()
    book.add_fx("2026-08-14", "USD", "400.00")
    book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 100, currency="USD", base_amount=posted)
    _, _, result, _ = run(book, tmp_path)
    assert (len(result.issues), len(result.anomalies)) == ((1, 0) if outcome == "issue" else (0, 1))


def test_missing_rate_falls_back_to_previous_available(tmp_path):
    book = Book()
    book.add_fx("2026-08-10", "USD", "470.00")
    book.add_fx("2026-08-20", "USD", "480.00")
    book.add_fx("2026-08-15", "EUR", "999.00")        # other pair on the exact day: ignored
    book.add_fx("2026-08-15", "KZT", "0.002", "USD")  # reverse direction: ignored
    inv = book.invoice("1", "1", "c", "2026-08-15T10:00:00+05:00", 100000, currency="USD")
    assert inv.fx_key == "2026-08-10:USD:KZT"
    _, violations, result, _ = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == []


def test_no_rate_on_or_before_date_is_undecidable(tmp_path):
    book = Book()
    book.add_fx("2026-08-10", "USD", "470.00")
    book.invoice("1", "1", "c", "2026-08-15T10:00:00+05:00", 100000, currency="USD")
    book.tables["fx"][0]["date"] = "2026-08-16"
    _, violations, result, _ = run(book, tmp_path)
    assert kinds(result) == ["fx_rate_unavailable"]
    assert [v.code for v in violations] == ["rate_unavailable"]


# --- things that never exist ---------------------------------------------------


def test_three_invoices_one_key_is_an_anomaly(tmp_path):
    book = Book()
    for doc in ("1", "2", "3"):
        book.invoice(doc, "i" + doc, "c", "2026-08-01T10:00:00+05:00", 5000, po_ref="P", service_period="2026-08")
    _, violations, result, path = run(book, tmp_path)
    assert result.issues == [] and kinds(result) == ["duplicate_group_too_large"]
    assert [v.control for v in violations] == ["C5"]
    assert not verify_episode(path).ok


def test_two_open_invoices_for_one_remainder_is_an_anomaly(tmp_path):
    book = Book()
    book.invoice("1", "1", "c", "2026-08-01T10:00:00+05:00", 5000)
    book.invoice("2", "2", "c", "2026-08-02T10:00:00+05:00", 5000)
    book.payment("3", "3", "c", "2026-08-05T10:00:00+05:00", 5000)
    _, violations, result, path = run(book, tmp_path)
    assert result.issues == [] and kinds(result) == ["ambiguous_open_invoice"]
    assert sorted(v.ids for v in violations if v.control == "C4") == [("1", "3"), ("2", "3")]
    assert not verify_episode(path).ok


def test_open_invoice_with_unequal_remainder_is_an_anomaly(tmp_path):
    book = Book()
    book.invoice("1", "1", "c", "2026-08-01T10:00:00+05:00", 5000)
    book.payment("2", "2", "c", "2026-08-05T10:00:00+05:00", 3000)
    _, _, result, path = run(book, tmp_path)
    assert result.issues == [] and kinds(result) == ["unequal_open_invoice"]
    assert not verify_episode(path).ok


def test_open_invoice_after_payment_or_other_currency_or_client_is_fine(tmp_path):
    book = with_fx(Book())
    book.payment("1", "1", "c", "2026-08-05T10:00:00+05:00", 3000)
    book.invoice("2", "2", "c", "2026-08-05T10:00:01+05:00", 3000)          # issued one second later
    book.invoice("3", "3", "c", "2026-08-01T10:00:00+05:00", 3000, currency="USD")
    book.invoice("4", "4", "d", "2026-08-01T10:00:00+05:00", 3000)          # another client
    _, violations, result, _ = run(book, tmp_path)
    assert violations == [] and result.issues == [] and result.anomalies == []


def test_issued_at_equal_to_payment_time_counts(tmp_path):
    book = Book()
    book.invoice("2", "2", "c", "2026-08-05T05:00:00+00:00", 3000)            # same instant, other offset
    book.payment("1", "1", "c", "2026-08-05T10:00:00+05:00", 3000)
    _, _, result, _ = run(book, tmp_path)
    assert [i["type"] for i in result.issues] == ["unallocated_payment"]


def test_c1_violation_is_an_anomaly(tmp_path):
    book = Book()
    book.background("2026-08-01T09:00:00+05:00", "RENT_EXPENSE", 1000)
    book.tables["ledger"][-1]["credit_minor"] = 999
    _, violations, result, path = run(book, tmp_path)
    assert [v.control for v in violations] == ["C1"]
    assert kinds(result) == ["c1_violation:unbalanced_currency"]
    assert not verify_episode(path).ok


def test_c2_over_credit_is_an_anomaly(tmp_path):
    book = Book()
    book.invoice("1", "1", "c", "2026-08-01T10:00:00+05:00", 5000)
    book.payment("2", "2", "c", "2026-08-05T10:00:00+05:00", 6000, allocations={"1": 6000})
    _, violations, result, path = run(book, tmp_path)
    assert "C2" in {v.control for v in violations}
    assert "c2_violation:over_credited" in kinds(result)
    assert not verify_episode(path).ok


@pytest.mark.parametrize("break_it, code", [
    (lambda b, h: b.tables["ledger"].remove(b.record("ledger", h.rows["revenue"])), "malformed:malformed_invoice_posting"),
    (lambda b, h: b.record("ledger", h.rows["ar"]).update(timestamp="2026-08-02T10:00:00+05:00"),
     "malformed:malformed_invoice_posting"),
    (lambda b, h: b.record("ledger", h.rows["revenue"]).update(entity_id="other"), "malformed:malformed_invoice_posting"),
    (lambda b, h: b.record("ledger", h.rows["ar"]).update(invoice_id="nope"), "malformed:orphan_row"),
    (lambda b, h: b.record("ledger", h.rows["fx_base"]).update(account="AR"), "malformed:malformed_invoice_posting"),
])
def test_malformed_postings_are_anomalies(tmp_path, break_it, code):
    book = with_fx(Book())
    inv = book.invoice("1", "1", "c", "2026-08-14T10:00:00+05:00", 100000, currency="USD")
    break_it(book, inv)
    _, violations, result, path = run(book, tmp_path)
    assert code in kinds(result)
    assert any(v.control == "STRUCT" for v in violations)
    assert not verify_episode(path).ok


def test_payment_rows_must_match_invoice_refs(tmp_path):
    book = Book()
    book.invoice("1", "1", "c", "2026-08-01T10:00:00+05:00", 5000)
    book.payment("2", "2", "c", "2026-08-05T10:00:00+05:00", 5000, allocations={"1": 5000})
    book.record("payments", "2")["invoice_refs"] = None
    _, _, result, _ = run(book, tmp_path)
    assert "malformed:malformed_payment" in kinds(result)


def test_overlap_outside_scenarios_is_an_anomaly(tmp_path):
    """cutoff_error and fx_error on one invoice is not S1 or S2."""
    book = with_fx(Book())
    book.invoice("1", "1", "c", "2026-09-30T10:00:00+05:00", 100000, currency="USD",
                 posted_at="2026-10-01T10:00:00+05:00", base_amount=49000000)
    _, _, result, _ = run(book, tmp_path)
    assert sorted(i["type"] for i in result.issues) == ["cutoff_error", "fx_error"]
    assert kinds(result) == ["overlapping_issues"]


def test_ids_are_opaque_and_may_collide_across_spaces(tmp_path):
    """doc_id == row_id == payment_id == invoice_id strings; a numeric-looking ID is just a string."""
    book = Book()
    book._row = 99  # rows "100", "101", ...
    book.invoice("100", "101", "100", "2026-08-01T10:00:00+05:00", 5000, po_ref="P", service_period="2026-08")
    book.invoice("0100", "100", "100", "2026-08-02T10:00:00+05:00", 5000, po_ref="P", service_period="2026-08")
    book.payment("101", "100", "100", "2026-08-05T10:00:00+05:00", 5000, allocations={"101": 5000})
    _, violations, result, _ = run(book, tmp_path)
    assert [v.control for v in violations] == ["C5"]
    assert result.anomalies == []
    assert result.issues == [{"type": "duplicate_invoice", "affected_ids": ["0100", "100"],
                              "expected_amount_minor": 5000, "currency": "KZT",
                              "evidence_sets": [["0100", "100"]]}]


def test_find_issues_does_not_mutate_tables(tmp_path):
    book, _ = lookalike_book()
    path = book.write(tmp_path / "ep")
    episode = load_public(path)
    tables = {k: list(v) for k, v in episode.tables.items()}
    snapshot = copy.deepcopy(tables)
    config, _ = config_from(episode.manifest, episode.policy)
    find_issues(Books(config, tables))
    assert tables == snapshot

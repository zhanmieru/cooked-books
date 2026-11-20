"""Edge cases of cookedbooks.env.core.AuditEnv in closed mode."""

from __future__ import annotations

import copy
import json
import math

import pytest

from cookedbooks.env import core
from cookedbooks.env.core import MAX_STEPS, AuditEnv, empty_answer, task_text
from cookedbooks.env.errors import MESSAGES, ToolError, error
from cookedbooks.env.tools import decode_cursor, encode_cursor
from tests.helpers.factory import MINI_ISSUE, build_clean, write_episode


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def closed(fixture, clock=None, team_config=None) -> tuple[AuditEnv, dict]:
    env = AuditEnv(clock=clock)
    config = {"mode": "closed", "episode_dir": str(fixture.path)}
    if team_config is not None:
        config["team_config"] = team_config
    observation, info = env.reset(0, config)
    assert info == {}
    return env, observation


def oracle(fixture) -> dict:
    return {"episode_id": fixture.episode_id, "findings": [
        {"type": i["type"], "affected_ids": list(i["affected_ids"]),
         "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
         "evidence_ids": list(i["evidence_sets"][0]), "explanation": "x"}
        for i in fixture.issues]}


def act(name, **args) -> dict:
    return {"name": name, "args": args}


def code_of(step_result) -> str | None:
    err = step_result[0]["error"]
    return None if err is None else err["code"]


def assert_json(value) -> None:
    assert json.loads(json.dumps(value, allow_nan=False)) == value


# --- errors module ----------------------------------------------------------------


def test_eight_codes_with_distinct_fixed_messages():
    assert set(MESSAGES) == {
        "unknown_action", "unknown_table", "unknown_id", "bad_arguments", "calc_error",
        "invalid_submission", "time_limit", "episode_over",
    }
    assert len(set(MESSAGES.values())) == 8
    for code, message in MESSAGES.items():
        assert error(code) == {"code": code, "message": message}
        assert 0 < len(message) <= 80
    with pytest.raises(ValueError):
        ToolError("no_such_code")


# --- config errors ----------------------------------------------------------------


@pytest.mark.parametrize("config", [
    None, "closed", [], 1,
    {},
    {"mode": "open", "episode_dir": "x"},
    {"mode": "CLOSED", "episode_dir": "x"},
    {"mode": "closed"},
    {"mode": "closed", "episode_dir": None},
    {"mode": "closed", "episode_dir": 123},
    {"mode": "closed", "episode_dir": "PLACEHOLDER", "team_config": []},
    {"mode": "closed", "episode_dir": "PLACEHOLDER", "team_config": "x"},
])
def test_bad_config_raises_value_error(config, clean_episode):
    if isinstance(config, dict) and config.get("episode_dir") == "PLACEHOLDER":
        config = dict(config, episode_dir=str(clean_episode.path))
    env = AuditEnv()
    with pytest.raises(ValueError):
        env.reset(0, config)
    with pytest.raises(RuntimeError):
        env.step(act("calculator", expression="1"))


@pytest.mark.parametrize("limit", [0, -1, 60.5, 61, True, False, "5", None, math.nan, math.inf])
def test_bad_time_limit_raises(limit, clean_episode):
    env = AuditEnv()
    with pytest.raises(ValueError):
        env.reset(0, {"mode": "closed", "episode_dir": str(clean_episode.path),
                      "team_config": {"time_limit_s": limit}})


@pytest.mark.parametrize("limit", [60, 60.0, 1, 0.001, 30])
def test_good_time_limit(limit, clean_episode):
    _, observation = closed(clean_episode, team_config={"time_limit_s": limit})
    assert f"Time: {float(limit):g} seconds" in observation["task"]


def test_unreadable_folder_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        AuditEnv().reset(0, {"mode": "closed", "episode_dir": str(tmp_path / "missing")})


def test_seed_and_team_config_ignored_with_episode_dir(clean_episode):
    a, obs_a = closed(clean_episode)
    b = AuditEnv()
    obs_b, _ = b.reset(12345, {"mode": "closed", "episode_dir": clean_episode.path,
                               "team_config": {"n_ledger_rows": 9}})
    assert obs_a == obs_b


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        AuditEnv().step(act("calculator", expression="1"))


def test_state_before_reset():
    env = AuditEnv()
    assert env.mode is None and env.final_findings is None
    assert env.terminated is False and env.truncated is False and env.inspection_steps == 0


# --- first observation ------------------------------------------------------------


def test_first_observation_shape(clean_episode):
    env, observation = closed(clean_episode)
    assert set(observation) == {"task", "manifest", "policy", "remaining_steps"}
    assert observation["remaining_steps"] == 100
    assert observation["manifest"]["episode_id"] == clean_episode.episode_id
    assert observation["policy"]["amount_tolerance_minor"]["fx_error"] == 0
    assert env.mode == "closed"
    assert_json(observation)


def test_task_text_content():
    text = task_text(60.0)
    for needle in ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error",
                   "doc_id", "base_minor = round(src_minor * rate * 10^(exp_base - exp_src))",
                   "amount_tolerance_minor", "30 findings", "10 evidence_ids",
                   "1000 characters", "100 actions", "60 seconds", "episode_id",
                   "claimed_amount_minor", "explanation"):
        assert needle in text, needle
    lowered = text.lower()
    for banned in ("truth", "provenance", "seed", "reward", "issue_id", "evidence_sets"):
        assert banned not in lowered


# --- action parsing and step accounting ---------------------------------------------


@pytest.mark.parametrize("action, expected", [
    (None, "unknown_action"),
    ("list_records", "unknown_action"),
    (["list_records", {}], "unknown_action"),
    ({}, "unknown_action"),
    ({"args": {}}, "unknown_action"),
    ({"name": None, "args": {}}, "unknown_action"),
    ({"name": ["submit"], "args": {}}, "unknown_action"),
    ({"name": {"a": 1}, "args": {}}, "unknown_action"),
    ({"name": "List_Records", "args": {"table": "ledger"}}, "unknown_action"),
    ({"name": "list_records ", "args": {"table": "ledger"}}, "unknown_action"),
    ({"name": "read_file", "args": {"path": "truth.json"}}, "unknown_action"),
    ({"name": "list_records"}, "bad_arguments"),
    ({"name": "list_records", "args": None}, "bad_arguments"),
    ({"name": "list_records", "args": ["ledger"]}, "bad_arguments"),
    (act("list_records"), "bad_arguments"),
    (act("list_records", table=None), "bad_arguments"),
    (act("list_records", table=["ledger"]), "bad_arguments"),
    (act("list_records", table="Ledger"), "unknown_table"),
    (act("list_records", table="ledger "), "unknown_table"),
    (act("list_records", table="truth"), "unknown_table"),
    (act("list_records", table="../truth.json"), "unknown_table"),
    (act("list_records", table="ledger", limit=0), "bad_arguments"),
    (act("list_records", table="ledger", limit=-5), "bad_arguments"),
    (act("list_records", table="ledger", limit=True), "bad_arguments"),
    (act("list_records", table="ledger", limit=1.0), "bad_arguments"),
    (act("list_records", table="ledger", limit="10"), "bad_arguments"),
    (act("list_records", table="ledger", cursor=1), "bad_arguments"),
    (act("list_records", table="ledger", cursor=""), "bad_arguments"),
    (act("list_records", table="ledger", cursor="garbage!"), "bad_arguments"),
    (act("list_records", table="ledger", cursor="djF8bGVkZ2VyfDI"), None),  # offset 2: valid
    (act("list_records", table="ledger", cursor="djF8bGVkZ2VyfDI="), "bad_arguments"),
    (act("get_record", table="ledger"), "bad_arguments"),
    (act("get_record", id="7"), "bad_arguments"),
    (act("get_record", table="ledger", id=7), "bad_arguments"),
    (act("get_record", table="ledger", id=None), "bad_arguments"),
    (act("get_record", table="docs", id="7"), "unknown_table"),
    (act("get_record", table="ledger", id="U"), "unknown_id"),
    (act("get_record", table="ledger", id="7 "), "unknown_id"),
    (act("get_record", table="ledger", id=""), "unknown_id"),
    (act("get_document"), "bad_arguments"),
    (act("get_document", doc_id=7), "bad_arguments"),
    (act("get_document", id="7"), "bad_arguments"),
    (act("get_document", doc_id="U"), "unknown_id"),
    (act("get_document", doc_id="truth.json"), "unknown_id"),
    (act("get_document", doc_id="../../etc/passwd"), "unknown_id"),
    (act("calculator"), "bad_arguments"),
    (act("calculator", expression=1), "bad_arguments"),
    (act("calculator", expression="x"), "calc_error"),
    (act("calculator", expression="1/0"), "calc_error"),
    (act("calculator", expression="1" * 201), "calc_error"),
])
def test_every_call_costs_one_action(action, expected, clean_episode):
    env, _ = closed(clean_episode)
    result = env.step(action)
    observation, reward, terminated, truncated, info = result
    assert code_of(result) == expected
    assert observation["remaining_steps"] == 99
    assert (reward, terminated, truncated, info) == (0.0, False, False, {})
    if expected is not None:
        assert observation["result"] is None
        assert observation["error"] == error(expected)
    assert_json(observation)
    # The environment keeps working after an error.
    follow = env.step(act("calculator", expression="2+2"))
    assert follow[0] == {"result": "4", "error": None, "remaining_steps": 98}


def test_extra_keys_are_ignored(clean_episode):
    env, _ = closed(clean_episode)
    result = env.step({"name": "get_record", "args": {"table": "ledger", "id": "7", "x": 1}, "y": 2})
    assert result[0]["result"]["row_id"] == "7"


def test_error_message_does_not_echo_input(clean_episode, tmp_path):
    env, _ = closed(clean_episode)
    secret = "SECRET-" + str(tmp_path)
    for action in (act("get_record", table="ledger", id=secret), act("get_document", doc_id=secret),
                   act("list_records", table=secret), act("calculator", expression=secret),
                   {"name": secret, "args": {}}, act("list_records", table="ledger", cursor=secret)):
        text = json.dumps(env.step(action))
        assert "SECRET" not in text and str(tmp_path) not in text


def test_object_that_breaks_on_access_does_not_crash(clean_episode):
    class Weird(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    env, _ = closed(clean_episode)
    result = env.step({"name": "list_records", "args": Weird(table="ledger")})
    assert code_of(result) == "bad_arguments"
    assert result[0]["remaining_steps"] == 99
    result = env.step(Weird(name="submit", args={}))
    assert code_of(result) == "unknown_action"

    class WeirdArgs(dict):
        def get(self, *a, **k):
            raise RuntimeError("boom")

    result = env.step({"name": "submit", "args": WeirdArgs(findings={})})
    assert code_of(result) == "unknown_action"
    assert result[0]["remaining_steps"] == 97 and result[2:4] == (False, False)


# --- list_records: cursor and limit -----------------------------------------------


def walk(env, table, limit=None):
    cursor, pages = None, []
    while True:
        args = {"table": table, "cursor": cursor}
        if limit is not None:
            args["limit"] = limit
        observation = env.step(act("list_records", **args))[0]
        assert observation["error"] is None
        pages.append(observation["result"])
        cursor = observation["result"]["next_cursor"]
        if cursor is None:
            return pages


@pytest.mark.parametrize("limit, sizes", [
    (None, [50, 50, 20]), (50, [50, 50, 20]), (51, [50, 50, 20]), (10 ** 9, [50, 50, 20]),
    (7, [7] * 17 + [1]), (120, [50, 50, 20]), (119, [50, 50, 20]), (1, None),
])
def test_limit_and_full_walk(large_episode, limit, sizes):
    env, _ = closed(large_episode)
    if limit == 1:
        env.step(act("list_records", table="ledger", limit=1))
        page = env.step(act("list_records", table="ledger", cursor=None, limit=1))[0]["result"]
        assert [r["row_id"] for r in page["records"]] == [large_episode.row_ids[0]]
        assert isinstance(page["next_cursor"], str)
        return
    pages = walk(env, "ledger", limit)
    assert [len(p["records"]) for p in pages] == sizes
    ids = [r["row_id"] for p in pages for r in p["records"]]
    assert ids == list(large_episode.row_ids)


def test_cursor_is_stable_across_resets_and_instances(large_episode):
    env_a, _ = closed(large_episode)
    first = walk(env_a, "ledger")
    env_a.reset(0, {"mode": "closed", "episode_dir": str(large_episode.path)})
    second = walk(env_a, "ledger")
    env_b, _ = closed(large_episode)
    third = walk(env_b, "ledger")
    assert first == second == third
    cursor = first[0]["next_cursor"]
    again = env_b.step(act("list_records", table="ledger", cursor=cursor))[0]["result"]
    assert again == first[1]


def test_cursor_bound_to_table_and_range(large_episode):
    env, _ = closed(large_episode)
    cursor = env.step(act("list_records", table="ledger"))[0]["result"]["next_cursor"]
    assert code_of(env.step(act("list_records", table="fx", cursor=cursor))) == "bad_arguments"
    for bad in (encode_cursor("ledger", 0), encode_cursor("ledger", 120),
                encode_cursor("ledger", 10 ** 6), encode_cursor("ledger", -1),
                encode_cursor("fx", 1), cursor + "=", cursor.upper(), "A" + cursor, cursor + "A"):
        assert code_of(env.step(act("list_records", table="ledger", cursor=bad))) == "bad_arguments"
    assert decode_cursor(encode_cursor("ledger", 119), "ledger", 120) == 119


def test_empty_and_small_tables(tmp_path):
    rows = [{"row_id": f"r{i}", "timestamp": "2026-08-01T10:00:00+05:00", "account": "CASH",
             "debit_minor": 100, "credit_minor": 0, "currency": "KZT", "entity_id": "c",
             "invoice_id": None, "payment_id": None} for i in range(5)]
    path = write_episode(tmp_path / "five", ledger=rows)
    env = AuditEnv()
    env.reset(None, {"mode": "closed", "episode_dir": str(path)})
    ledger = env.step(act("list_records", table="ledger"))[0]["result"]
    assert ledger == {"records": rows, "next_cursor": None}
    for table in ("invoices", "payments", "fx"):
        assert env.step(act("list_records", table=table))[0]["result"] == {"records": [], "next_cursor": None}
    assert code_of(env.step(act("get_document", doc_id="r0"))) == "unknown_id"
    assert env.step(act("get_record", table="ledger", id="r4"))[0]["result"] == rows[4]
    result = env.step(act("submit", findings={"episode_id": "ep-test", "findings": []}))
    assert result[0]["result"] == {"accepted": True} and result[2] is True


# --- lookups ----------------------------------------------------------------------


def get(env, table, record_id):
    return env.step(act("get_record", table=table, id=record_id))[0]


def doc(env, doc_id):
    return env.step(act("get_document", doc_id=doc_id))[0]


def test_lookup_collisions_mini(mini_episode):
    env, _ = closed(mini_episode)
    assert get(env, "ledger", "7")["result"]["account"] == "AR"
    assert get(env, "ledger", "b7")["result"]["credit_minor"] == 10000
    assert get(env, "ledger", "B7")["result"]["debit_minor"] == 10000
    assert get(env, "invoices", "7")["result"]["invoice_id"] == "Kx"      # doc_id first
    assert get(env, "invoices", "Kx")["result"]["doc_id"] == "7"          # then invoice_id
    assert get(env, "invoices", "5")["result"]["doc_id"] == "kX"          # invoice_id "5"
    assert get(env, "invoices", "kx")["error"]["code"] == "unknown_id"
    assert get(env, "payments", "5")["result"]["payment_id"] == "7"       # doc_id
    assert get(env, "payments", "7")["result"]["doc_id"] == "5"           # payment_id
    assert get(env, "ledger", "kX")["error"]["code"] == "unknown_id"
    assert doc(env, "7")["result"]["invoice_id"] == "Kx"
    assert doc(env, "kX")["result"]["invoice_id"] == "5"
    assert doc(env, "5")["result"]["payment_id"] == "7"
    assert doc(env, "Kx")["error"]["code"] == "unknown_id"               # invoice_id is not a doc_id
    assert doc(env, "KX")["error"]["code"] == "unknown_id"
    fx = get(env, "fx", "2026-08-03:USD:KZT")["result"]
    assert fx == {"date": "2026-08-03", "source_currency": "USD", "target_currency": "KZT",
                  "rate": "470.25", "id": "2026-08-03:USD:KZT"}
    assert get(env, "fx", "2026-08-03:usd:KZT")["error"]["code"] == "unknown_id"


def test_lookup_collisions_large(large_episode):
    env, _ = closed(large_episode)
    first_row = large_episode.row_ids[0]
    assert get(env, "ledger", first_row)["result"]["row_id"] == first_row
    assert doc(env, first_row)["result"]["invoice_id"] == "9"            # doc_id equal to a row_id
    assert doc(env, "q ")["result"]["payment_id"] == "Q"                  # trailing space kept
    assert doc(env, "q")["error"]["code"] == "unknown_id"
    assert get(env, "invoices", "q")["result"]["doc_id"] == "Q"
    assert get(env, "payments", "Q")["result"]["doc_id"] == "q "
    assert get(env, "invoices", "9")["result"]["invoice_id"] == "Rw37"    # doc_id "9" wins over invoice_id "9"
    assert get(env, "payments", "10")["result"]["amount_minor"] == -500
    assert get(env, "payments", "0010")["result"]["doc_id"] == "10"
    assert get(env, "fx", "2026-07-01:USD:KZT")["result"]["rate"] == "471.10"
    assert get(env, "fx", "2026-07-01:USD:KZT#2")["result"]["rate"] == "471.15"
    assert get(env, "fx", "2026-07-01:USD:KZT#3")["error"]["code"] == "unknown_id"


def test_results_are_copies(mini_episode):
    env, observation = closed(mini_episode)
    observation["manifest"]["episode_id"] = "changed"
    observation["policy"].clear()
    record = get(env, "invoices", "7")["result"]
    record["line_items"][0]["amount_minor"] = -1
    page = env.step(act("list_records", table="invoices"))[0]["result"]
    page["records"][0]["total_minor"] = -1
    assert get(env, "invoices", "7")["result"]["line_items"][0]["amount_minor"] == 10000
    assert env.step(act("list_records", table="invoices"))[0]["result"]["records"][0]["total_minor"] == 10000
    assert env.reset(0, {"mode": "closed", "episode_dir": str(mini_episode.path)})[0]["manifest"]["episode_id"] == "ep-mini"


# --- budget -------------------------------------------------------------------------


def test_submit_as_action_100_is_accepted(mini_episode):
    env, _ = closed(mini_episode)
    for i in range(99):
        result = env.step(act("calculator", expression="1+1"))
        assert result[0]["remaining_steps"] == 99 - i and result[2:4] == (False, False)
    result = env.step(act("submit", findings=oracle(mini_episode)))
    assert result[0] == {"result": {"accepted": True}, "error": None, "remaining_steps": 0}
    assert result[1:] == (0.0, True, False, {})
    assert env.inspection_steps == 99
    assert env.final_findings == oracle(mini_episode)


def test_action_100_without_submit_truncates(mini_episode):
    env, _ = closed(mini_episode)
    for _ in range(99):
        env.step(act("calculator", expression="1"))
    result = env.step(act("get_record", table="ledger", id="7"))
    assert result[0]["result"]["row_id"] == "7"          # executed normally
    assert result[0]["remaining_steps"] == 0
    assert result[1:] == (0.0, False, True, {})
    assert env.final_findings == {"episode_id": "ep-mini", "findings": []}
    assert env.inspection_steps == 100
    assert env.truncated and not env.terminated


def test_action_100_error_still_truncates(mini_episode):
    env, _ = closed(mini_episode)
    for _ in range(99):
        env.step("junk")
    result = env.step(act("get_record", table="ledger"))  # no id: bad_arguments
    assert code_of(result) == "bad_arguments"
    assert result[2:4] == (False, True)


def test_invalid_submit_as_action_100_terminates(mini_episode):
    env, _ = closed(mini_episode)
    for _ in range(99):
        env.step("junk")
    result = env.step(act("submit", findings=[]))
    assert code_of(result) == "invalid_submission"
    assert result[2:4] == (True, False)


# --- submit -----------------------------------------------------------------------


@pytest.mark.parametrize("findings", [
    None, [], "x", {"findings": []}, {"episode_id": 1, "findings": []},
    {"episode_id": "ep-mini"}, {"episode_id": "ep-mini", "findings": {}},
    {"episode_id": "ep-mini", "findings": [dict(oracle_finding := {
        "type": "duplicate_invoice", "affected_ids": ["7", "kX"], "claimed_amount_minor": True,
        "currency": "KZT", "evidence_ids": ["7", "kX"], "explanation": ""})]},
    {"episode_id": "ep-mini", "findings": [dict(oracle_finding, evidence_ids=["7"] * 11)]},
    {"episode_id": "ep-mini", "findings": [dict(oracle_finding, explanation="a" * 1001)]},
    {"episode_id": "ep-mini", "findings": [oracle_finding] * 31},
])
def test_invalid_submission(mini_episode, findings):
    env, _ = closed(mini_episode)
    env.step(act("calculator", expression="1"))
    result = env.step(act("submit", findings=findings))
    assert result[0] == {"result": None, "error": error("invalid_submission"), "remaining_steps": 98}
    assert result[1:] == (0.0, True, False, {})
    assert env.final_findings == findings
    assert env.inspection_steps == 1


@pytest.mark.parametrize("findings", [
    {"episode_id": "ep-mini", "findings": []},
    {"episode_id": "wrong id", "findings": []},
    {"episode_id": "ep-mini", "findings": [{"type": "nonsense", "affected_ids": [],
                                             "claimed_amount_minor": -5, "currency": "XXX",
                                             "evidence_ids": ["U"], "explanation": "", "extra": 1}]},
])
def test_wrong_but_valid_submission_is_accepted(mini_episode, findings):
    env, _ = closed(mini_episode)
    result = env.step(act("submit", findings=findings))
    assert result[0] == {"result": {"accepted": True}, "error": None, "remaining_steps": 99}
    assert result[1:] == (0.0, True, False, {})
    assert env.inspection_steps == 0


@pytest.mark.parametrize("action", [
    {"name": "submit"},
    {"name": "submit", "args": None},
    {"name": "submit", "args": ["findings"]},
    {"name": "submit", "args": "x"},
    {"name": "submit", "args": {}},
    {"name": "submit", "args": {"finding": {"episode_id": "ep-mini", "findings": []}}},
    {"name": "submit", "args": {"findings": None}},
])
def test_submit_without_findings_ends_episode(mini_episode, action):
    env, _ = closed(mini_episode)
    env.step(act("calculator", expression="1"))
    env.step("junk")
    result = env.step(action)
    assert result[0] == {"result": None, "error": error("invalid_submission"), "remaining_steps": 97}
    assert result[1:] == (0.0, True, False, {})
    assert env.terminated and not env.truncated
    assert env.inspection_steps == 2          # the submit itself is not an inspection step
    assert env.final_findings is None
    after = env.step(act("submit", findings=oracle(mini_episode)))
    assert after[0]["error"] == error("episode_over") and after[2:4] == (True, False)
    assert env.final_findings is None


def test_bare_submit_as_action_100_terminates(mini_episode):
    env, _ = closed(mini_episode)
    for _ in range(99):
        env.step("junk")
    result = env.step({"name": "submit"})
    assert code_of(result) == "invalid_submission"
    assert result[2:4] == (True, False)
    assert env.inspection_steps == 99


def test_bare_submit_after_time_limit_is_time_limit(mini_episode):
    clock = FakeClock(0.0)
    env, _ = closed(mini_episode, clock=clock)
    clock.now = 61.0
    result = env.step({"name": "submit"})
    assert code_of(result) == "time_limit" and result[2:4] == (False, True)
    assert env.final_findings == {"episode_id": "ep-mini", "findings": []}


def test_stored_findings_are_a_copy(mini_episode):
    env, _ = closed(mini_episode)
    findings = oracle(mini_episode)
    env.step(act("submit", findings=findings))
    findings["findings"].clear()
    env.final_findings["findings"].clear()
    assert env.final_findings == oracle(mini_episode)


def test_unjsonable_findings_do_not_crash(mini_episode):
    env, _ = closed(mini_episode)
    loop: list = []
    loop.append(loop)
    result = env.step(act("submit", findings={"episode_id": "e", "findings": loop}))
    assert code_of(result) == "invalid_submission"
    assert result[2] is True


# --- after the end ----------------------------------------------------------------


@pytest.mark.parametrize("ending", ["submit", "invalid", "steps", "time"])
def test_step_after_end(mini_episode, ending):
    clock = FakeClock()
    env, _ = closed(mini_episode, clock=clock)
    if ending == "submit":
        last = env.step(act("submit", findings=oracle(mini_episode)))
    elif ending == "invalid":
        last = env.step(act("submit", findings={}))
    elif ending == "steps":
        for _ in range(100):
            last = env.step(act("calculator", expression="1"))
    else:
        clock.now += 61
        last = env.step(act("calculator", expression="1"))
    flags = last[2:4]
    final = env.final_findings
    steps = env.inspection_steps
    remaining = last[0]["remaining_steps"]
    for action in (act("submit", findings=oracle(mini_episode)), act("get_record", table="ledger", id="7"),
                   "junk", act("submit", findings={"episode_id": "x", "findings": []})):
        result = env.step(action)
        assert result[0] == {"result": None, "error": error("episode_over"), "remaining_steps": remaining}
        assert result[1] == 0.0 and result[2:4] == flags and result[4] == {}
    assert env.final_findings == final
    assert env.inspection_steps == steps
    assert (env.terminated, env.truncated) == flags


# --- time -------------------------------------------------------------------------


def test_time_limit_injected_clock(mini_episode):
    clock = FakeClock(500.0)
    env, _ = closed(mini_episode, clock=clock)
    clock.now = 560.0  # exactly at the limit: still in time
    assert code_of(env.step(act("calculator", expression="1"))) is None
    clock.now = 560.001
    result = env.step(act("submit", findings=oracle(mini_episode)))
    assert result[0] == {"result": None, "error": error("time_limit"), "remaining_steps": 98}
    assert result[1:] == (0.0, False, True, {})
    assert env.final_findings == {"episode_id": "ep-mini", "findings": []}
    assert env.inspection_steps == 2


def test_time_limit_lowered_by_team_config(mini_episode):
    clock = FakeClock(0.0)
    env, _ = closed(mini_episode, clock=clock, team_config={"time_limit_s": 5})
    clock.now = 5.5
    assert code_of(env.step(act("calculator", expression="1"))) == "time_limit"


def test_clock_starts_on_entry_to_reset(mini_episode, monkeypatch):
    clock = FakeClock(0.0)
    real_load = core.load_public

    def slow_load(path):
        clock.now += 59.5  # loading "takes" 59.5 seconds
        return real_load(path)

    monkeypatch.setattr(core, "load_public", slow_load)
    env, _ = closed(mini_episode, clock=clock)
    clock.now += 1.0
    assert code_of(env.step(act("calculator", expression="1"))) == "time_limit"


def test_default_clock_is_monotonic(mini_episode, monkeypatch):
    clock = FakeClock(0.0)
    monkeypatch.setattr(core.time, "monotonic", clock)
    env, _ = closed(mini_episode)
    clock.now = 61.0
    assert code_of(env.step(act("calculator", expression="1"))) == "time_limit"


# --- reset isolation --------------------------------------------------------------


def test_second_reset_starts_from_scratch(mini_episode, tmp_path):
    clock = FakeClock(0.0)
    env, first = closed(mini_episode, clock=clock)
    for _ in range(5):
        env.step(act("calculator", expression="1"))
    env.step(act("submit", findings=oracle(mini_episode)))
    clock.now = 1000.0  # far beyond the old deadline
    again, info = env.reset(0, {"mode": "closed", "episode_dir": str(mini_episode.path)})
    assert again == first and info == {}
    assert env.final_findings is None and not env.terminated and not env.truncated
    assert env.inspection_steps == 0
    result = env.step(act("get_record", table="ledger", id="7"))
    assert result[0]["remaining_steps"] == 99 and result[2:4] == (False, False)

    other = build_clean(tmp_path / "other")
    rows = [{"row_id": "only-here", "timestamp": "2026-08-01T10:00:00+05:00", "account": "CASH",
             "debit_minor": 1, "credit_minor": 0, "currency": "KZT", "entity_id": "c",
             "invoice_id": None, "payment_id": None}]
    write_episode(other.path, ledger=rows)
    env.reset(0, {"mode": "closed", "episode_dir": str(other.path)})
    assert code_of(env.step(act("get_record", table="ledger", id="7"))) == "unknown_id"
    assert code_of(env.step(act("get_document", doc_id="7"))) == "unknown_id"
    assert env.step(act("get_record", table="ledger", id="only-here"))[0]["result"] == rows[0]


def test_failed_reset_forgets_previous_episode(mini_episode):
    env, _ = closed(mini_episode)
    with pytest.raises(ValueError):
        env.reset(0, {"mode": "nope"})
    with pytest.raises(RuntimeError):
        env.step(act("calculator", expression="1"))
    assert env.mode is None


def test_same_actions_same_transitions(mini_episode):
    actions = [act("list_records", table="ledger", limit=2), act("get_document", doc_id="kX"),
               "junk", act("calculator", expression="1/3"), act("get_record", table="fx", id="x"),
               act("submit", findings=oracle(mini_episode)), act("calculator", expression="1")]

    def run():
        env, observation = closed(mini_episode)
        return [observation] + [env.step(a) for a in actions]

    assert json.dumps(run()) == json.dumps(run())


# --- JSON compatibility ---------------------------------------------------------------


def test_everything_is_json(large_episode):
    env, observation = closed(large_episode)
    outputs = [observation]
    for table in ("ledger", "invoices", "payments", "fx"):
        outputs += [{"page": p} for p in walk(env, table)]
    outputs.append(env.step(act("get_record", table="invoices", id="q")))
    outputs.append(env.step(act("get_document", doc_id="10")))
    outputs.append(env.step(act("calculator", expression="1/7")))
    outputs.append(env.step("junk"))
    outputs.append(env.step(act("submit", findings={"episode_id": "ep-large", "findings": []})))
    outputs.append(env.step("junk"))
    for output in outputs:
        assert_json(list(output) if isinstance(output, tuple) else output)
        if isinstance(output, tuple):
            _, reward, terminated, truncated, info = output
            assert type(reward) is float and type(terminated) is bool and type(truncated) is bool
            assert type(info) is dict


def test_empty_answer_helper():
    assert empty_answer({"episode_id": "e"}) == {"episode_id": "e", "findings": []}
    assert empty_answer({}) == {"episode_id": "", "findings": []}


def test_policy_mutation_of_action_does_not_leak(mini_episode):
    env, _ = closed(mini_episode)
    action = act("get_record", table="ledger", id="7")
    snapshot = copy.deepcopy(action)
    env.step(action)
    assert action == snapshot


def test_max_steps_constant():
    assert MAX_STEPS == 100

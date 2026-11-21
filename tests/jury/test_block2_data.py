"""Jury block 2, data and ground truth: D01-D20 (docs/jury-checks.md, section 3).

The jury generates every episode through OUR generator with OUR documented
settings: `judge_adapter.generate_episode(seed, recipe, dir)` with the recipe
`configs/matrix/Dnn.json`. Each `test_Dnn_*` does the same for SEEDS and checks:

1. The six jury conditions, through the independent verifier
   (`cookedbooks.verifier.verify_episode`, written from DESIGN.md without the
   generator's code). Its conditions map onto the jury's like this:
   readable, formats -> 5; clean_books -> 1; injections -> 2;
   truth_matches -> 3 and 6; clean_episode -> 3; evidence -> 4.
2. What makes the episode THAT scenario of the jury's table, read from
   `truth.json`, `provenance.json` and the public documents.
3. The reward: with issues, the oracle answer submitted through the
   environment in closed mode as the first step scores exactly 1.0 at 0
   inspection steps; a clean episode scores 1.0 for the empty answer and
   -1 for any single accusation.

A missing recipe is a failure ("recipe missing"), never a skip.

`test_independent_knobs` switches on exactly one knob of contract section 8
from an empty team_config; `test_recipes_are_distinct` checks that no two
recipes are the same effective config. More seeds:
`COOKEDBOOKS_D_SEEDS=50 pytest -q tests/jury/test_block2_data.py`.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pytest

import judge_adapter
from cookedbooks.grader.oracle import oracle_findings
from cookedbooks.verifier import CONDITIONS, verify_episode

MATRIX = Path(__file__).resolve().parents[2] / "configs" / "matrix"
SEEDS = tuple(range(int(os.environ.get("COOKEDBOOKS_D_SEEDS", "0") or 0))) or (0, 1, 2, 3, 4, 5, 77, 20261122)
ACCUSATION_SEEDS = SEEDS[:2]  # the clean-episode accusation sweep is the slow part
JURY_CONDITION = {
    "readable": "5 (formats)", "formats": "5 (formats)", "clean_books": "1 (books balance before the error)",
    "injections": "2 (the change makes the declared discrepancy)",
    "truth_matches": "3 and 6 (truth matches the change; no contradictory data)",
    "clean_episode": "3 (clean episode has empty truth)", "evidence": "4 (an evidence set proves it)",
}
ISSUE_TYPES = ("duplicate_invoice", "unallocated_payment", "fx_error", "cutoff_error")


@pytest.fixture(autouse=True)
def grader_log(tmp_path, monkeypatch):
    monkeypatch.setenv("COOKEDBOOKS_GRADER_LOG", str(tmp_path / "grader-log.jsonl"))


# --- generating and checking one episode -------------------------------------------


def recipe(name: str) -> dict:
    """The jury recipe `configs/matrix/<name>.json`; a missing file fails the test."""
    path = MATRIX / f"{name}.json"
    if not path.is_file():
        pytest.fail(f"recipe missing: configs/matrix/{name}.json does not exist", pytrace=False)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Episode:
    name: str
    seed: int
    folder: Path
    config: dict
    issues: list[dict]
    provenance: dict
    manifest: dict
    documents: dict[str, dict]  # doc_id -> invoice or payment record
    invoices_by_id: dict[str, dict]  # invoice_id -> invoice

    def where(self) -> str:
        return f"{self.name} seed {self.seed}"

    @property
    def types(self) -> Counter:
        return Counter(issue["type"] for issue in self.issues)

    @property
    def distractor_kinds(self) -> set[str]:
        return {d["kind"] for d in self.provenance["distractors"]}

    def issues_of(self, injection: dict) -> list[dict]:
        wanted = set(injection["issue_ids"])
        return [issue for issue in self.issues if issue["issue_id"] in wanted]


def generate(name: str, config: dict, seed: int, tmp_path: Path) -> Episode:
    folder = tmp_path / f"{name}-{seed}"
    judge_adapter.generate_episode(seed, config, str(folder))
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    invoices = _jsonl(folder / "invoices.jsonl")
    payments = _jsonl(folder / "payments.jsonl")
    return Episode(
        name=name, seed=seed, folder=folder, config=config, issues=truth["issues"],
        provenance=json.loads((folder / "provenance.json").read_text(encoding="utf-8")),
        manifest=json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
        documents={d["doc_id"]: d for d in [*invoices, *payments]},
        invoices_by_id={i["invoice_id"]: i for i in invoices},
    )


def assert_jury_conditions(ep: Episode) -> None:
    """All six jury conditions through the independent verifier; a failure names the condition."""
    report = verify_episode(ep.folder)
    failed = [f"{c} -> jury condition {JURY_CONDITION[c]}" for c in CONDITIONS if not report.conditions[c]]
    assert not failed, f"{ep.where()}: failed {failed}\n{report.render()}"
    assert report.ok, f"{ep.where()}\n{report.render()}"


def requested(config: dict) -> Counter:
    """(kind, variant or None, scenario or None) -> count, as the recipe asks."""
    wanted: Counter = Counter()
    for kind, value in config.get("errors", {}).items():
        if isinstance(value, int):
            wanted[(kind, None, None)] += value
        else:
            for variant, count in value.items():
                wanted[(kind, variant, None)] += count
    for name, count in config.get("scenarios", {}).items():
        wanted[(name, None, name)] += count
    return +wanted


def requested_issue_count(config: dict) -> int:
    return sum(n * (2 if scenario else 1) for (_, _, scenario), n in requested(config).items())


def assert_matches_recipe(ep: Episode) -> None:
    """truth and provenance hold exactly the injections the recipe asks for."""
    assert len(ep.issues) == requested_issue_count(ep.config), ep.where()
    made: Counter = Counter()
    for injection in ep.provenance["injections"]:
        key = (injection["scenario"] or injection["kind"], injection["variant"], injection["scenario"])
        made[key] += 1
    for (kind, variant, scenario), count in requested(ep.config).items():
        if variant is None and scenario is None and kind in ("unallocated_payment", "fx_error", "cutoff_error"):
            got = sum(n for (k, _, s), n in made.items() if k == kind and s is None)
        else:
            got = made[(kind, variant, scenario)]
        assert got == count, (ep.where(), kind, variant, scenario, dict(made))
    issued = sorted(i for inj in ep.provenance["injections"] for i in inj["issue_ids"])
    assert issued == sorted(issue["issue_id"] for issue in ep.issues), ep.where()
    wanted_distractors = {k for k, n in ep.config.get("distractors", {}).items() if n}
    assert ep.distractor_kinds == wanted_distractors, ep.where()


def submit_through_env(ep: Episode, findings: dict) -> None:
    env = judge_adapter.make_env()
    observation, info = env.reset(ep.seed, {"mode": "closed", "episode_dir": str(ep.folder), "team_config": {}})
    assert info == {} and observation["remaining_steps"] == 100
    result, reward, terminated, truncated, step_info = env.step({"name": "submit", "args": {"findings": findings}})
    assert result["error"] is None and result["result"] == {"accepted": True}, (ep.where(), result)
    assert (reward, terminated, truncated, step_info) == (0.0, True, False, {}), ep.where()


def assert_oracle_scores_one(ep: Episode) -> None:
    findings = oracle_findings(ep.folder)
    assert len(findings["findings"]) == len(ep.issues), ep.where()
    submit_through_env(ep, findings)
    metrics = judge_adapter.score_episode(str(ep.folder), findings, 0)
    assert metrics["return"] == 1.0, (ep.where(), metrics)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (len(ep.issues), 0, 0), (ep.where(), metrics)


def accusations(ep: Episode) -> list[dict]:
    """Structurally valid single findings: every type against every look-alike, plus a few plain documents."""
    base = ep.manifest["base_currency"]
    targets = [d["ids"] for d in ep.provenance["distractors"]]
    targets += [[doc] for doc in sorted(ep.documents)[:3]]
    out = []
    for ids in targets:
        amount = next((ep.documents[i].get("total_minor") or ep.documents[i].get("amount_minor")
                       for i in ids if i in ep.documents), 100)
        currency = next((ep.documents[i]["currency"] for i in ids if i in ep.documents), base)
        for kind in ISSUE_TYPES:
            out.append({"type": kind, "affected_ids": list(ids), "claimed_amount_minor": abs(amount),
                        "currency": currency, "evidence_ids": list(ids)[:10],
                        "explanation": "Accusation against a correct record."})
    return out


def assert_clean_rewards(ep: Episode) -> None:
    empty = {"episode_id": ep.manifest["episode_id"], "findings": []}
    submit_through_env(ep, empty)
    assert judge_adapter.score_episode(str(ep.folder), empty, 0)["return"] == 1.0, ep.where()
    if ep.seed not in ACCUSATION_SEEDS:
        return
    for finding in accusations(ep):
        answer = {"episode_id": ep.manifest["episode_id"], "findings": [finding]}
        metrics = judge_adapter.score_episode(str(ep.folder), answer, 0)
        assert (metrics["fp"], metrics["return"]) == (1, -1.0), (ep.where(), finding, metrics)


def episodes(name: str, tmp_path: Path) -> list[Episode]:
    """Generate the recipe for every seed and check the six conditions and the recipe."""
    config = recipe(name)
    out = []
    for seed in SEEDS:
        ep = generate(name, config, seed, tmp_path)
        assert_jury_conditions(ep)
        assert_matches_recipe(ep)
        out.append(ep)
    return out


# --- per-scenario checks -----------------------------------------------------------


def check_clean(name: str, tmp_path: Path, distractor: str | None) -> None:
    for ep in episodes(name, tmp_path):
        assert ep.issues == [] and ep.provenance["injections"] == [], ep.where()
        if distractor is None:
            assert ep.provenance["distractors"] == [], ep.where()
        else:
            assert distractor in ep.distractor_kinds, (ep.where(), ep.distractor_kinds)
        assert_clean_rewards(ep)


def check_only(name: str, tmp_path: Path, kind: str) -> list[Episode]:
    eps = episodes(name, tmp_path)
    for ep in eps:
        assert ep.issues and set(ep.types) == {kind}, (ep.where(), ep.types)
        assert all(inj["scenario"] is None and inj["kind"] == kind for inj in ep.provenance["injections"]), ep.where()
        assert_oracle_scores_one(ep)
    return eps


def variants(eps: list[Episode]) -> set[str]:
    return {inj["variant"] for ep in eps for inj in ep.provenance["injections"]}


def recipe_variants(name: str, kind: str) -> set[str]:
    value = recipe(name).get("errors", {}).get(kind, {})
    return {v for v, n in value.items() if n} if isinstance(value, dict) else set()


def substance(config: dict) -> tuple:
    """What a recipe is about, beyond its seed: size, currencies, zone, rounding, requests."""
    return (config.get("n_ledger_rows"), config.get("n_documents"), config.get("base_currency"),
            tuple(config.get("currencies", [])), config.get("timezone"), config.get("rounding"),
            json.dumps(config.get("errors", {}), sort_keys=True),
            json.dumps(config.get("scenarios", {}), sort_keys=True))


def assert_recipes_differ(names: list[str]) -> None:
    for a, b in combinations(names, 2):
        assert substance(recipe(a)) != substance(recipe(b)), f"{a} and {b} differ only in look-alikes or not at all"


# --- D01-D05: clean ----------------------------------------------------------------


def test_D01_clean_company(tmp_path):
    check_clean("D01", tmp_path, None)


def test_D02_clean_split_payment(tmp_path):
    check_clean("D02", tmp_path, "split_payment")


def test_D03_clean_refund(tmp_path):
    check_clean("D03", tmp_path, "refund")


def test_D04_clean_recurring_amounts(tmp_path):
    check_clean("D04", tmp_path, "recurring_amount")


def test_D05_clean_boundary_case(tmp_path):
    check_clean("D05", tmp_path, "boundary")


# --- D06-D08: duplicate invoice ----------------------------------------------------


def check_duplicates(name: str, tmp_path: Path) -> None:
    for ep in check_only(name, tmp_path, "duplicate_invoice"):
        for issue in ep.issues:
            pair = [ep.documents[d] for d in issue["affected_ids"]]
            assert len(pair) == 2, ep.where()
            first, second = pair
            assert first["doc_id"] != second["doc_id"] and first["invoice_id"] != second["invoice_id"], ep.where()
            key = ("entity_id", "po_ref", "service_period", "currency", "total_minor")
            assert [first[k] for k in key] == [second[k] for k in key], ep.where()
            assert first["po_ref"] is not None and first["service_period"] is not None, ep.where()
            assert (issue["expected_amount_minor"], issue["currency"]) == (first["total_minor"], first["currency"])
    assert_recipes_differ(["D06", "D07", "D08"])


def test_D06_duplicate_invoice(tmp_path):
    check_duplicates("D06", tmp_path)


def test_D07_duplicate_invoice(tmp_path):
    check_duplicates("D07", tmp_path)


def test_D08_duplicate_invoice(tmp_path):
    check_duplicates("D08", tmp_path)


# --- D09-D11: unallocated payment --------------------------------------------------


def check_unallocated(name: str, tmp_path: Path) -> None:
    eps = check_only(name, tmp_path, "unallocated_payment")
    for ep in eps:
        for injection in ep.provenance["injections"]:
            assert injection["variant"] in ("missing", "partial"), ep.where()
            (issue,) = ep.issues_of(injection)
            docs = [ep.documents[d] for d in issue["affected_ids"]]
            payment = next(d for d in docs if "payment_id" in d)
            invoice = next(d for d in docs if "invoice_id" in d and "payment_id" not in d)
            if injection["variant"] == "missing":
                assert payment["invoice_refs"] is None, ep.where()
            else:
                assert invoice["invoice_id"] in (payment["invoice_refs"] or []), ep.where()
            assert issue["currency"] == payment["currency"] == invoice["currency"], ep.where()
    assert variants(eps) == recipe_variants(name, "unallocated_payment"), name
    covered = set().union(*(recipe_variants(n, "unallocated_payment") for n in ("D09", "D10", "D11")))
    assert covered >= {"missing", "partial"}, f"D09-D11 together need a missing and a partial case, got {covered}"


def test_D09_unallocated_payment(tmp_path):
    check_unallocated("D09", tmp_path)


def test_D10_unallocated_payment(tmp_path):
    check_unallocated("D10", tmp_path)


def test_D11_unallocated_payment(tmp_path):
    check_unallocated("D11", tmp_path)


# --- D12-D13: fx error -------------------------------------------------------------


def check_fx(name: str, other: str, tmp_path: Path) -> None:
    eps = check_only(name, tmp_path, "fx_error")
    for ep in eps:
        for issue in ep.issues:
            (invoice,) = [ep.documents[d] for d in issue["affected_ids"]]
            assert invoice["currency"] != ep.manifest["base_currency"], ep.where()
            assert issue["currency"] == ep.manifest["base_currency"], ep.where()
    mine = variants(eps)
    assert len(mine) == 1 and mine == recipe_variants(name, "fx_error"), (name, mine)
    assert {frozenset(mine), frozenset(recipe_variants(other, "fx_error"))} == {
        frozenset({"wrong_date"}), frozenset({"wrong_pair"})}, "D12 and D13: one wrong_date, one wrong_pair"


def test_D12_fx_error(tmp_path):
    check_fx("D12", "D13", tmp_path)


def test_D13_fx_error(tmp_path):
    check_fx("D13", "D12", tmp_path)


# --- D14-D15: cutoff error ---------------------------------------------------------


def check_cutoff(name: str, tmp_path: Path) -> None:
    for ep in check_only(name, tmp_path, "cutoff_error"):
        for injection in ep.provenance["injections"]:
            assert injection["variant"] in ("late", "early"), ep.where()
        for issue in ep.issues:
            (invoice,) = [ep.documents[d] for d in issue["affected_ids"]]
            assert (issue["expected_amount_minor"], issue["currency"]) == (invoice["total_minor"],
                                                                          invoice["currency"]), ep.where()


def test_D14_cutoff_error(tmp_path):
    check_cutoff("D14", tmp_path)


def test_D15_cutoff_error(tmp_path):
    check_cutoff("D15", tmp_path)


# --- D16-D20: interaction scenarios ------------------------------------------------


def check_scenario(name: str, tmp_path: Path, scenario: str, group: list[str]) -> None:
    for ep in episodes(name, tmp_path):
        injections = ep.provenance["injections"]
        assert injections and all(inj["scenario"] == scenario for inj in injections), (ep.where(), injections)
        for injection in injections:
            issues = {issue["type"]: issue for issue in ep.issues_of(injection)}
            fx = issues.get("fx_error")
            if scenario == "S1":
                assert sorted(issues) == ["fx_error", "unallocated_payment"], (ep.where(), sorted(issues))
                (invoice,) = fx["affected_ids"]
                assert invoice in issues["unallocated_payment"]["affected_ids"], ep.where()  # same invoice
            else:
                assert sorted(issues) == ["duplicate_invoice", "fx_error"], (ep.where(), sorted(issues))
                (copy,) = fx["affected_ids"]
                assert copy in issues["duplicate_invoice"]["affected_ids"], ep.where()
                added = {c["key"] for c in injection["changes"] if c["file"] == "invoices.jsonl" and c["op"] == "add"}
                assert copy in added, f"{ep.where()}: the fx_error is not on the added copy"
        assert_oracle_scores_one(ep)
    assert_recipes_differ(group)


def test_D16_scenario_s1(tmp_path):
    check_scenario("D16", tmp_path, "S1", ["D16", "D17", "D18"])


def test_D17_scenario_s1(tmp_path):
    check_scenario("D17", tmp_path, "S1", ["D16", "D17", "D18"])


def test_D18_scenario_s1(tmp_path):
    check_scenario("D18", tmp_path, "S1", ["D16", "D17", "D18"])


def test_D19_scenario_s2(tmp_path):
    check_scenario("D19", tmp_path, "S2", ["D19", "D20"])


def test_D20_scenario_s2(tmp_path):
    check_scenario("D20", tmp_path, "S2", ["D19", "D20"])


# --- independent settings ----------------------------------------------------------

# Every knob of contract section 8, switched on alone. A knob that needs a
# foreign currency gets one (currencies are a setting, not a knob).
FOREIGN = {"currencies": ["KZT", "USD"]}
KNOBS: dict[str, tuple[dict, tuple[str, str | None, str | None] | None, str | None]] = {
    "duplicate_invoice": ({"errors": {"duplicate_invoice": 1}}, ("duplicate_invoice", None, None), None),
    **{f"unallocated_payment.{v}": ({"errors": {"unallocated_payment": {v: 1}}},
                                     ("unallocated_payment", v, None), None) for v in ("missing", "partial")},
    **{f"fx_error.{v}": ({**FOREIGN, "errors": {"fx_error": {v: 1}}}, ("fx_error", v, None), None)
       for v in ("wrong_date", "wrong_pair")},
    **{f"cutoff_error.{v}": ({"errors": {"cutoff_error": {v: 1}}}, ("cutoff_error", v, None), None)
       for v in ("late", "early")},
    **{f"scenario.{s}": ({**FOREIGN, "scenarios": {s: 1}}, (s, None, s), None) for s in ("S1", "S2")},
    **{f"distractor.{d}": ({"distractors": {d: 1}}, None, d)
       for d in ("split_payment", "refund", "recurring_amount", "boundary", "advance", "overpayment")},
}
SCENARIO_TYPES = {"S1": {"fx_error", "unallocated_payment"}, "S2": {"duplicate_invoice", "fx_error"}}


@pytest.mark.parametrize("knob", sorted(KNOBS))
def test_independent_knobs(tmp_path, knob):
    config, injection, distractor = KNOBS[knob]
    for seed in SEEDS[:3]:
        ep = generate(knob, config, seed, tmp_path)
        assert_jury_conditions(ep)
        made = [(inj["scenario"] or inj["kind"], inj["variant"], inj["scenario"])
                for inj in ep.provenance["injections"]]
        if injection is None:
            assert made == [] and ep.issues == [], (ep.where(), made)
        else:
            assert made == [injection], (ep.where(), made)
            kind, _, scenario = injection
            assert set(ep.types) == (SCENARIO_TYPES[scenario] if scenario else {kind}), (ep.where(), ep.types)
        assert ep.distractor_kinds == ({distractor} if distractor else set()), (ep.where(), ep.distractor_kinds)


def test_recipes_are_distinct():
    from cookedbooks.world.config import parse_config

    names = sorted(p.stem for p in MATRIX.glob("D*.json"))
    effective = {name: json.dumps(parse_config(recipe(name)).normalized, sort_keys=True) for name in names}
    same = [(a, b) for a, b in combinations(names, 2) if effective[a] == effective[b]]
    assert not same, f"recipes with the same effective config: {same}"

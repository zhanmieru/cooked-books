"""Measure lazy shortcuts: how well a trivial feature of a record predicts that it belongs to an error.

    python -m cookedbooks.verify.shortcuts [--seeds N]

Generates episodes through the public entry point (`judge_adapter.generate_episode`)
from every recipe in `configs/matrix` that asks for errors, plus one mixed
config, and reads them back the way a policy would: only the public files
(through `cookedbooks.episode.loader.load_public`). `truth.json` is read only
to score each feature. Nothing here looks at the generator's internals.

For every feature it prints how many documents the feature flags, how many of
those are in a truth issue (`affected_ids`), the precision, and the lift:
precision divided by the base rate, once over all documents and once over
documents of the same kind (invoices or payments).

Two groups of features:
- "trivial": surface properties a lazy policy might use instead of the rules
  (customer size, timing, file position, ID shape, payment shape). None of
  these should predict errors well.
- "rule": a piece of the DESIGN.md rule itself (an invoice still open that a
  payment references, a shared obligation key). High precision is expected
  and fine: that is the auditing the task asks for.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "configs" / "matrix"
DEFAULT_SEEDS = 60
MIX_CONFIG = {
    "currencies": ["KZT", "USD", "EUR"], "n_ledger_rows": 1200, "n_documents": 160,
    "errors": {"duplicate_invoice": 3, "unallocated_payment": {"missing": 3, "partial": 3}},
    "distractors": {"split_payment": 3, "advance": 3, "refund": 2, "recurring_amount": 2, "boundary": 4,
                    "overpayment": 3},
}
ADVANCE_SHARE_THRESHOLD = 20  # percent of a payment left in CUSTOMER_ADVANCES


def default_configs() -> dict[str, dict]:
    """Every recipe that asks for at least one error, plus the mixed config."""
    configs = {}
    for path in sorted(MATRIX.glob("D*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if _requested(config):
            configs[path.stem] = config
    configs["mix"] = MIX_CONFIG
    return configs


def _requested(config: dict) -> int:
    total = 0
    for value in config.get("errors", {}).values():
        total += value if isinstance(value, int) else sum(value.values())
    return total + sum(config.get("scenarios", {}).values())


# --- reading one episode ---------------------------------------------------------


def _instant(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _offset(text: str) -> timezone:
    sign = -1 if text[0] == "-" else 1
    return timezone(sign * timedelta(hours=int(text[1:3]), minutes=int(text[4:6])))


@dataclass
class Episode:
    """The public records with a few derived quantities; `affected` comes from truth.json."""

    invoices: list[dict]
    payments: list[dict]
    ledger: list[dict]
    zone: timezone
    first: date
    last: date
    affected: set[str]
    ar_credit: Counter = field(default_factory=Counter)  # invoice_id -> sum of Cr AR rows
    credits: dict = field(default_factory=lambda: defaultdict(Counter))  # payment_id -> account -> credit
    customer_of: dict = field(default_factory=dict)  # payment_id -> entity_id

    @classmethod
    def load(cls, folder: Path) -> "Episode":
        from cookedbooks.episode.loader import load_public

        public = load_public(folder)
        manifest = public.manifest
        zone = _offset(manifest["utc_offset"])
        truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
        issues = truth["issues"] if isinstance(truth, dict) else truth
        episode = cls(
            invoices=list(public.tables["invoices"]), payments=list(public.tables["payments"]),
            ledger=list(public.tables["ledger"]), zone=zone,
            first=_instant(manifest["period_start"]).astimezone(zone).date(),
            last=_instant(manifest["period_end"]).astimezone(zone).date(),
            affected={doc for issue in issues for doc in issue.get("affected_ids", [])},
        )
        for row in episode.ledger:
            if row["payment_id"] is None:
                continue
            episode.customer_of[row["payment_id"]] = row["entity_id"]
            if row["credit_minor"]:
                episode.credits[row["payment_id"]][row["account"]] += row["credit_minor"]
                if row["account"] == "AR" and row["invoice_id"] is not None:
                    episode.ar_credit[row["invoice_id"]] += row["credit_minor"]
        return episode

    def local_day(self, text: str) -> date:
        return _instant(text).astimezone(self.zone).date()

    def distance(self, day: date) -> int:
        """Days to the nearest period bound, counted on the day's own side (0 at and next to a bound)."""
        to_start = (day - self.first).days if day >= self.first else (self.first - day).days - 1
        to_end = (self.last - day).days if day <= self.last else (day - self.last).days - 1
        return min(to_start, to_end)

    def open_balance(self, invoice: dict) -> int:
        return invoice["total_minor"] - self.ar_credit[invoice["invoice_id"]]


# --- features --------------------------------------------------------------------

Feature = tuple[str, str, str]  # (group, kind, name)


def _hex_low(doc_id: str) -> bool:
    """ID control: is the last character of the ID in the lower half of its alphabet? Should not matter."""
    return doc_id[-1] in "01234567abcdefghijklm"


def invoice_features(ep: Episode) -> Iterable[tuple[Feature, dict]]:
    per_customer = Counter(i["entity_id"] for i in ep.invoices)
    open_per_pair = Counter((i["entity_id"], i["currency"]) for i in ep.invoices if ep.open_balance(i) > 0)
    totals = Counter((i["entity_id"], i["currency"], i["total_minor"]) for i in ep.invoices)
    keys = Counter((i["entity_id"], i["po_ref"], i["service_period"], i["currency"], i["total_minor"])
                   for i in ep.invoices if i["po_ref"] is not None and i["service_period"] is not None)
    groups = defaultdict(list)
    for i in ep.invoices:
        groups[(i["entity_id"], i["currency"], i["total_minor"])].append(i)
    referenced_open = set()
    for p in ep.payments:
        for ref in p["invoice_refs"] or []:
            referenced_open.add(ref)
    position = {i["doc_id"]: n for n, i in enumerate(ep.invoices)}
    end_cut = ep.last - timedelta(days=10)
    for inv in ep.invoices:
        size = per_customer[inv["entity_id"]]
        day = ep.local_day(inv["issued_at"])
        is_open = ep.open_balance(inv) > 0
        trivial = {
            "customer has exactly 1 invoice": size == 1,
            "customer has exactly 2 invoices": size == 2,
            "customer has 3+ invoices": size >= 3,
            "issued in the last 10 days of the period or later": day > end_cut,
            "issued within 2 days of a period bound": ep.distance(day) <= 2,
            "issued on a weekend": day.weekday() >= 5,
            "issued outside 08:00-20:00 local": not 8 <= _instant(inv["issued_at"]).astimezone(ep.zone).hour < 20,
            "issued_at written with another offset": _instant(inv["issued_at"]).utcoffset()
                                                      != ep.zone.utcoffset(None),
            "open at the end of the export": is_open,
            "the only open invoice of its customer in its currency":
                is_open and open_per_pair[(inv["entity_id"], inv["currency"])] == 1,
            "po_ref is null": inv["po_ref"] is None,
            "customer has another invoice with the same total": totals[(inv["entity_id"], inv["currency"],
                                                                        inv["total_minor"])] >= 2,
            "control: in the first half of invoices.jsonl": position[inv["doc_id"]] * 2 < len(ep.invoices),
            "control: doc_id ends in a low character": _hex_low(inv["doc_id"]),
        }
        group = groups[(inv["entity_id"], inv["currency"], inv["total_minor"])]
        if len(group) >= 2:
            paid = sum(1 for g in group if ep.open_balance(g) == 0)
            trivial["same-total group: no invoice paid"] = paid == 0
            trivial["same-total group: some but not all paid"] = 0 < paid < len(group)
            trivial["same-total group: all paid"] = paid == len(group)
        for name, flag in trivial.items():
            if flag:
                yield ("trivial", "invoice", name), inv
        rule = {
            "shares the obligation key with another invoice": keys.get(
                (inv["entity_id"], inv["po_ref"], inv["service_period"], inv["currency"], inv["total_minor"]), 0) >= 2,
            "open and referenced by a payment": is_open and inv["invoice_id"] in referenced_open,
        }
        for name, flag in rule.items():
            if flag:
                yield ("rule", "invoice", name), inv


def payment_features(ep: Episode) -> Iterable[tuple[Feature, dict]]:
    by_invoice_id = {i["invoice_id"]: i for i in ep.invoices}
    open_totals = defaultdict(set)
    for inv in ep.invoices:
        if ep.open_balance(inv) > 0:
            open_totals[(inv["entity_id"], inv["currency"])].add(inv["total_minor"])
    per_customer = Counter(i["entity_id"] for i in ep.invoices)
    position = {p["doc_id"]: n for n, p in enumerate(ep.payments)}
    for pay in ep.payments:
        if pay["amount_minor"] <= 0:
            continue
        credits = ep.credits[pay["payment_id"]]
        customer = ep.customer_of.get(pay["payment_id"])
        advances = credits["CUSTOMER_ADVANCES"]
        share = advances * 100 // pay["amount_minor"]
        refs = pay["invoice_refs"]
        trivial = {
            "credits both AR and CUSTOMER_ADVANCES": credits["AR"] > 0 and advances > 0,
            "incoming with no invoice reference": refs is None,
            f"advances share >= {ADVANCE_SHARE_THRESHOLD}% (with a Cr AR row)":
                credits["AR"] > 0 and share >= ADVANCE_SHARE_THRESHOLD,
            f"advances share < {ADVANCE_SHARE_THRESHOLD}% (with a Cr AR row)":
                credits["AR"] > 0 and 0 < share < ADVANCE_SHARE_THRESHOLD,
            "amount equals the total of an open invoice of its customer":
                pay["amount_minor"] in open_totals[(customer, pay["currency"])],
            "amount larger than its referenced invoices together": bool(refs) and pay["amount_minor"] > sum(
                by_invoice_id[r]["total_minor"] for r in refs if r in by_invoice_id),
            "customer has at most 1 invoice": per_customer[customer] <= 1,
            "customer has 3+ invoices": per_customer[customer] >= 3,
            "control: in the first half of payments.jsonl": position[pay["doc_id"]] * 2 < len(ep.payments),
            "control: doc_id ends in a low character": _hex_low(pay["doc_id"]),
        }
        for name, flag in trivial.items():
            if flag:
                yield ("trivial", "payment", name), pay
        rule = {
            "remainder in advances while it references an open invoice": advances > 0 and any(
                r in by_invoice_id and ep.open_balance(by_invoice_id[r]) > 0 for r in refs or []),
        }
        for name, flag in rule.items():
            if flag:
                yield ("rule", "payment", name), pay


# --- measuring -------------------------------------------------------------------


@dataclass
class Row:
    group: str
    kind: str
    name: str
    flagged: int
    hits: int
    precision: Fraction
    lift_all: Fraction
    lift_kind: Fraction


@dataclass
class Measurement:
    episodes: int
    documents: Counter  # kind -> count
    affected: Counter  # kind -> affected count
    rows: list[Row]

    def row(self, name: str, kind: str | None = None) -> Row:
        return next(r for r in self.rows if r.name == name and (kind is None or r.kind == kind))


def measure(seeds: int = DEFAULT_SEEDS, configs: dict[str, dict] | None = None,
            generate: Callable[[int, dict, str], None] | None = None) -> Measurement:
    """Generate `seeds` episodes per config and score every feature."""
    if generate is None:
        import judge_adapter

        generate = judge_adapter.generate_episode
    configs = default_configs() if configs is None else configs
    flagged: Counter = Counter()
    hits: Counter = Counter()
    documents: Counter = Counter()
    affected: Counter = Counter()
    episodes = 0
    root = Path(tempfile.mkdtemp(prefix="cookedbooks-shortcuts-"))
    try:
        for name, config in configs.items():
            for seed in range(seeds):
                folder = root / "episode"
                shutil.rmtree(folder, ignore_errors=True)
                generate(seed, config, str(folder))
                ep = Episode.load(folder)
                episodes += 1
                for kind, records in (("invoice", ep.invoices), ("payment", ep.payments)):
                    documents[kind] += len(records)
                    affected[kind] += sum(1 for r in records if r["doc_id"] in ep.affected)
                for feature, record in [*invoice_features(ep), *payment_features(ep)]:
                    flagged[feature] += 1
                    hits[feature] += record["doc_id"] in ep.affected
    finally:
        shutil.rmtree(root, ignore_errors=True)
    rate_all = Fraction(sum(affected.values()), max(1, sum(documents.values())))
    rows = []
    for feature in flagged:
        group, kind, name = feature
        precision = Fraction(hits[feature], flagged[feature])
        rate_kind = Fraction(affected[kind], max(1, documents[kind]))
        rows.append(Row(group, kind, name, flagged[feature], hits[feature], precision,
                        precision / rate_all if rate_all else Fraction(0),
                        precision / rate_kind if rate_kind else Fraction(0)))
    rows.sort(key=lambda r: (r.group != "trivial", -r.precision, r.kind, r.name))
    return Measurement(episodes, documents, affected, rows)


def render(m: Measurement) -> str:
    total, hit = sum(m.documents.values()), sum(m.affected.values())
    lines = [
        f"episodes {m.episodes}; documents {total}; in an issue {hit} "
        f"(base rate {_show(Fraction(hit, max(1, total)), 3)}; "
        f"invoices {m.affected['invoice']}/{m.documents['invoice']}, "
        f"payments {m.affected['payment']}/{m.documents['payment']})",
        "",
        f"{'group':8s} {'kind':8s} {'feature':62s} {'flagged':>8s} {'in issue':>9s} {'precision':>10s} "
        f"{'lift':>6s} {'lift/kind':>9s}",
    ]
    for r in m.rows:
        lines.append(f"{r.group:8s} {r.kind:8s} {r.name:62s} {r.flagged:8d} {r.hits:9d} "
                     f"{_show(r.precision, 3):>10s} {_show(r.lift_all, 1):>6s} {_show(r.lift_kind, 1):>9s}")
    return "\n".join(lines)


def _show(value: Fraction, digits: int) -> str:
    """A fraction as a fixed-point string, rounded half up, without going through float."""
    scale = 10 ** digits
    units = (value.numerator * scale * 2 + value.denominator) // (2 * value.denominator)
    return f"{units // scale}.{units % scale:0{digits}d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.verify.shortcuts", description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds per config (default 60)")
    args = parser.parse_args(argv)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    print(render(measure(args.seeds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

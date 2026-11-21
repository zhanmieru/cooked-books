"""Copy an episode folder with its records reordered and/or every ID replaced.

This is the jury's block 4 transformation (docs/jury-checks.md, section 5),
done on files only: it reads a source folder, writes a new folder, and never
touches the source. The ID spaces and every place they occur are the ones of
EVALUATION_CONTRACT.md, section 7.

Permutation (`permute=True`) reorders the rows of `ledger.csv`, `fx.csv`,
`invoices.jsonl` and `payments.jsonl` with a seeded shuffle. Nothing inside a
record is reordered (`line_items` keep their order), and every other list
(evidence sets, provenance changes) keeps its order too.

Relabelling (`relabel=True`) replaces every ID of the six spaces in SPACES
with a bijection per space, consistently in all eight files: definitions,
references, `truth.affected_ids`, `truth.evidence_sets`, and in
`provenance.json` the change `key`s, the full `before`/`after` records, the
`issue_ids` and `distractors[].ids`. Spaces are mapped separately, so with
the `int` style a row, a document and an invoice may all be called "7".
Never changed: fx keys (`date:SRC:TGT`), `episode_id`, `po_ref`,
`service_period`, account names, currencies, amounts and nulls.

Evidence IDs are a mix of spaces. The transformer finds each one's space by
looking it up among the IDs the source folder defines or references; an fx
key or an ID found nowhere is left as it is. An ID that belongs to several
spaces of the SOURCE is read as a doc_id when the same issue lists it in
`affected_ids`; otherwise it is an error (AmbiguousIdError), because no
mapping can be chosen for it.

Styles of new IDs (STYLES):
- `hex`: 12 random hex digits, no prefix, unique across all spaces;
- `int`: "1", "2", ... per space, so the same string names records of
  different spaces;
- `awkward`: case-only pairs, leading/trailing/inner spaces and tabs,
  non-ASCII letters, composed vs decomposed accents, leading zeros, commas,
  quotes, `#`, `|` and U+2028; unique across spaces;
- `shuffle`: the space's own IDs, moved around (every ID now names another
  record of the same space).

Everything is deterministic for a given seed; random streams are seeded from
sha256 of the seed and a label. CSV cells are handled as raw text, so numbers
and empty cells are written back exactly as they were read.

CLI: `python -m cookedbooks.verify.transform SRC DST [--permute] [--relabel]
[--seed N] [--style S] [--mapping FILE]`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SPACES = ("row_id", "doc_id", "invoice_id", "payment_id", "entity_id", "issue_id")
STYLES = ("hex", "int", "awkward", "shuffle")
DEFAULT_STYLE = "hex"
# Spaces an evidence ID may come from (EVALUATION_CONTRACT.md, section 4); fx keys are never mapped.
EVIDENCE_SPACES = ("row_id", "doc_id", "invoice_id", "payment_id")
# affected_ids and distractors are doc_ids; other spaces are tried only if the ID is not a doc_id.
DOC_FIRST = ("doc_id", "row_id", "invoice_id", "payment_id")

COPIED_FILES = ("manifest.json", "policy.json")
TABLE_FILES = ("ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
CLOSED_FILES = ("truth.json", "provenance.json")
EPISODE_FILES = COPIED_FILES + TABLE_FILES + CLOSED_FILES

# Field -> ID space, per record format. A field holding a list maps every element.
RECORD_FIELDS: dict[str, dict[str, str]] = {
    "ledger.csv": {"row_id": "row_id", "entity_id": "entity_id", "invoice_id": "invoice_id",
                   "payment_id": "payment_id"},
    "invoices.jsonl": {"doc_id": "doc_id", "invoice_id": "invoice_id", "entity_id": "entity_id"},
    "payments.jsonl": {"doc_id": "doc_id", "payment_id": "payment_id", "invoice_refs": "invoice_id",
                       "entity_id": "entity_id"},
    "fx.csv": {},
}
# The space of `provenance.injections[].changes[].key`, by `file`.
KEY_SPACE = {"ledger.csv": "row_id", "invoices.jsonl": "doc_id", "payments.jsonl": "doc_id"}
# Loader reads these CSV cells as null; they are never IDs.
_CSV_NULLS = ("", "null")
_DOMAIN = "cookedbooks-transform-v1"

IdMapping = dict[str, dict[str, str]]


class AmbiguousIdError(ValueError):
    """An ID in a mixed-space list (evidence) belongs to more than one space."""


# --- seeded randomness ---------------------------------------------------------------


def _rng(seed: int, *labels: object) -> random.Random:
    text = "|".join(str(part) for part in (_DOMAIN, seed, *labels))
    return random.Random(int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16))


# --- reading and writing the files ---------------------------------------------------


def _no_float(text: str) -> Any:
    raise ValueError(f"non-integer number {text} in an episode file")


def _no_constant(text: str) -> Any:
    raise ValueError(f"non-finite number {text} in an episode file")


def _loads(text: str) -> Any:
    return json.loads(text, parse_float=_no_float, parse_constant=_no_constant)


def _read_text(path: Path) -> str:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def read_csv(path: Path) -> tuple[list[str], list[list[str]], str]:
    """Header, raw rows (blank lines dropped) and the line terminator of a CSV file."""
    text = _read_text(path)
    terminator = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    if not rows:
        return [], [], terminator
    body = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    return rows[0], body, terminator


def write_csv(path: Path, header: list[str], rows: Iterable[list[str]], terminator: str) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=terminator)
    if header:
        writer.writerow(header)
    writer.writerows(rows)
    _write_text(path, buffer.getvalue())


def read_jsonl(path: Path) -> list[Any]:
    """Every non-blank line as a JSON value. Only \\n, \\r and \\r\\n end a line."""
    text = _read_text(path)
    return [_loads(line) for line in io.StringIO(text, newline=None) if line.strip()]


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    _write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def read_json(path: Path) -> Any:
    return _loads(_read_text(path))


def write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


# --- the ID spaces of one folder -----------------------------------------------------


class _Collector:
    """Every ID per space, in order of first occurrence (definitions and references alike)."""

    def __init__(self) -> None:
        self.spaces: dict[str, dict[str, None]] = {space: {} for space in SPACES}

    def add(self, space: str, value: Any) -> None:
        if isinstance(value, str):
            self.spaces[space].setdefault(value, None)

    def add_record(self, file_name: str, record: Any) -> None:
        if not isinstance(record, dict):
            return
        for field, space in RECORD_FIELDS.get(file_name, {}).items():
            value = record.get(field)
            for item in value if isinstance(value, list) else [value]:
                self.add(space, item)

    def known(self) -> dict[str, frozenset[str]]:
        return {space: frozenset(ids) for space, ids in self.spaces.items()}


def _issues_of(truth: Any) -> list[Any]:
    """The issue list of a wrapper object or a bare list (anything else: no issues)."""
    if isinstance(truth, dict) and isinstance(truth.get("issues"), list):
        return truth["issues"]
    return truth if isinstance(truth, list) else []


def _changes_of(provenance: Any) -> Iterable[dict]:
    if not isinstance(provenance, dict) or not isinstance(provenance.get("injections"), list):
        return
    for injection in provenance["injections"]:
        if isinstance(injection, dict) and isinstance(injection.get("changes"), list):
            yield from (c for c in injection["changes"] if isinstance(c, dict))


def _collect(csv_tables: Mapping[str, tuple[list[str], list[list[str]], str]],
             json_tables: Mapping[str, list[Any]], truth: Any, provenance: Any) -> _Collector:
    ids = _Collector()
    header, rows, _ = csv_tables["ledger.csv"]
    for row in rows:
        ids.add_record("ledger.csv", _csv_record(header, row))
    for file_name, records in json_tables.items():
        for record in records:
            ids.add_record(file_name, record)
    for change in _changes_of(provenance):
        file_name = change.get("file")
        if file_name in KEY_SPACE:
            ids.add(KEY_SPACE[file_name], change.get("key"))
        for side in ("before", "after"):
            ids.add_record(file_name, change.get(side))
    for issue in _issues_of(truth):
        if isinstance(issue, dict):
            ids.add("issue_id", issue.get("issue_id"))
    if isinstance(provenance, dict):
        for injection in provenance.get("injections") or []:
            if isinstance(injection, dict) and isinstance(injection.get("issue_ids"), list):
                for issue_id in injection["issue_ids"]:
                    ids.add("issue_id", issue_id)
    return ids


def _csv_record(header: Sequence[str], row: Sequence[str]) -> dict[str, str | None]:
    return {column: (row[k] if k < len(row) and row[k] not in _CSV_NULLS else None)
            for k, column in enumerate(header)}


# --- new IDs per style ---------------------------------------------------------------


def _hex_ids(count: int, rng: random.Random, used: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < count:
        candidate = f"{rng.getrandbits(48):012x}"
        if candidate not in used:
            used.add(candidate)
            out.append(candidate)
    return out


def _int_ids(count: int) -> list[str]:
    return [str(n) for n in range(1, count + 1)]


_TAGS = {"row_id": "r", "doc_id": "d", "invoice_id": "i", "payment_id": "p", "entity_id": "e", "issue_id": "x"}


def _awkward_forms(space: str, j: int) -> list[str]:
    """Awkward IDs number `j` of `space`; unique across spaces because of the tag letter and zero count."""
    tag = _TAGS[space]
    zeros = "0" * (SPACES.index(space) + 1)
    mixed = f"Id{tag}{j}"
    return [
        mixed, mixed.swapcase(),                     # equal after case folding
        f" {tag} {j}", f"{tag}{j}  ", f"{tag}\t{j}",  # leading, inner, trailing spaces, a tab
        f"{zeros}{j}",                               # leading zero(s), all digits
        f"Счёт-{tag}{j}", f"編號{tag}{j}", f"ß{tag}{j}", f"\U0001F9FE{tag}{j}",
        f"Caf\u00e9{tag}{j}", f"Cafe\u0301{tag}{j}",  # composed vs decomposed accent
        f"{tag},{j}", f'"{tag}"{j}', f"{tag}#{j}", f"{tag}|{j}", f"{tag}\u2028{j}",
    ]


def _awkward_ids(space: str, count: int) -> list[str]:
    out: list[str] = []
    j = 1
    while len(out) < count:
        out.extend(_awkward_forms(space, j))
        j += 1
    return out[:count]


def build_mapping(known: Mapping[str, Iterable[str]], *, seed: int, style: str) -> IdMapping:
    """A bijection per space from the space's IDs to new IDs of `style`."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {', '.join(STYLES)}")
    used: set[str] = set()
    mapping: IdMapping = {}
    for space in SPACES:
        originals = sorted(set(known.get(space, ())))  # sorted only to be independent of row order
        rng = _rng(seed, "relabel", style, space)
        if style == "shuffle":
            order = list(originals)
            rng.shuffle(order)
            # A cyclic shift of a shuffled list moves every ID when there are two or more.
            mapping[space] = {old: order[(k + 1) % len(order)] for k, old in enumerate(order)}
            continue
        if style == "hex":
            new_ids = _hex_ids(len(originals), rng, used)
        elif style == "int":
            new_ids = _int_ids(len(originals))
        else:
            new_ids = _awkward_ids(space, len(originals))
        rng.shuffle(new_ids)
        mapping[space] = dict(zip(originals, new_ids))
    return mapping


def identity_mapping(known: Mapping[str, Iterable[str]]) -> IdMapping:
    return {space: {i: i for i in sorted(set(known.get(space, ())))} for space in SPACES}


# --- applying a mapping --------------------------------------------------------------


def resolve(value: str, mapping: Mapping[str, Mapping[str, str]], spaces: Sequence[str], *,
            first_wins: bool, doc_ids: Iterable[str] = ()) -> str:
    """Map one ID of unknown space. Unknown everywhere: unchanged (fx keys).

    `first_wins`: the first space in `spaces` that knows the ID decides.
    Otherwise an ID known to several spaces is read as a doc_id if it is in
    `doc_ids` (the same issue's or finding's affected_ids, which are doc_ids
    by contract), and raises AmbiguousIdError if not.
    """
    found = [space for space in spaces if value in mapping.get(space, {})]
    if not found:
        return value
    if len(found) > 1 and not first_wins and len({mapping[space][value] for space in found}) > 1:
        if "doc_id" in found and value in set(doc_ids):
            return mapping["doc_id"][value]
        raise AmbiguousIdError(f"ID {value!r} belongs to several spaces: {', '.join(found)}")
    return mapping[found[0]][value]


def _map_one(value: Any, table: Mapping[str, str]) -> Any:
    return table.get(value, value) if isinstance(value, str) else value


def map_record(file_name: str, record: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """A copy of one record (JSON or CSV dict form) with its ID fields mapped; nulls stay null."""
    if not isinstance(record, dict):
        return copy.deepcopy(record)
    out = copy.deepcopy(record)
    for field, space in RECORD_FIELDS.get(file_name, {}).items():
        if field not in out:
            continue
        table = mapping.get(space, {})
        value = out[field]
        out[field] = [_map_one(v, table) for v in value] if isinstance(value, list) else _map_one(value, table)
    return out


def _map_csv_row(header: Sequence[str], row: list[str], mapping: Mapping[str, Mapping[str, str]]) -> list[str]:
    fields = RECORD_FIELDS["ledger.csv"]
    out = list(row)
    for k, column in enumerate(header):
        if k < len(out) and column in fields and out[k] not in _CSV_NULLS:
            out[k] = mapping.get(fields[column], {}).get(out[k], out[k])
    return out


def _map_list(values: Any, mapping: Mapping[str, Mapping[str, str]], spaces: Sequence[str], *,
              first_wins: bool, doc_ids: Any = ()) -> Any:
    if not isinstance(values, list):
        return copy.deepcopy(values)
    docs = [d for d in doc_ids if isinstance(d, str)] if isinstance(doc_ids, list) else []
    return [resolve(v, mapping, spaces, first_wins=first_wins, doc_ids=docs) if isinstance(v, str)
            else copy.deepcopy(v) for v in values]


def map_issue(issue: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """A copy of one truth issue with `issue_id`, `affected_ids` and `evidence_sets` mapped."""
    if not isinstance(issue, dict):
        return copy.deepcopy(issue)
    out = copy.deepcopy(issue)
    if "issue_id" in out:
        out["issue_id"] = _map_one(out["issue_id"], mapping.get("issue_id", {}))
    if "affected_ids" in out:
        out["affected_ids"] = _map_list(issue["affected_ids"], mapping, DOC_FIRST, first_wins=True)
    if isinstance(issue.get("evidence_sets"), list):
        out["evidence_sets"] = [_map_list(s, mapping, EVIDENCE_SPACES, first_wins=False,
                                          doc_ids=issue.get("affected_ids"))
                                for s in issue["evidence_sets"]]
    return out


def map_truth(truth: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """A copy of truth.json (wrapper or bare list) with every issue mapped."""
    if isinstance(truth, list):
        return [map_issue(i, mapping) for i in truth]
    out = copy.deepcopy(truth)
    if isinstance(out, dict) and isinstance(truth.get("issues"), list):
        out["issues"] = [map_issue(i, mapping) for i in truth["issues"]]
    return out


def map_provenance(provenance: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """A copy of provenance.json with change keys, before/after, issue_ids and distractor ids mapped."""
    out = copy.deepcopy(provenance)
    if not isinstance(out, dict):
        return out
    for injection in out.get("injections") or []:
        if not isinstance(injection, dict):
            continue
        if isinstance(injection.get("issue_ids"), list):
            table = mapping.get("issue_id", {})
            injection["issue_ids"] = [_map_one(i, table) for i in injection["issue_ids"]]
        for change in injection.get("changes") or []:
            if not isinstance(change, dict):
                continue
            file_name = change.get("file")
            if file_name in KEY_SPACE and "key" in change:
                change["key"] = _map_one(change["key"], mapping.get(KEY_SPACE[file_name], {}))
            for side in ("before", "after"):
                if side in change:
                    change[side] = map_record(file_name, change[side], mapping)
    for distractor in out.get("distractors") or []:
        if isinstance(distractor, dict) and "ids" in distractor:
            distractor["ids"] = _map_list(distractor["ids"], mapping, DOC_FIRST, first_wins=True)
    return out


def map_findings(findings: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """A copy of a findings object (`{"episode_id", "findings"}`) with affected and evidence IDs mapped."""
    out = copy.deepcopy(findings)
    if not isinstance(out, dict) or not isinstance(out.get("findings"), list):
        return out
    for index, item in enumerate(out["findings"]):
        if not isinstance(item, dict):
            continue
        if "affected_ids" in item:
            item["affected_ids"] = _map_list(item["affected_ids"], mapping, DOC_FIRST, first_wins=True)
        if "evidence_ids" in item:
            item["evidence_ids"] = _map_list(item["evidence_ids"], mapping, EVIDENCE_SPACES, first_wins=False,
                                             doc_ids=findings["findings"][index].get("affected_ids"))
    return out


def invert(mapping: Mapping[str, Mapping[str, str]]) -> IdMapping:
    """The inverse mapping per space. Raises ValueError if a space is not one-to-one."""
    inverse: IdMapping = {}
    for space, table in mapping.items():
        inverse[space] = {new: old for old, new in table.items()}
        if len(inverse[space]) != len(table):
            raise ValueError(f"the mapping of {space} is not one-to-one")
    return inverse


def restore_findings(findings: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """Map a findings object written with the NEW IDs back to the original IDs.

    Raises AmbiguousIdError when an evidence ID is a new ID of several spaces
    (the `int` style); compare in the forward direction (`map_findings`) then.
    """
    return map_findings(findings, invert(mapping))


def restore_issue(issue: Any, mapping: Mapping[str, Mapping[str, str]]) -> Any:
    """Map a truth issue written with the NEW IDs back to the original IDs (see restore_findings)."""
    return map_issue(issue, invert(mapping))


def is_ambiguous(mapping: Mapping[str, Mapping[str, str]]) -> bool:
    """True if some new ID is used by more than one evidence space."""
    seen: set[str] = set()
    for space in EVIDENCE_SPACES:
        new_ids = set(mapping.get(space, {}).values())
        if seen & new_ids:
            return True
        seen |= new_ids
    return False


# --- permutation ---------------------------------------------------------------------


def _permuted(rows: list[Any], seed: int, label: str) -> list[Any]:
    """A seeded shuffle of `rows`; with two or more rows the order always changes."""
    order = list(range(len(rows)))
    _rng(seed, "permute", label).shuffle(order)
    if len(order) > 1 and order == sorted(order):
        order = order[1:] + order[:1]
    return [rows[k] for k in order]


# --- the transformation --------------------------------------------------------------


def transform_episode(src: str | os.PathLike, dst: str | os.PathLike, *, permute: bool, relabel: bool,
                      seed: int, style: str = DEFAULT_STYLE,
                      mapping_path: str | os.PathLike | None = None,
                      mapping: Mapping[str, Mapping[str, str]] | None = None) -> IdMapping:
    """Write a transformed copy of episode folder `src` into `dst` and return the ID mapping.

    `dst` must not exist or must be an empty folder, and must not be `src`.
    The mapping is `{space: {old_id: new_id}}` for every space in SPACES
    (the identity when `relabel` is False). A given `mapping` (with
    `relabel=True`) is applied instead of a new one, for example
    `invert(...)` of an earlier run; it must cover every ID of the source
    and be one-to-one per space. With `mapping_path` it is also
    written as JSON there; that file must lie outside `dst`. Files other than
    the eight episode files are not copied. A missing optional file stays
    missing.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {', '.join(STYLES)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source, target = Path(src), Path(dst)
    if not source.is_dir():
        raise ValueError(f"source folder {source} does not exist")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"destination {target} exists and is not an empty folder")
    if _same_or_inside(target, source) or _same_or_inside(source, target):
        raise ValueError("source and destination must be separate folders")
    if mapping_path is not None and _same_or_inside(Path(mapping_path), target):
        raise ValueError("the mapping file must lie outside the destination folder")

    present = {name for name in EPISODE_FILES if (source / name).is_file()}
    csv_tables = {name: read_csv(source / name) if name in present else ([], [], "\n")
                  for name in ("ledger.csv", "fx.csv")}
    json_tables = {name: read_jsonl(source / name) if name in present else []
                   for name in ("invoices.jsonl", "payments.jsonl")}
    truth = read_json(source / "truth.json") if "truth.json" in present else None
    provenance = read_json(source / "provenance.json") if "provenance.json" in present else None

    known = _collect(csv_tables, json_tables, truth, provenance).known()
    if not relabel:
        mapping = identity_mapping(known)
    elif mapping is None:
        mapping = build_mapping(known, seed=seed, style=style)
    else:
        mapping = _checked_mapping(mapping, known)

    if relabel:
        header, rows, term = csv_tables["ledger.csv"]
        csv_tables["ledger.csv"] = (header, [_map_csv_row(header, r, mapping) for r in rows], term)
        json_tables = {name: [map_record(name, r, mapping) for r in records]
                       for name, records in json_tables.items()}
        truth = map_truth(truth, mapping) if truth is not None else None
        provenance = map_provenance(provenance, mapping) if provenance is not None else None
    if permute:
        for name, (header, rows, term) in csv_tables.items():
            csv_tables[name] = (header, _permuted(rows, seed, name), term)
        json_tables = {name: _permuted(records, seed, name) for name, records in json_tables.items()}

    target.mkdir(parents=True, exist_ok=True)
    for name in COPIED_FILES:
        if name in present:
            shutil.copyfile(source / name, target / name)
    for name, (header, rows, term) in csv_tables.items():
        if name in present:
            write_csv(target / name, header, rows, term)
    for name, records in json_tables.items():
        if name in present:
            write_jsonl(target / name, records)
    if truth is not None:
        write_json(target / "truth.json", truth)
    if provenance is not None:
        write_json(target / "provenance.json", provenance)
    if mapping_path is not None:
        write_mapping(mapping, mapping_path)
    return mapping


def _checked_mapping(mapping: Mapping[str, Mapping[str, str]], known: Mapping[str, frozenset[str]]) -> IdMapping:
    """A copy of a caller's mapping, restricted to the source's IDs; ValueError if it misses one."""
    out: IdMapping = {}
    for space in SPACES:
        table = mapping.get(space, {})
        missing = sorted(known[space] - set(table))
        if missing:
            raise ValueError(f"the mapping has no new ID for {len(missing)} {space}(s), e.g. {missing[0]!r}")
        out[space] = {old: table[old] for old in sorted(known[space])}
        if len(set(out[space].values())) != len(out[space]):
            raise ValueError(f"the mapping of {space} is not one-to-one")
    return out


def _same_or_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except ValueError:
        return False
    return True


def write_mapping(mapping: Mapping[str, Mapping[str, str]], path: str | os.PathLike) -> None:
    """Write `{space: {old: new}}` as JSON (keys sorted)."""
    text = json.dumps({space: dict(sorted(mapping[space].items())) for space in sorted(mapping)},
                      ensure_ascii=False, indent=2, sort_keys=True)
    _write_text(Path(path), text + "\n")


def read_mapping(path: str | os.PathLike) -> IdMapping:
    value = read_json(Path(path))
    if not isinstance(value, dict) or not all(
        isinstance(t, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in t.items())
        for t in value.values()
    ):
        raise ValueError("a mapping file is an object of {space: {old_id: new_id}}")
    return value


# --- CLI -----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None, out: Callable[[str], Any] = print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cookedbooks.verify.transform",
        description="Copy an episode folder with records reordered and/or IDs replaced (jury block 4).",
    )
    parser.add_argument("src", help="source episode folder (never modified)")
    parser.add_argument("dst", help="destination folder (must not exist or be empty)")
    parser.add_argument("--permute", action="store_true", help="reorder the rows of the four tables")
    parser.add_argument("--relabel", action="store_true", help="replace every ID, space by space")
    parser.add_argument("--seed", type=int, default=0, help="seed of the shuffles and new IDs (default 0)")
    parser.add_argument("--style", choices=STYLES, default=DEFAULT_STYLE, help="style of new IDs")
    parser.add_argument("--mapping", help="write the {space: {old: new}} mapping to this JSON file")
    args = parser.parse_args(argv)
    try:
        mapping = transform_episode(args.src, args.dst, permute=args.permute, relabel=args.relabel,
                                    seed=args.seed, style=args.style, mapping_path=args.mapping)
    except (ValueError, OSError, csv.Error) as exc:
        out(f"transform failed: {exc}")
        return 2
    counts = ", ".join(f"{space} {len(mapping[space])}" for space in SPACES)
    what = " + ".join(w for w, on in (("permuted", args.permute), (f"relabelled ({args.style})", args.relabel))
                      if on) or "copied"
    out(f"{what} {args.src} -> {args.dst} (seed {args.seed}); IDs: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

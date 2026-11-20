"""The four read-only tools: list_records, get_record, get_document, calculator.

Every tool works on data already loaded in memory. Lookups are dictionary
lookups by exact string; no caller string becomes a file path, a query or a
command. Tools return deep copies, so a policy can mutate what it gets
without touching the episode. A rejected call raises ToolError with a
public error code.
"""

from __future__ import annotations

import base64
import binascii
import copy
from typing import Any

from cookedbooks.env import calculator
from cookedbooks.env.errors import (
    BAD_ARGUMENTS, CALC_ERROR, UNKNOWN_ID, UNKNOWN_TABLE, ToolError,
)
from cookedbooks.episode.loader import TABLES, PublicEpisode

MAX_LIMIT = 50
_CURSOR_PREFIX = "v1"


def encode_cursor(table: str, offset: int) -> str:
    """Opaque cursor for page start `offset` of `table` (same input, same string)."""
    raw = f"{_CURSOR_PREFIX}|{table}|{offset}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, table: str, size: int) -> int:
    """Return the offset a cursor points to, or raise ToolError(bad_arguments).

    Only cursors this module could have issued for `table` are accepted:
    the string must re-encode to itself, and the offset must be inside the
    table (a cursor is never issued for offset 0 or for the end).
    """
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        text = base64.urlsafe_b64decode(padded.encode("ascii")).decode("ascii")
        prefix, cursor_table, offset_text = text.split("|")
        offset = int(offset_text)
    except (ValueError, UnicodeError, binascii.Error):
        raise ToolError(BAD_ARGUMENTS) from None
    if prefix != _CURSOR_PREFIX or cursor_table != table or not 0 < offset < size:
        raise ToolError(BAD_ARGUMENTS)
    if encode_cursor(table, offset) != cursor:
        raise ToolError(BAD_ARGUMENTS)
    return offset


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _table_arg(args: dict) -> str:
    table = args.get("table")
    if not isinstance(table, str):
        raise ToolError(BAD_ARGUMENTS)
    if table not in TABLES:
        raise ToolError(UNKNOWN_TABLE)
    return table


def _string_arg(args: dict, name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str):
        raise ToolError(BAD_ARGUMENTS)
    return value


class Tools:
    """Read-only tools over one loaded episode.

    `tables` is read from the episode once here and cached, because the
    loader deep-copies the tables on every access.
    """

    def __init__(self, episode: PublicEpisode) -> None:
        self._episode = episode
        self._tables: dict[str, tuple[dict, ...]] = episode.tables

    def list_records(self, args: dict) -> dict:
        """One page of a table in file order: {"records": [...], "next_cursor": str | None}."""
        table = _table_arg(args)
        limit = args.get("limit", None)
        if limit is None:
            limit = MAX_LIMIT
        elif not _is_int(limit) or limit < 1:
            raise ToolError(BAD_ARGUMENTS)
        limit = min(limit, MAX_LIMIT)
        records = self._tables[table]
        cursor = args.get("cursor", None)
        if cursor is None:
            offset = 0
        elif isinstance(cursor, str):
            offset = decode_cursor(cursor, table, len(records))
        else:
            raise ToolError(BAD_ARGUMENTS)
        end = offset + limit
        page = [copy.deepcopy(record) for record in records[offset:end]]
        next_cursor = encode_cursor(table, end) if end < len(records) else None
        return {"records": page, "next_cursor": next_cursor}

    def get_record(self, args: dict) -> dict:
        """A record by exact ID in its table's own ID space (contract section 3)."""
        table = _table_arg(args)
        record_id = _string_arg(args, "id")
        record = self._episode.get_record(table, record_id)
        if record is None:
            raise ToolError(UNKNOWN_ID)
        return record

    def get_document(self, args: dict) -> dict:
        """An invoice or a payment by exact doc_id."""
        doc_id = _string_arg(args, "doc_id")
        document = self._episode.get_document(doc_id)
        if document is None:
            raise ToolError(UNKNOWN_ID)
        return document

    def calculator(self, args: dict) -> str:
        """Evaluate an arithmetic expression; the result is a decimal string."""
        expression = _string_arg(args, "expression")
        try:
            return calculator.evaluate(expression)
        except calculator.CalcError:
            raise ToolError(CALC_ERROR) from None

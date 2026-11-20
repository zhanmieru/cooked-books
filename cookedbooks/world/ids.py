"""Opaque public IDs, assigned at export time after every record exists.

An ID is a readable prefix plus 8 hex digits of sha256 over (seed, attempt,
ID space, position in a seeded shuffle). The hex carries no information about
time, file position, record type (invoices and payments share the `doc-`
space) or whether a record was injected. Hex parts are unique across all
spaces.
"""

from __future__ import annotations

from cookedbooks.world.rng import digest_hex, stream

PREFIXES = {"row": "row-", "doc": "doc-", "invoice_id": "INV-", "payment_id": "PAY-", "entity": "cust-"}
HEX_DIGITS = 8


def assign_ids(seed: int, attempt: int, spaces: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """Map each internal ref of each space to a public ID. `spaces` lists refs in a deterministic order."""
    rng = stream(seed, attempt, "ids")
    used: set[str] = set()
    mapping: dict[str, dict[str, str]] = {}
    for space, prefix in PREFIXES.items():
        refs = list(spaces.get(space, []))
        rng.shuffle(refs)
        mapping[space] = {}
        for position, ref in enumerate(refs):
            salt = 0
            while True:
                hex_part = digest_hex(seed, attempt, "id", space, position, salt)[:HEX_DIGITS]
                if hex_part not in used:
                    break
                salt += 1
            used.add(hex_part)
            mapping[space][ref] = prefix + hex_part
    return mapping

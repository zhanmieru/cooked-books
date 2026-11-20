"""The injector registry: INJECTORS maps (kind, variant) to an injector object.

kind is an error type ("duplicate_invoice", ...) or a scenario ("S1", "S2").
variant is a variant name ("missing", ...) or None. An entry with variant
None handles every variant of its kind (it receives the variant in
`ctx.request.variant`, None when the config gave a bare number).
"""

from __future__ import annotations

import random
from typing import Callable

from cookedbooks.world.config import ERROR_VARIANTS
from cookedbooks.world.inject.base import Injector

INJECTORS: dict[tuple[str, str | None], Injector] = {}


def register(kind: str, variant: str | None = None) -> Callable:
    """Class decorator: `@register("duplicate_invoice")` registers one instance."""
    def wrap(cls):
        INJECTORS[(kind, variant)] = cls()
        return cls
    return wrap


def resolve(kind: str, variant: str | None, rng: random.Random) -> tuple[Injector, str | None]:
    """Find the injector for a request; picks a registered variant when `variant` is None.

    Raises NotImplementedError naming the kind (and variant) when none is registered.
    """
    if variant is not None:
        for key in ((kind, variant), (kind, None)):
            if key in INJECTORS:
                return INJECTORS[key], variant
        raise NotImplementedError(f"no injector is registered for {kind}.{variant}")
    if (kind, None) in INJECTORS:
        return INJECTORS[(kind, None)], None
    variants = [v for v in ERROR_VARIANTS.get(kind, ()) if (kind, v) in INJECTORS]
    if not variants:
        raise NotImplementedError(f"no injector is registered for {kind}")
    chosen = rng.choice(variants)
    return INJECTORS[(kind, chosen)], chosen

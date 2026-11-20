"""Injection machinery: the injector interface, the registry and the change recorder.

See `base.py` for the interface. Production injectors register themselves on
import; they are imported at the bottom of this file, so any import of the
package (the pipeline imports `inject.base`) registers all of them.
"""

from cookedbooks.world.inject.base import (
    ApplyContext, Footprint, InjectionOutcome, InjectionPlan, InjectionRequest, Injector,
    IssueDraft, PlanContext, ReserveSpec,
)
from cookedbooks.world.inject.recorder import Change, ChangeRecorder
from cookedbooks.world.inject.registry import INJECTORS, register, resolve

__all__ = [
    "ApplyContext", "Change", "ChangeRecorder", "Footprint", "INJECTORS", "InjectionOutcome",
    "InjectionPlan", "InjectionRequest", "Injector", "IssueDraft", "PlanContext", "ReserveSpec",
    "register", "resolve",
]

# Production injectors: importing a module registers it (see registry.register).
from cookedbooks.world.inject import duplicate_invoice, unallocated_payment  # noqa: E402,F401

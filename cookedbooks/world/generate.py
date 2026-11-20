"""generate_episode: plan -> simulate -> inject -> order and assign IDs -> export.

Same (seed, team_config, attempt) -> byte-identical files. `attempt` selects
an independent re-generation (the verifier-driven retry loop of DESIGN.md
section 7 passes 0, 1, 2, ...).
"""

from __future__ import annotations

import json
import os
from typing import Any

from cookedbooks.world import GENERATOR_VERSION
from cookedbooks.world.config import ERROR_VARIANTS, SCENARIOS, WorldConfig, parse_config
from cookedbooks.world.export import AppliedInjection, EpisodeMeta, export_episode
from cookedbooks.world.inject.base import (
    ApplyContext, InjectionPlan, InjectionRequest, Injector, PlanContext,
)
from cookedbooks.world.inject.recorder import ChangeRecorder
from cookedbooks.world.inject.registry import resolve
from cookedbooks.world.model import World
from cookedbooks.world.rng import digest_hex, stream
from cookedbooks.world.simulate import simulate
from cookedbooks.world.timeline import Timeline


def expand_requests(config: WorldConfig) -> list[tuple[str, str | None, str | None]]:
    """(kind, variant or None, scenario or None) for every requested error, in a fixed order."""
    requests: list[tuple[str, str | None, str | None]] = []
    for kind in ERROR_VARIANTS:
        value = config.errors[kind]
        if isinstance(value, int):
            requests += [(kind, None, None)] * value
        else:
            for variant, count in value.items():
                requests += [(kind, variant, None)] * count
    for name in SCENARIOS:
        requests += [(name, None, name)] * config.scenarios[name]
    return requests


def _resolve_all(config: WorldConfig, seed: int, attempt: int) -> list[tuple[InjectionRequest, Injector]]:
    resolved = []
    for index, (kind, variant, scenario) in enumerate(expand_requests(config)):
        injector, chosen = resolve(kind, variant, stream(seed, attempt, f"variant:{index}"))
        resolved.append((InjectionRequest(kind, chosen, scenario, index), injector))
    return resolved


def _episode_id(seed: int, attempt: int, normalized: dict) -> str:
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return "ep-" + digest_hex(seed, attempt, "episode", canonical)[:8]


def build_world(seed: int, team_config: Any, *, attempt: int = 0
                ) -> tuple[WorldConfig, World, list[AppliedInjection]]:
    """Everything up to export: the final world and the recorded injections."""
    config = parse_config(team_config)
    resolved = _resolve_all(config, seed, attempt)  # NotImplementedError before any work
    timeline = Timeline(config.period_first, config.period_last, config.utc_offset)
    plans: list[InjectionPlan] = []
    for request, injector in resolved:
        ctx = PlanContext(request=request, base_currency=config.base_currency,
                          foreign_currencies=config.foreign_currencies, timeline=timeline,
                          rng=stream(seed, attempt, f"plan:{request.index}"))
        plans.append(injector.plan(ctx))
    world = simulate(config, seed, attempt, plans)
    applied: list[AppliedInjection] = []
    for (request, injector), plan in zip(resolved, plans):
        recorder = ChangeRecorder(world)
        docs, rows = world.document_count(), world.row_count()
        outcome = injector.apply(ApplyContext(
            world=world, request=request, plan=plan, targets=world.reserved.get(request.index, ()),
            recorder=recorder, rng=stream(seed, attempt, f"inject:{request.index}"),
        ))
        delta = (world.document_count() - docs, world.row_count() - rows)
        declared = (plan.footprint.documents, plan.footprint.ledger_rows)
        if delta != declared:
            raise RuntimeError(f"injector for {request.kind} declared footprint {declared} "
                               f"(documents, rows) but changed {delta}")
        applied.append(AppliedInjection(request, outcome, list(recorder.changes)))
    return config, world, applied


def generate_episode(seed: int, team_config: dict, output_dir: str | os.PathLike, *, attempt: int = 0) -> None:
    """Write the eight episode files for (seed, team_config) into `output_dir` (created if missing)."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError("attempt must be a non-negative int")
    config, world, applied = build_world(seed, team_config, attempt=attempt)
    meta = EpisodeMeta(
        episode_id=_episode_id(seed, attempt, config.normalized), seed=seed, attempt=attempt,
        generator_version=GENERATOR_VERSION, timezone=config.timezone, team_config=config.normalized,
    )
    export_episode(world, applied, meta, output_dir)

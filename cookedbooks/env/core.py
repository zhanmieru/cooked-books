"""AuditEnv: the reset/step environment of EVALUATION_CONTRACT.md, section 3.

Isolation. In `closed` mode the environment reads only the six public files
through `cookedbooks.episode.loader`, never opens `truth.json` or
`provenance.json`, never imports `cookedbooks.grader`, and every reward is
0.0 with an empty `info`. In `train` mode it calls the grader once, when the
episode ends, through a lazy import.

Robustness. `step` never raises for anything the policy sends: every
malformed action becomes a public error code and costs one action. Only
calling `step` before `reset` (a runner bug) raises RuntimeError, and only a
bad `config` makes `reset` raise ValueError.
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import time
from typing import Any, Callable

from cookedbooks.env.errors import (
    BAD_ARGUMENTS, EPISODE_OVER, INVALID_SUBMISSION, TIME_LIMIT, UNKNOWN_ACTION,
    ToolError, error,
)
from cookedbooks.env.tools import MAX_LIMIT, Tools
from cookedbooks.episode.loader import PublicEpisode, load_public
from cookedbooks.findings import (
    MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS, MAX_FINDINGS, validate_findings,
)

MAX_STEPS = 100
DEFAULT_TIME_LIMIT_S = 60.0
MODES = ("closed", "train")
ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")

Observation = dict[str, Any]
StepResult = tuple[Observation, float, bool, bool, dict]


class AuditEnv:
    """One auditing episode at a time: `reset(seed, config)`, then `step(action)`.

    Read-only state for a runner on the judge side (not part of any
    observation): `mode`, `inspection_steps`, `final_findings`,
    `terminated`, `truncated`.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        """`clock` returns seconds as a float; None means `time.monotonic` (looked up per call)."""
        self._clock = clock
        self._temp_dir: str | None = None
        self._clear()

    # --- read-only state --------------------------------------------------------

    @property
    def mode(self) -> str | None:
        """"closed" or "train"; None before the first successful reset."""
        return self._mode

    @property
    def inspection_steps(self) -> int:
        """Step calls before the submit (the submit itself excluded); all calls on truncation."""
        if self._submitted:
            return self._steps_taken - 1
        return self._steps_taken

    @property
    def final_findings(self) -> Any:
        """None while the episode runs; then the submitted object (valid or not) or the empty answer."""
        return _safe_copy(self._final_findings)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    # --- reset ------------------------------------------------------------------

    def reset(self, seed: Any = None, config: Any = None) -> tuple[Observation, dict]:
        """Start a fresh episode and return `(observation, info)`.

        config = {"mode": "closed" | "train", "episode_dir": str, "team_config": {...}}.
        With `episode_dir` the folder is loaded as is and `seed` is ignored.
        Without it (train only) a new episode is generated from `seed` and
        `team_config` into a temporary folder. Raises ValueError for a bad
        config or an unreadable folder.
        """
        start = self._now()
        self._clear()
        self._remove_temp_dir()

        mode, episode_dir, team_config = _parse_config(config)
        time_limit = _time_limit(team_config)
        if episode_dir is None:
            episode_dir = self._generate(seed, team_config)
        episode = load_public(episode_dir)

        self._mode = mode
        self._episode_dir = episode_dir
        self._episode = episode
        self._tools = Tools(episode)
        self._manifest = episode.manifest
        self._deadline = start + time_limit
        self._ready = True
        observation = {
            "task": task_text(time_limit),
            "manifest": episode.manifest,
            "policy": episode.policy,
            "remaining_steps": MAX_STEPS,
        }
        return observation, {}

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.monotonic()

    def close(self) -> None:
        """Remove the temporary folder of a generated episode, if any."""
        self._remove_temp_dir()

    def _generate(self, seed: Any, team_config: dict) -> str:
        try:
            from cookedbooks.world.generate import generate_episode
        except ImportError:
            raise RuntimeError("the episode generator (cookedbooks.world.generate) is not available") from None
        self._temp_dir = tempfile.mkdtemp(prefix="cookedbooks-episode-")
        generate_episode(seed, copy.deepcopy(team_config), self._temp_dir)
        return self._temp_dir

    def _remove_temp_dir(self) -> None:
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def _clear(self) -> None:
        """Forget everything about the previous episode."""
        self._ready = False
        self._mode: str | None = None
        self._episode_dir: str | None = None
        self._episode: PublicEpisode | None = None
        self._tools: Tools | None = None
        self._manifest: dict = {}
        self._deadline = 0.0
        self._steps_taken = 0
        self._submitted = False
        self._terminated = False
        self._truncated = False
        self._final_findings: Any = None

    # --- step -------------------------------------------------------------------

    def step(self, action: Any) -> StepResult:
        """Run one action and return `(observation, reward, terminated, truncated, info)`."""
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        if self._terminated or self._truncated:
            return self._observation(None, EPISODE_OVER), 0.0, self._terminated, self._truncated, {}

        self._steps_taken += 1
        if self._now() > self._deadline:
            return self._truncate(None, TIME_LIMIT)

        try:
            name, args, code = _parse_action(action)
            if name == "submit":
                # Every submit ends the episode. Missing `args` or `findings`
                # is submitted as null, which fails the structure check.
                findings = args.get("findings") if isinstance(args, dict) else None
                return self._submit(findings)
        except Exception:  # an action object that misbehaves on access
            name, args, code = None, {}, UNKNOWN_ACTION
            if self._terminated:  # the failure came from scoring, not from the action
                raise

        result = None
        if code is None:
            try:
                result = getattr(self._tools, name)(args)
            except ToolError as exc:
                code = exc.code
            except Exception:  # a tool must never take the environment down
                code = BAD_ARGUMENTS

        if self._steps_taken >= MAX_STEPS:
            return self._truncate(result, code)
        return self._observation(result, code), 0.0, False, False, {}

    def _submit(self, findings: Any) -> StepResult:
        try:
            valid = validate_findings(findings) is None
        except Exception:
            valid = False
        stored = _safe_copy(findings)
        self._submitted = True
        self._terminated = True
        self._final_findings = stored
        reward, info = self._final_reward(stored)
        observation = self._observation({"accepted": True}, None) if valid \
            else self._observation(None, INVALID_SUBMISSION)
        return observation, reward, True, False, info

    def _truncate(self, result: Any, code: str | None) -> StepResult:
        self._truncated = True
        self._final_findings = empty_answer(self._manifest)
        reward, info = self._final_reward(self._final_findings)
        return self._observation(result, code), reward, False, True, info

    def _final_reward(self, findings: Any) -> tuple[float, dict]:
        """0.0 and {} in closed mode; the grader's return and metrics in train mode."""
        if self._mode != "train":
            return 0.0, {}
        from cookedbooks.grader.score import score_episode

        metrics = score_episode(self._episode_dir, _safe_copy(findings), self.inspection_steps)
        return float(metrics["return"]), {"metrics": copy.deepcopy(metrics)}

    def _observation(self, result: Any, code: str | None) -> Observation:
        return {
            "result": None if code is not None else result,
            "error": None if code is None else error(code),
            "remaining_steps": MAX_STEPS - self._steps_taken,
        }


# --- helpers -----------------------------------------------------------------------


def _safe_copy(value: Any) -> Any:
    """A deep copy, or the value itself if it cannot be copied (never JSON from a real policy)."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def empty_answer(manifest: dict) -> dict:
    """The answer scored when an episode is truncated."""
    episode_id = manifest.get("episode_id") if isinstance(manifest, dict) else None
    return {"episode_id": episode_id if isinstance(episode_id, str) else "", "findings": []}


def _parse_action(action: Any) -> tuple[str | None, dict, str | None]:
    """Return (name, args, error_code); error_code is None for a well-formed action."""
    if not isinstance(action, dict):
        return None, {}, UNKNOWN_ACTION
    name = action.get("name")
    if not isinstance(name, str) or name not in ACTIONS:
        return None, {}, UNKNOWN_ACTION
    args = action.get("args")
    if not isinstance(args, dict):
        return name, {}, BAD_ARGUMENTS
    return name, args, None


def _parse_config(config: Any) -> tuple[str, str | None, dict]:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    mode = config.get("mode")
    if mode not in MODES:
        raise ValueError('config["mode"] must be "closed" or "train"')
    episode_dir = config.get("episode_dir")
    if episode_dir is not None:
        if not isinstance(episode_dir, (str, os.PathLike)):
            raise ValueError('config["episode_dir"] must be a path')
        episode_dir = os.fspath(episode_dir)
        if not isinstance(episode_dir, str):
            raise ValueError('config["episode_dir"] must be a path')
    elif mode == "closed":
        raise ValueError('closed mode needs config["episode_dir"]')
    team_config = config.get("team_config")
    if team_config is None:
        team_config = {}
    if not isinstance(team_config, dict):
        raise ValueError('config["team_config"] must be a dict')
    return mode, episode_dir, team_config


def _time_limit(team_config: dict) -> float:
    if "time_limit_s" not in team_config:
        return DEFAULT_TIME_LIMIT_S
    value = team_config["time_limit_s"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("team_config.time_limit_s must be a number")
    if not math.isfinite(value) or not 0 < value <= DEFAULT_TIME_LIMIT_S:
        raise ValueError("team_config.time_limit_s must be in (0, 60]")
    return float(value)


def task_text(time_limit_s: float) -> str:
    """The task statement of the first observation."""
    return f"""\
You are auditing the books of a fictional company for one reporting period. The manifest gives the period (both bounds inclusive), the base currency, currency exponents and the company's fixed UTC offset; local dates are always taken at that offset. The policy gives the rules every finding is judged by.

Send each action as {{"name": "<action>", "args": {{...}}}}. Actions:
- list_records(table, cursor, limit): table is one of ledger, invoices, payments, fx. Records come in file order. The first cursor is null; pass next_cursor to get the next page; next_cursor is null after the last page. limit is an integer from 1, at most {MAX_LIMIT} (default {MAX_LIMIT}).
- get_record(table, id): ledger by row_id; invoices by doc_id, then invoice_id; payments by doc_id, then payment_id; fx by its key "date:SRC:TGT". IDs are compared exactly.
- get_document(doc_id): an invoice or a payment by doc_id.
- calculator(expression): decimal numbers, + - * /, parentheses and unary minus, up to 200 characters. Returns a decimal string.
- submit(findings): your final answer. It ends the episode.

Submit this object:
{{"episode_id": "<manifest.episode_id>", "findings": [{{"type": "<type>", "affected_ids": ["<id>"], "claimed_amount_minor": <integer>, "currency": "<code>", "evidence_ids": ["<id>"], "explanation": "<text>"}}]}}
Money is an integer in minor units of the currency. Report one finding per error; an empty findings list means the books are clean.

The four types:
- duplicate_invoice: one obligation invoiced twice (same values of policy.duplicate_key). affected_ids: the doc_id of both invoices. Amount: the invoice total_minor in the invoice currency. Evidence: both doc_ids.
- unallocated_payment: a payment left fully or partly unallocated (the rest sits in CUSTOMER_ADVANCES) while an open invoice of the same customer and currency, issued no later than the payment, exists (policy.allocation_rule). affected_ids: the doc_id of the payment and of the invoice. Amount: the unallocated part, in the payment currency. Evidence: both doc_ids.
- fx_error: a foreign-currency invoice converted to the base currency at the rate of the wrong date or the wrong currency pair. affected_ids: the invoice doc_id. Amount: the absolute difference between the posted and the correct base amount, in the base currency. Evidence: the invoice doc_id, the row_id of its REVENUE ledger line and the fx key of the correct rate.
- cutoff_error: an invoice posted on the other side of a period bound than its local issued_at (policy.cutoff_basis). affected_ids: the invoice doc_id. Amount: the invoice total_minor in the invoice currency. Evidence: the invoice doc_id and the row_id of its AR line (or of its REVENUE line).
Documents in affected_ids are identified by doc_id only, never by invoice_id or payment_id. evidence_ids may hold row_ids, doc_ids, invoice_ids, payment_ids and fx keys; every one must exist.

FX conversion: base_minor = round(src_minor * rate * 10^(exp_base - exp_src)), where an fx record means 1 source_currency = rate target_currency, exp is from manifest.currency_exponents and rounding follows policy.rounding. The rate is the one for the local date of issued_at (policy.fx_rate_date); if that date has no rate, use the previous available one (policy.fx_missing_rate). Only differences meeting policy.fx_materiality are errors.

Amount tolerance: a claimed amount matches if it is within policy.amount_tolerance_minor[type] of the correct one; a missing value means 0 (exact).

Limits: at most {MAX_FINDINGS} findings, at most {MAX_EVIDENCE_IDS} evidence_ids per finding (repeats count), explanation at most {MAX_EXPLANATION_CHARS} characters. A submission that breaks this structure is rejected as invalid and still ends the episode.

Budget: {MAX_STEPS} actions per episode including submit; every action counts, failed ones too. Time: {time_limit_s:g} seconds from reset. If the actions or the time run out before you submit, your answer is empty."""

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
            import cookedbooks.world.generate  # noqa: F401  (availability check)
            from cookedbooks.pipeline import generate_verified
        except ImportError:
            raise RuntimeError("the episode generator (cookedbooks.world.generate) is not available") from None
        self._temp_dir = tempfile.mkdtemp(prefix="cookedbooks-episode-")
        generate_verified(seed, copy.deepcopy(team_config), self._temp_dir)
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
    """The task statement of the first observation: the complete rule book an agent needs."""
    return f"""\
You are auditing the books of a fictional company for one reporting period. Find every error of the four types in section 4 and submit them with evidence. The manifest, the policy, the four tables and the rules below are all you need; the rules are complete.

1. Actions. Send {{"name": "<action>", "args": {{...}}}}.
- list_records(table, cursor, limit): table is one of ledger, invoices, payments, fx. Records come in file order; order carries no meaning. The first cursor is null; pass next_cursor to get the next page; next_cursor is null after the last page. limit is an integer from 1, at most {MAX_LIMIT} (default {MAX_LIMIT}). A page holds at most {MAX_LIMIT} records. Books have at most 2000 ledger rows and at most 200 documents (invoices plus payments), so every table can be read in full within the budget.
- get_record(table, id): ledger by row_id; invoices by doc_id, then invoice_id; payments by doc_id, then payment_id; fx by its key "date:SRC:TGT".
- get_document(doc_id): an invoice or a payment by doc_id.
- calculator(expression): decimal numbers, + - * /, parentheses and unary minus, up to 200 characters. Returns a decimal string.
- submit(findings): your final answer. It ends the episode.

2. Values. IDs are opaque strings compared exactly (case and spaces count); nothing follows from their form or from row order. Money is an integer in minor units of its currency. Times are ISO-8601 with an offset and are compared as instants: 2026-09-30T21:00:00+00:00 and 2026-10-01T02:00:00+05:00 are the same moment. The local date of a time is its calendar date after conversion to manifest.utc_offset. A time is inside the period if period_start <= time <= period_end (both bounds inclusive). An fx record means 1 source_currency = rate target_currency on its date; its key is "date:SRC:TGT"; rates exist on business days only.

3. Ledger terms. Each ledger row has exactly one of debit_minor, credit_minor above 0 (Dr = debit, Cr = credit). The ledger's invoice_id and payment_id hold the documents' business keys, not doc_ids. Rows without invoice_id, payment_id and entity_id are background entries and are never part of an error.
- An invoice's posting is every ledger row with its invoice_id and an empty payment_id; these rows share one timestamp, the posting time. Its AR line is the Dr AR row for total_minor in the invoice currency; its REVENUE line is the Cr REVENUE row in the base currency. For a foreign-currency invoice the REVENUE credit is the posted base amount.
- A payment's rows are every ledger row with its payment_id. The payment's customer is the entity_id on these rows (payments have no entity_id field). An incoming payment (amount_minor > 0) has Dr CASH for the full amount, one Cr AR row with invoice_id per invoice it pays, and the rest in one Cr CUSTOMER_ADVANCES row. A payment with amount_minor < 0 is a refund to the customer and is never an error.
- Allocated to an invoice = the sum of all Cr AR rows with its invoice_id, from any payment. An invoice is open if total_minor minus allocated is above 0 at the end of the books. The unallocated remainder of an incoming payment = amount_minor minus the sum of its own Cr AR rows (it equals its CUSTOMER_ADVANCES credit).

4. Error types. Condition; affected_ids; amount and currency; evidence.
- duplicate_invoice: two invoices with equal values of every field of policy.duplicate_key (entity_id, po_ref, service_period, currency, total_minor). An invoice with any of these fields null or empty takes no part. affected_ids: both invoice doc_ids. Amount: total_minor in the invoice currency. Evidence: both doc_ids.
- unallocated_payment: an incoming payment has an unallocated remainder above 0 while its customer has an open invoice in the payment currency issued no later than the payment (issued_at <= payment timestamp; policy.allocation_rule). There is never more than one such invoice; with none, the remainder is a real advance. affected_ids: the payment doc_id and the invoice doc_id. Amount: the payment's unallocated remainder, in the payment currency. Evidence: both doc_ids.
- fx_error: an invoice in a currency other than manifest.base_currency whose posted base amount misses the correct one. Correct base amount: base_minor = round(src_minor * rate * 10^(exp_base - exp_src)), with src_minor = total_minor, exp from manifest.currency_exponents, computed exactly in decimals (no floating point) and rounded to an integer by policy.rounding (half_up: halves away from zero; half_even: halves to the even integer). The rate is the fx record with source_currency = the invoice currency and target_currency = the base currency (policy.fx_direction) on the local date of issued_at (policy.fx_rate_date); if that pair has no record on that date, the record of the same pair with the latest earlier date (policy.fx_missing_rate). It is an error only if the difference |posted - correct| is at least policy.fx_materiality.min_minor AND at least policy.fx_materiality.min_relative times the correct amount (both inclusive). affected_ids: the invoice doc_id. Amount: the difference, in the base currency. Evidence: the invoice doc_id, the row_id of its REVENUE line and the key of the fx record that gives the correct rate (the earlier date's key when the fallback applies).
- cutoff_error: issued_at and the posting time of an invoice lie on different sides of a period bound, each time being before period_start, inside the period or after period_end (policy.cutoff_basis). Both bounds count. affected_ids: the invoice doc_id. Amount: total_minor in the invoice currency. Evidence: the invoice doc_id and the row_id of its AR line, or the invoice doc_id and the row_id of its REVENUE line.

5. Look-alikes. The books hold legitimate cases that resemble errors; do not report them: a monthly invoice with the same amount but another service_period; an invoice paid by several fully allocated payments; a real advance; an overpayment (the referenced invoice is fully paid, the excess is a real advance); a refund after a double payment; an invoice near midnight or a period bound whose local date differs from the date as written; an invoice on a day without a rate converted at the previous available rate; a posting a day or two from issued_at on the same side of the bounds. A document belongs to at most one error, with two exceptions: a foreign-currency invoice can carry both an fx_error and an unallocated_payment (with its partly allocated payment), and one invoice of a duplicate_invoice pair can also carry an fx_error. Books may be clean.

6. Answer. Submit this object:
{{"episode_id": "<manifest.episode_id>", "findings": [{{"type": "<type>", "affected_ids": ["<id>"], "claimed_amount_minor": <integer>, "currency": "<code>", "evidence_ids": ["<id>"], "explanation": "<text>"}}]}}
Report one finding per error; an empty findings list means the books are clean. type, currency and explanation are strings, affected_ids and evidence_ids are lists of strings, claimed_amount_minor is a JSON integer. Documents in affected_ids are identified by doc_id only, never by invoice_id or payment_id; the set must be exactly the one above. evidence_ids must include a whole evidence set; extra IDs are allowed if they exist (row_ids, doc_ids, invoice_ids, payment_ids, fx keys); one ID that does not exist makes the finding wrong. Currency codes are compared exactly. Amount tolerance: a claimed amount matches if it is within policy.amount_tolerance_minor[type] of the correct one; a missing value means 0 (exact). The explanation is free text and is not used to judge a finding.
Limits: at most {MAX_FINDINGS} findings, at most {MAX_EVIDENCE_IDS} evidence_ids per finding (repeats count), explanation at most {MAX_EXPLANATION_CHARS} characters. A submission that breaks this structure is rejected as invalid and still ends the episode.

7. Budget: {MAX_STEPS} actions per episode including submit; every action counts, failed ones too. Time: {time_limit_s:g} seconds from reset. If the actions or the time run out before you submit, your answer is empty."""

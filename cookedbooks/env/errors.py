"""Public error codes of the environment (EVALUATION_CONTRACT.md, section 3).

Each code has one fixed short English message. An error object is exactly
`{"code": ..., "message": ...}`: no paths, tracebacks, seeds, echoed IDs or
exception text ever reach the policy.
"""

from __future__ import annotations

UNKNOWN_ACTION = "unknown_action"
UNKNOWN_TABLE = "unknown_table"
UNKNOWN_ID = "unknown_id"
BAD_ARGUMENTS = "bad_arguments"
CALC_ERROR = "calc_error"
INVALID_SUBMISSION = "invalid_submission"
TIME_LIMIT = "time_limit"
EPISODE_OVER = "episode_over"

MESSAGES: dict[str, str] = {
    UNKNOWN_ACTION: "Unknown action.",
    UNKNOWN_TABLE: "Unknown table.",
    UNKNOWN_ID: "No record with this ID.",
    BAD_ARGUMENTS: "Missing or invalid arguments.",
    CALC_ERROR: "Expression not supported or division by zero.",
    INVALID_SUBMISSION: "Submission structure is invalid.",
    TIME_LIMIT: "Episode time limit exceeded.",
    EPISODE_OVER: "Episode is over.",
}


class ToolError(Exception):
    """Raised inside a tool to return the public error `code` to the policy."""

    def __init__(self, code: str) -> None:
        if code not in MESSAGES:
            raise ValueError("unknown error code")
        super().__init__(code)
        self.code = code


def error(code: str) -> dict[str, str]:
    """Return a fresh error object for `code`."""
    return {"code": code, "message": MESSAGES[code]}

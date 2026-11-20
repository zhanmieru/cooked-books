"""Independent verifier of generated episodes (DESIGN.md section 7).

Written from DESIGN.md and EVALUATION_CONTRACT.md alone. It must never
import `cookedbooks.world`: if the verifier and the generator shared code,
their agreement would prove nothing.
"""

from .books import Books, Config, Violation, config_from
from .identities import check_controls
from .pipeline import verify_episode
from .predicates import FindResult, find_issues
from .report import CONDITIONS, Problem, Report
from .revert import RevertResult, revert

__all__ = [
    "Books", "CONDITIONS", "Config", "FindResult", "Problem", "Report", "RevertResult", "Violation",
    "check_controls", "config_from", "find_issues", "revert", "verify_episode",
]

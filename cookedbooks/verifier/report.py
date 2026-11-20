"""The verifier's report: one boolean per jury condition and the details behind each failure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Jury block 2 (docs/jury-checks.md section 3), as the verifier splits it.
CONDITIONS = (
    "readable",       # the folder loads at all
    "formats",        # contract section 2 formats, manifest counts (jury 5)
    "clean_books",    # revert -> C1..C6 hold, no issues, no anomalies (jury 1)
    "truth_matches",  # find_issues on published books == truth.json, no anomalies (jury 3, 6)
    "injections",     # each injection makes exactly its declared issues and controls (jury 2)
    "evidence",       # every evidence set re-derives its issue (jury 4)
    "clean_episode",  # no issues <=> empty truth <=> no injections (jury 3)
)


@dataclass
class Problem:
    condition: str
    code: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"condition": self.condition, "code": self.code, "message": self.message, "details": self.details}


@dataclass
class Report:
    """Result of `verify_episode`. `ok` is True only if every condition holds."""

    episode_dir: str
    episode_id: str | None = None
    problems: list[Problem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    found_issues: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    truth_issue_count: int | None = None
    injection_count: int | None = None
    _evaluated: set[str] = field(default_factory=set)

    def fail(self, condition: str, code: str, message: str, **details: Any) -> None:
        self.problems.append(Problem(condition, code, message, details))

    def evaluated(self, condition: str) -> None:
        self._evaluated.add(condition)

    @property
    def conditions(self) -> dict[str, bool]:
        failed = {p.condition for p in self.problems}
        return {c: c in self._evaluated and c not in failed for c in CONDITIONS}

    @property
    def ok(self) -> bool:
        return all(self.conditions.values())

    def failures(self, condition: str | None = None) -> list[Problem]:
        return [p for p in self.problems if condition is None or p.condition == condition]

    def codes(self) -> set[str]:
        return {p.code for p in self.problems}

    def to_dict(self) -> dict:
        return {
            "episode_dir": self.episode_dir,
            "episode_id": self.episode_id,
            "ok": self.ok,
            "conditions": self.conditions,
            "problems": [p.to_dict() for p in self.problems],
            "warnings": list(self.warnings),
            "found_issues": self.found_issues,
            "anomalies": self.anomalies,
            "truth_issue_count": self.truth_issue_count,
            "injection_count": self.injection_count,
        }

    def render(self, max_problems: int = 15) -> str:
        """A short human-readable summary."""
        head = "OK  " if self.ok else "FAIL"
        lines = [f"{head} {self.episode_dir} (episode {self.episode_id!r}, "
                 f"{len(self.found_issues)} issues found, truth {self.truth_issue_count}, "
                 f"injections {self.injection_count})"]
        lines.append("  " + "  ".join(f"{c}={'yes' if v else 'NO'}" for c, v in self.conditions.items()))
        for problem in self.problems[:max_problems]:
            lines.append(f"  - [{problem.condition}] {problem.code}: {problem.message}")
        if len(self.problems) > max_problems:
            lines.append(f"  ... {len(self.problems) - max_problems} more problems")
        return "\n".join(lines)

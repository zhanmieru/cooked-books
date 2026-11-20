"""Metrics and reward from TP/FP/FN (EVALUATION_CONTRACT.md, section 6).

All arithmetic is exact (`Fraction`). `to_output` is the only place where
values become `float`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

STEP_COST = Fraction(1, 500)  # 0.002 per inspection step
RETURN_MIN = Fraction(-1)
RETURN_MAX = Fraction(1)


@dataclass(frozen=True)
class Metrics:
    """Exact metrics of one scored submission."""

    tp: int
    fp: int
    fn: int
    precision: Fraction
    recall: Fraction
    f1: Fraction
    ret: Fraction


def _ratio(numerator: int, denominator: int) -> Fraction:
    """numerator / denominator, or 0 when the denominator is 0."""
    return Fraction(numerator, denominator) if denominator else Fraction(0)


def f1_score(tp: int, fp: int, fn: int) -> Fraction:
    """2TP / (2TP + FP + FN). Empty truth and empty findings (all zero) give 1."""
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return Fraction(1)
    return Fraction(2 * tp, denominator)


def episode_return(f1: Fraction, inspection_steps: int) -> Fraction:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)."""
    raw = 2 * f1 - 1 - STEP_COST * inspection_steps
    return max(RETURN_MIN, min(RETURN_MAX, raw))


def compute_metrics(tp: int, fp: int, fn: int, inspection_steps: int) -> Metrics:
    """Precision and recall (0 on a zero denominator), F1 and the clipped return."""
    f1 = f1_score(tp, fp, fn)
    return Metrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=f1,
        ret=episode_return(f1, inspection_steps),
    )


def to_output(metrics: Metrics) -> dict:
    """The seven-key result of `score_episode`: ints for counts, floats for the rest."""
    return {
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "return": float(metrics.ret),
    }


def structural_error_output() -> dict:
    """The result for a structurally invalid submission."""
    return {
        "tp": None,
        "fp": None,
        "fn": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "return": float(RETURN_MIN),
    }

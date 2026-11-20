"""Exact metrics and reward (EVALUATION_CONTRACT.md, section 6)."""

from __future__ import annotations

import ast
from fractions import Fraction
from pathlib import Path

import pytest

import cookedbooks.grader.reward as reward_module
from cookedbooks.grader.reward import (
    STEP_COST,
    compute_metrics,
    episode_return,
    f1_score,
    structural_error_output,
    to_output,
)


def test_step_cost_is_exact():
    assert STEP_COST == Fraction(2, 1000)


@pytest.mark.parametrize(("tp", "fp", "fn", "f1"), [
    (1, 0, 0, Fraction(1)),
    (0, 0, 1, Fraction(0)),
    (0, 0, 0, Fraction(1)),   # empty truth and empty answer
    (0, 1, 0, Fraction(0)),   # empty truth with findings
    (1, 1, 0, Fraction(2, 3)),
    (1, 19, 0, Fraction(2, 21)),
    (1, 1, 1, Fraction(1, 2)),
])
def test_f1(tp, fp, fn, f1):
    assert f1_score(tp, fp, fn) == f1


def test_empty_truth_and_answer_has_zero_precision_and_recall():
    m = compute_metrics(0, 0, 0, 10)
    assert (m.precision, m.recall, m.f1, m.ret) == (0, 0, 1, Fraction(49, 50))


def test_zero_denominators_give_zero():
    m = compute_metrics(0, 0, 3, 0)
    assert (m.precision, m.recall) == (0, 0)
    m = compute_metrics(0, 2, 0, 0)
    assert (m.precision, m.recall, m.f1) == (0, 0, 0)


@pytest.mark.parametrize(("f1", "steps", "ret"), [
    (Fraction(1), 0, Fraction(1)),
    (Fraction(1), 10, Fraction(98, 100)),
    (Fraction(1), 50, Fraction(9, 10)),
    (Fraction(1), 100, Fraction(8, 10)),
    (Fraction(1), 1000, Fraction(-1)),      # clipped
    (Fraction(0), 0, Fraction(-1)),
    (Fraction(0), 10, Fraction(-1)),        # clipped
    (Fraction(2, 3), 10, Fraction(47, 150)),
    (Fraction(2, 21), 10, Fraction(-871, 1050)),
    (Fraction(1, 2), 10, Fraction(-1, 50)),
])
def test_return(f1, steps, ret):
    assert episode_return(f1, steps) == ret


def test_output_types():
    out = to_output(compute_metrics(1, 1, 0, 10))
    assert list(out) == ["tp", "fp", "fn", "precision", "recall", "f1", "return"]
    assert all(type(out[k]) is int for k in ("tp", "fp", "fn"))
    assert all(type(out[k]) is float for k in ("precision", "recall", "f1", "return"))
    assert out["return"] == float(Fraction(47, 150))


def test_structural_error_output():
    assert structural_error_output() == {"tp": None, "fp": None, "fn": None, "precision": None,
                                         "recall": None, "f1": None, "return": -1.0}


def test_no_float_literals_in_reward_arithmetic():
    tree = ast.parse(Path(reward_module.__file__).read_text(encoding="utf-8"))
    floats = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == []

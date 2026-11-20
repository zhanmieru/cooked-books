"""Tests for cookedbooks.env.calculator."""

from __future__ import annotations

import ast
import time

import pytest

from cookedbooks.env import calculator
from cookedbooks.env.calculator import CalcError, evaluate


@pytest.mark.parametrize("expression, expected", [
    ("0.1+0.2", "0.3"),
    ("1+2", "3"),
    ("10/4", "2.5"),
    ("10/5", "2"),
    ("1.50*2", "3"),
    ("2.50", "2.5"),
    ("100", "100"),
    ("1000000*1000000", "1000000000000"),
    ("(1+2)*3", "9"),
    ("1+2*3", "7"),
    ("8/4/2", "1"),
    ("10-2-3", "5"),
    ("-5", "-5"),
    ("+5", "5"),
    ("-(-1)", "1"),
    ("2*-0.5", "-1"),
    ("--1", "1"),
    ("-5.0-0.5", "-5.5"),
    (".5+1.", "1.5"),
    ("0.000001", "0.000001"),
    ("  1 +\t2 ", "3"),
    ("((((((((((1))))))))))", "1"),
    ("47025000/100000", "470.25"),
    ("100000*470.25", "47025000"),
    ("123456789012345678901234567890*10", "1234567890123456789012345678900"),
    ("9" * 50, "9" * 50),
    ("1." + "0" * 80, "1"),
])
def test_accepted(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize("expression", ["0", "-0", "0.0", "-0.000", "1-1", "-(0)", "0*-5", "-1+1"])
def test_zero_is_plain_zero(expression):
    assert evaluate(expression) == "0"


def test_inexact_division_rounds_to_50_digits():
    assert evaluate("1/3") == "0." + "3" * 50
    assert evaluate("2/3") == "0." + "6" * 49 + "7"
    assert evaluate("1/3*7") == "2." + "3" * 49


@pytest.mark.parametrize("expression", [
    "1e5", "1E5", "1.5e-3", "0x10", "0o7", "0b1", "1_000", "1j", "2J", "007",
    "1..2", "1.2.3", ".", "",
])
def test_rejects_non_plain_literals(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize("expression", [
    "2**3", "7//2", "7%2", "1<2", "1==1", "1&1", "1|1", "1^1", "~1", "1<<2", "not 1",
    "1 if 1 else 2", "1 and 2",
])
def test_rejects_other_operators(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize("expression", [
    "x", "abs(1)", "__import__('os').system('true')", "open('/etc/passwd').read()",
    "(1).real", "'1'", '"1"', "b'1'", "(1,2)", "[1]", "{1}", "{1: 2}", "True", "None",
    "lambda: 1", "1; 2", "1 # comment", "1\n+2", "1+1\n", "\\", "x := 1",
    "١+١",  # Arabic-Indic digits
    "1 + 2 ,",
])
def test_rejects_names_calls_and_other_syntax(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize("expression", ["1/0", "1/(2-2)", "0/0", "1/0.0", "5/-0"])
def test_division_by_zero(expression):
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize("value", [None, 1, 1.5, b"1+1", ["1"], {"e": "1"}, True])
def test_rejects_non_string_input(value):
    with pytest.raises(CalcError):
        evaluate(value)


def test_length_limit():
    ok = "+".join(["1"] * 100)  # 199 characters
    assert len(ok) == 199
    assert evaluate(ok) == "100"
    assert evaluate("1" + " " * 199) == "1"  # exactly 200
    with pytest.raises(CalcError):
        evaluate("1" + " " * 200)  # 201


def test_literal_must_fit_exactly():
    with pytest.raises(CalcError):
        evaluate("9" * 51)


@pytest.mark.parametrize("expression", [
    "*".join(["9" * 20] * 9),                       # result far above the exponent bound
    "*".join(["99999999999999"] * 13),
    "1/" + "/".join(["1" + "0" * 20] * 6),          # result far below the exponent bound
    "(" * 70 + "1" + ")" * 70 + "*" + "9" * 50,     # deep nesting, still bounded
])
def test_magnitude_is_bounded(expression):
    assert len(expression) <= 200
    started = time.monotonic()
    try:
        result = evaluate(expression)
    except CalcError:
        result = None
    assert time.monotonic() - started < 1.0
    if result is not None:
        assert len(result) < 200


def test_deep_nesting_does_not_crash():
    for expression in ("(" * 100 + "1" + ")" * 100, "-" * 199 + "1", "+" * 199 + "1"):
        try:
            evaluate(expression)
        except CalcError:
            pass


def test_output_never_uses_exponent():
    for expression in ("1000000000000000000000000000000*1000000000000000000000000000000",
                       "0.0000000000000000000001", "100/1", "1200"):
        out = evaluate(expression)
        assert "e" not in out.lower()
        assert not (("." in out) and out.endswith("0"))


def test_one_exception_type():
    assert issubclass(CalcError, Exception)
    for bad in ("x", "1/0", "2**2", 5):
        with pytest.raises(CalcError):
            evaluate(bad)


def test_no_eval_or_exec_in_source():
    tree = ast.parse(open(calculator.__file__, encoding="utf-8").read())
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__"}

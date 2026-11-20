"""The `calculator` tool (EVALUATION_CONTRACT.md, section 3).

Grammar: decimal numbers, `+ - * /`, parentheses, unary minus and plus.
The expression is parsed with `ast` and walked against a whitelist of node
types; nothing is ever evaluated by Python itself (no eval, exec or
compile). Every literal is converted to `Decimal` from its source text, so
`0.1+0.2` is exactly `"0.3"`.

Bounds: at most 200 characters; arithmetic runs in a private decimal
context with 50 significant digits and exponents within +-99, so no
expression can blow up in time or memory. A literal must fit in 50 digits
exactly. A result that needs more digits is rounded half-even to 50
significant digits (as in `1/3`); a result outside the exponent range is
rejected, never silently turned into zero or infinity.
"""

from __future__ import annotations

import ast
import decimal
import re
from decimal import Decimal

MAX_LENGTH = 200
PRECISION = 50
MAX_EXPONENT = 99

# Only these characters may appear at all: digits, point, operators, parentheses, blanks.
_ALLOWED_CHARS = re.compile(r"[0-9.+\-*/() \t]*")
# A literal's source text: plain decimal only (no exponent, hex, underscore, j suffix).
_DECIMAL_LITERAL = re.compile(r"[0-9]+(\.[0-9]*)?|\.[0-9]+")

_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_UNARY_OPS = (ast.USub, ast.UAdd)


class CalcError(Exception):
    """Raised for any expression the calculator does not accept."""


def _context() -> decimal.Context:
    return decimal.Context(
        prec=PRECISION,
        rounding=decimal.ROUND_HALF_EVEN,
        Emax=MAX_EXPONENT,
        Emin=-MAX_EXPONENT,
        traps=[
            decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow,
            decimal.Underflow, decimal.Subnormal,
        ],
    )


def evaluate(expression: str) -> str:
    """Evaluate an arithmetic expression and return a plain decimal string.

    Raises CalcError for anything outside the grammar, division by zero, or
    a result too large or too small for the bounded context.
    """
    if not isinstance(expression, str) or len(expression) > MAX_LENGTH:
        raise CalcError
    if not _ALLOWED_CHARS.fullmatch(expression):
        raise CalcError
    # ast.parse rejects leading blanks ("unexpected indent"); blanks carry no meaning here.
    expression = expression.strip(" \t")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise CalcError from None
    ctx = _context()
    try:
        value = _walk(tree.body, expression, ctx)
        return _format(value, ctx)
    except (decimal.DecimalException, RecursionError, ArithmeticError):
        raise CalcError from None


def _walk(node: ast.AST, source: str, ctx: decimal.Context) -> Decimal:
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY_OPS):
        left = _walk(node.left, source, ctx)
        right = _walk(node.right, source, ctx)
        if isinstance(node.op, ast.Add):
            return ctx.add(left, right)
        if isinstance(node.op, ast.Sub):
            return ctx.subtract(left, right)
        if isinstance(node.op, ast.Mult):
            return ctx.multiply(left, right)
        if right.is_zero():
            raise CalcError
        return ctx.divide(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
        operand = _walk(node.operand, source, ctx)
        return ctx.minus(operand) if isinstance(node.op, ast.USub) else ctx.plus(operand)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        text = ast.get_source_segment(source, node)
        if text is None or not _DECIMAL_LITERAL.fullmatch(text):
            raise CalcError
        # A literal must fit the context exactly; only results of operations round.
        exact = ctx.copy()
        exact.traps[decimal.Inexact] = True
        return exact.create_decimal(text)
    raise CalcError


def _format(value: Decimal, ctx: decimal.Context) -> str:
    """Plain notation, no trailing zeros, integers without a point, never "-0"."""
    if not value.is_finite():
        raise CalcError
    if value.is_zero():
        return "0"
    return format(value.normalize(ctx), "f")

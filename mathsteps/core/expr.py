"""Shared validation for user-supplied expressions and evaluated values."""
from __future__ import annotations

import math
from typing import Any, Iterable

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr as _sympy_parse_expr,
    standard_transformations,
)

# ``x^2`` -> ``x**2`` and ``2x`` -> ``2*x``. Symbol splitting is deliberately
# NOT enabled: it would read ``theta`` as t*h*e*t*a.
_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication)


def parse_expr(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    """``sympify`` ``text`` and reject symbols outside ``symbols``.

    A free constant such as ``k`` in ``"k*x - 3"`` would otherwise only fail
    later with an opaque ``TypeError`` (or, for ``e``, silently mean a
    symbol instead of Euler's number).
    """
    try:
        expr = sp.sympify(
            _sympy_parse_expr(str(text), local_dict=dict(symbols), transformations=_TRANSFORMATIONS)
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {text!r} as a math expression: {exc}") from exc
    check_symbols(expr, symbols.values(), text)
    return expr


def check_symbols(expr: sp.Basic, allowed: Iterable[sp.Symbol], text: str) -> None:
    allowed = list(allowed)
    extra = sorted(str(s) for s in expr.free_symbols - set(allowed))
    if not extra:
        return
    hint = " (use E for Euler's number)" if "e" in extra else ""
    allowed_names = ", ".join(sorted(str(s) for s in allowed))
    raise ValueError(
        f"Unknown symbol(s) {extra} in {text!r}{hint}. Only {allowed_names} may appear; "
        f"substitute numbers for any constants."
    )


def finite_float(value: Any, what: str) -> float:
    """Convert ``value`` to a finite ``float`` or raise a descriptive ``ValueError``."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not a real number: {value!r}.") from exc
    if not math.isfinite(v):
        raise ValueError(
            f"{what} is not finite ({v}); the function is undefined or singular there."
        )
    return v

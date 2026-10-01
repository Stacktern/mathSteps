"""Shared utilities for root-finding solvers.

A root-finding ``problem`` dict has the common shape::

    {
        "type": "root_finding",
        "function": "x**3 - x - 2",   # sympy-compatible expression in `variable`
        "variable": "x",
        "tol": 1e-10,                 # optional, default 1e-10
        "max_iter": 50                # optional, default 50
    }

Each method may add its own keys (e.g. ``a``/``b`` for bisection,
``x0`` for Newton-Raphson).
"""
from __future__ import annotations

from typing import Callable

import sympy as sp

from mathsteps.core.expr import parse_expr


def parse_function(expr_str: str, var_str: str) -> tuple[sp.Expr, sp.Symbol, Callable[[float], float]]:
    """Parse ``expr_str`` as a SymPy expression and return ``(expr, var, f_numeric)``.

    ``f_numeric`` is a fast Python callable for evaluating ``expr`` at
    floats — avoids the cost of going through SymPy lambdify each
    iteration.
    """
    var = sp.symbols(var_str)
    expr = parse_expr(expr_str, {var_str: var})
    f_numeric = sp.lambdify(var, expr, modules=["numpy"])
    return expr, var, f_numeric


def get_tol(problem: dict, default: float = 1e-10) -> float:
    return float(problem.get("tol", default))


def get_max_iter(problem: dict, default: int = 50) -> int:
    return int(problem.get("max_iter", default))

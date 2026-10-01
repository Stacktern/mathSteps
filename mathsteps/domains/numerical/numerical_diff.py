"""Numerical differentiation via finite differences.

Problem schema::

    {
        "type": "numerical_diff",
        "function": "sin(x)",
        "variable": "x",
        "x": 1.0,
        "h": 0.001,
        "method": "forward" | "backward" | "central"   # default central
    }

Returns the numerical derivative ``f'(x)`` plus per-step work; the
"step" here is each finite-difference formula application.
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import parse_function


@register
class NumericalDifferentiationSolver(Solver):
    name = "numerical_diff"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "numerical_diff"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, f = parse_function(problem["function"], problem["variable"])
        x = float(problem["x"])
        h = float(problem["h"])
        method = problem.get("method", "central")
        if not h > 0:
            raise ValueError(f"Step size h must be positive; got h = {h}.")
        needed = {
            "forward": (x, x + h),
            "backward": (x - h, x),
            "central": (x - h, x + h),
        }.get(method, ())
        for point in needed:
            finite_float(f(point), f"f({point})")

        if method == "forward":
            num = (f(x + h) - f(x)) / h
            formula = "f'(x) ≈ (f(x+h) - f(x)) / h"
            fxh, fxx = float(f(x + h)), float(f(x))
        elif method == "backward":
            num = (f(x) - f(x - h)) / h
            formula = "f'(x) ≈ (f(x) - f(x-h)) / h"
            fxx, fxh = float(f(x)), float(f(x - h))
        elif method == "central":
            num = (f(x + h) - f(x - h)) / (2 * h)
            formula = "f'(x) ≈ (f(x+h) - f(x-h)) / (2h)"
            fxmh, fxx, fxh = float(f(x - h)), float(f(x)), float(f(x + h))
        else:
            raise ValueError(f"Unknown method {method!r}; expected forward/backward/central.")

        sym_deriv = sp.diff(expr, var)
        try:
            exact = float(sym_deriv.subs(var, sp.nsimplify(x)))
        except (TypeError, ValueError):
            exact = None  # derivative undefined / infinite at this point

        steps: list[Step] = [
            Step(
                description=(
                    f"Numerical differentiation ({method}) of f(x) = {expr} "
                    f"at x = {x}, step h = {h}."
                ),
                before="",
                after=f"formula = {formula}",
            ),
            Step(
                description=(
                    f"Apply formula: f(x-h)={float(f(x - h)) if method != 'forward' else 'n/a'}, "
                    f"f(x)={float(f(x))}, f(x+h)={float(f(x + h))}; "
                    f"f'(x) ≈ {float(num)}."
                ),
                before="",
                after=f"f'({x}) ≈ {float(num)}",
                data={"x": x, "h": h, "method": method, "numerical": float(num)},
            ),
            Step(
                description=(
                    f"Exact derivative: f'(x) = {sym_deriv}; "
                    f"f'({x}) = {exact}. Absolute error ≈ {abs(float(num) - exact):.3e}."
                    if exact is not None
                    else f"Exact derivative f'(x) = {sym_deriv} is undefined at x = {x}."
                ),
                before="",
                after=f"f'({x}) = {exact}" if exact is not None else "f'(x) undefined here",
            ),
        ]
        return steps, float(num)

"""Numerical integration: trapezoidal rule and Simpson's rule.

Problem schema::

    {
        "type": "numerical_integration",
        "function": "sin(x)",
        "variable": "x",
        "a": 0, "b": 3.141592653589793,
        "n": 100,                 # sub-intervals; n even for Simpson
        "method": "trapezoidal" | "simpson"   # default trapezoidal
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import parse_function


@register
class NumericalIntegrationSolver(Solver):
    name = "numerical_integration"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "numerical_integration"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, f = parse_function(problem["function"], problem["variable"])
        a = float(problem["a"])
        b = float(problem["b"])
        n = int(problem["n"])
        method = problem.get("method", "trapezoidal")
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if method == "simpson" and n % 2 != 0:
            raise ValueError("Simpson's rule requires an even number of sub-intervals.")

        h = (b - a) / n
        xs = np.linspace(a, b, n + 1)
        ys = np.array(
            [
                finite_float(
                    f(x),
                    f"The integrand f({x}) (sampled at an endpoint or node)",
                )
                for x in xs
            ],
            dtype=float,
        )

        if method == "trapezoidal":
            integral = h * (0.5 * ys[0] + 0.5 * ys[-1] + ys[1:-1].sum())
            formula = (
                f"integral ≈ h * ( (y0 + yn)/2 + sum_{{i=1}}^{{n-1}} y_i ), "
                f"h = (b - a) / n = {h}"
            )
        elif method == "simpson":
            integral = (h / 3.0) * (
                ys[0] + ys[-1] + 4.0 * ys[1:-1:2].sum() + 2.0 * ys[2:-1:2].sum()
            )
            formula = (
                f"integral ≈ h/3 * (y0 + yn + 4*sum_odd + 2*sum_even), "
                f"h = (b - a) / n = {h}"
            )
        else:
            raise ValueError(f"Unknown method {method!r}; expected trapezoidal/simpson.")

        sym_int = sp.integrate(expr, (var, a, b))
        exact = float(sym_int)

        steps: list[Step] = [
            Step(
                description=(
                    f"Numerical integration ({method}) of f(x) = {expr} "
                    f"on [{a}, {b}] with n = {n}, h = {h}."
                ),
                before="",
                after=f"n = {n}, h = {h}",
            ),
            Step(
                description=(
                    f"Composite formula: {formula}; "
                    f"sum over {n} sub-interval{'s' if n != 1 else ''}."
                ),
                before="",
                after=f"integral ≈ {integral}",
            ),
            Step(
                description=(
                    f"Exact integral: ∫ f dx from {a} to {b} = {sym_int} ≈ {exact}. "
                    f"Absolute error ≈ {abs(float(integral) - exact):.3e}."
                ),
                before="",
                after=f"integral = {exact}",
            ),
        ]
        return steps, float(integral)

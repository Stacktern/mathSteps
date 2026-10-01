"""Newton-Raphson method for root finding.

Problem schema::

    {
        "type": "root_finding",
        "method": "newton_raphson",
        "function": "x**3 - x - 2",
        "variable": "x",
        "x0": 1.5,
        "tol": 1e-10,           # optional
        "max_iter": 50          # optional
    }

Derivative ``f'`` is computed symbolically with SymPy, then evaluated
at each iteration with NumPy (fast). Iteration math is float-based;
SymPy is used only for display.
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import get_max_iter, get_tol, parse_function


@register
class NewtonRaphsonSolver(Solver):
    name = "newton_raphson"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "root_finding"
            and problem.get("method") == "newton_raphson"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, f = parse_function(problem["function"], problem["variable"])
        fprime_expr = sp.diff(expr, var)
        fprime = sp.lambdify(var, fprime_expr, modules=["numpy"])

        x = float(problem["x0"])
        tol = get_tol(problem)
        max_iter = get_max_iter(problem)

        steps: list[Step] = [
            Step(
                description=(
                    f"Newton-Raphson on f(x) = {expr}, "
                    f"f'(x) = {fprime_expr}, starting at x0 = {x}."
                ),
                before="",
                after=f"x0 = {x}",
            )
        ]

        for k in range(1, max_iter + 1):
            fx = finite_float(f(x), f"f({x})")
            fpx = finite_float(fprime(x), f"f'({x})")
            if fpx == 0.0:
                raise ValueError(
                    f"f'(x) = 0 at x = {x}; Newton-Raphson cannot continue."
                )
            x_new = x - fx / fpx
            delta = abs(x_new - x)
            steps.append(
                Step(
                    description=(
                        f"Iteration {k}: x = {x}, f(x) = {fx}, f'(x) = {fpx}; "
                        f"x_new = x - f(x)/f'(x) = {x_new}; |delta x| = {delta}."
                    ),
                    before=f"x = {x}",
                    after=f"x = {x_new}",
                    data={"iter": k, "x": x, "fx": fx, "fpx": fpx, "delta": delta},
                )
            )
            if delta <= tol:
                steps.append(
                    Step(
                        description=f"Converged: |delta x| = {delta} <= tol = {tol}.",
                        before="",
                        after=f"x = {x_new}",
                    )
                )
                x_final = float(x_new)
                steps.append(
                    Step(
                        description="Final root.",
                        before="",
                        after=f"x = {x_final}",
                        data={"answer": x_final, "converged": True},
                    )
                )
                return steps, x_final
            x = x_new

        x_final = float(x)
        steps.append(
            Step(
                description=(
                    f"Did NOT converge: reached max_iter={max_iter} with "
                    f"|delta x| > tol = {tol}; returning last iterate."
                ),
                before="",
                after=f"x ≈ {x_final}",
                data={"converged": False},
            )
        )
        return steps, x_final

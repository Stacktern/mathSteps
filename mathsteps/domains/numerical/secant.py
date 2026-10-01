"""Secant method for root finding.

Problem schema::

    {
        "type": "root_finding", "method": "secant",
        "function": "x**3 - x - 2", "variable": "x",
        "x0": 1.0, "x1": 2.0,           # two initial guesses
        "tol": 1e-10, "max_iter": 50
    }

Like Newton-Raphson but uses a finite-difference slope between the two
most recent iterates instead of the analytical derivative.
"""
from __future__ import annotations

from typing import Any


from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import get_max_iter, get_tol, parse_function


@register
class SecantSolver(Solver):
    name = "secant"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "root_finding"
            and problem.get("method") == "secant"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, f = parse_function(problem["function"], problem["variable"])
        tol = get_tol(problem)
        max_iter = get_max_iter(problem)

        x_prev = float(problem["x0"])
        x_curr = float(problem["x1"])
        f_prev = finite_float(f(x_prev), f"f({x_prev})")
        f_curr = finite_float(f(x_curr), f"f({x_curr})")

        steps: list[Step] = [
            Step(
                description=(
                    f"Secant method on f(x) = {expr}, "
                    f"starting from x0 = {x_prev}, x1 = {x_curr}."
                ),
                before="",
                after=f"[x0, x1] = [{x_prev}, {x_curr}]",
            )
        ]

        for k in range(1, max_iter + 1):
            denom = f_curr - f_prev
            if denom == 0.0:
                raise ValueError(
                    f"f(x_curr) - f(x_prev) = 0 at iteration {k}; secant cannot continue."
                )
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            delta = abs(x_new - x_curr)
            steps.append(
                Step(
                    description=(
                        f"Iteration {k}: x_prev = {x_prev}, x_curr = {x_curr}; "
                        f"f(x_prev) = {f_prev}, f(x_curr) = {f_curr}; "
                        f"x_new = {x_new}; |delta x| = {delta}."
                    ),
                    before=f"x_curr = {x_curr}",
                    after=f"x_new = {x_new}",
                    data={"iter": k, "x_prev": x_prev, "x_curr": x_curr, "delta": delta},
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
                        "Final root.", "", f"x = {x_final}",
                        data={"answer": x_final, "converged": True},
                    )
                )
                return steps, x_final
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            f_curr = finite_float(f(x_curr), f"f({x_curr})")

        x_final = float(x_curr)
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

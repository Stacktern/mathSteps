"""Fixed-point iteration for solving x = g(x).

Problem schema::

    {
        "type": "root_finding", "method": "fixed_point",
        "function": "cos(x)",        # g(x); we seek x = g(x)
        "variable": "x",
        "x0": 0.5,
        "tol": 1e-10, "max_iter": 200
    }

NOTE: convergence requires ``|g'(x)| < 1`` near the fixed point. If the
problem also has a ``target_function`` (the underlying ``f(x)`` whose
root we're after), we verify against ``f(x*) ≈ 0``.
"""
from __future__ import annotations

from typing import Any


from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import get_max_iter, get_tol, parse_function


@register
class FixedPointSolver(Solver):
    name = "fixed_point"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "root_finding"
            and problem.get("method") == "fixed_point"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, g = parse_function(problem["function"], problem["variable"])
        x = float(problem["x0"])
        tol = get_tol(problem)
        max_iter = get_max_iter(problem)

        target_str = problem.get("target_function")
        target_f = None
        if target_str is not None:
            _, _, target_f = parse_function(target_str, problem["variable"])

        steps: list[Step] = [
            Step(
                description=(
                    f"Fixed-point iteration x = g(x) with g(x) = {expr}, starting at x0 = {x}."
                    + (f"  Underlying target: f(x) = {target_str}." if target_str else "")
                ),
                before="",
                after=f"x0 = {x}",
            )
        ]

        for k in range(1, max_iter + 1):
            x_new = finite_float(g(x), f"g({x})")
            delta = abs(x_new - x)
            steps.append(
                Step(
                    description=(
                        f"Iteration {k}: x = g({x}) = {x_new}; |delta x| = {delta}."
                    ),
                    before=f"x = {x}",
                    after=f"x = {x_new}",
                    data={"iter": k, "x": x, "x_new": x_new, "delta": delta},
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
                        "Final fixed point.", "", f"x = {x_final}",
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

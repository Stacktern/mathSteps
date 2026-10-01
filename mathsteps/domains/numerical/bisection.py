"""Bisection method for root finding.

Problem schema::

    {
        "type": "root_finding",
        "method": "bisection",
        "function": "x**3 - x - 2",
        "variable": "x",
        "a": 1.0, "b": 2.0,
        "tol": 1e-10,           # optional
        "max_iter": 50          # optional
    }

Requires ``f(a) * f(b) < 0`` (sign change on the bracket).

Iteration math uses NumPy floats (fast, well-behaved). SymPy is used
only for symbolic display: the exact ``expr`` in the intro step and
the final rationalised root.
"""
from __future__ import annotations

from typing import Any


from mathsteps.core.expr import finite_float
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import get_max_iter, get_tol, parse_function


@register
class BisectionSolver(Solver):
    name = "bisection"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "root_finding"
            and problem.get("method", "bisection") == "bisection"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        expr, var, f = parse_function(problem["function"], problem["variable"])
        a = float(problem["a"])
        b = float(problem["b"])
        tol = get_tol(problem)
        max_iter = get_max_iter(problem)

        fa = finite_float(f(a), f"f({a})")
        fb = finite_float(f(b), f"f({b})")
        if fa == 0.0:
            return [Step("a is already a root.", "", f"x = {a}", data={"converged": True})], a
        if fb == 0.0:
            return [Step("b is already a root.", "", f"x = {b}", data={"converged": True})], b
        if fa * fb > 0:
            raise ValueError(
                f"Bisection requires f(a) and f(b) to have opposite signs; "
                f"got f({a})={fa}, f({b})={fb}."
            )

        steps: list[Step] = [
            Step(
                description=(
                    f"Bisection on [{a}, {b}] for f(x) = {expr}. "
                    f"f({a}) = {fa}, f({b}) = {fb} (opposite signs)."
                ),
                before="",
                after=f"bracket = [{a}, {b}]",
            )
        ]

        x_root = a
        for k in range(1, max_iter + 1):
            mid = 0.5 * (a + b)
            fmid = finite_float(f(mid), f"f({mid})")
            width = abs(b - a)
            steps.append(
                Step(
                    description=(
                        f"Iteration {k}: midpoint m = ({a}+{b})/2 = {mid}; "
                        f"f(m) = {fmid}; bracket width = {width}."
                    ),
                    before=f"[{a}, {b}]",
                    after=f"m = {mid}, f(m) = {fmid}",
                    data={"iter": k, "a": a, "b": b, "m": mid, "fm": fmid},
                )
            )
            if fmid == 0.0 or width <= tol:
                x_root = mid
                steps.append(
                    Step(
                        description=(
                            f"Stopping: {'f(m) = 0 exactly' if fmid == 0.0 else f'bracket width {width} <= tol {tol}'}."
                        ),
                        before="",
                        after=f"x = {x_root}",
                    )
                )
                x_final = float(x_root)
                steps.append(
                    Step(
                        description="Final root.",
                        before="",
                        after=f"x = {x_final}",
                        data={"answer": x_final, "converged": True},
                    )
                )
                return steps, x_final
            if fa * fmid < 0:
                old_b = b
                b = mid
                steps.append(
                    Step(
                        description=(
                            f"f(a) and f(m) have opposite signs -> root is in [{a}, {mid}]; "
                            f"set b = mid."
                        ),
                        before=f"b = {old_b}",
                        after=f"b = {mid}",
                    )
                )
            else:
                old_a = a
                a = mid
                fa = fmid
                steps.append(
                    Step(
                        description=(
                            f"f(a) and f(m) have the same sign -> root is in [{mid}, {b}]; "
                            f"set a = mid (and update f(a))."
                        ),
                        before=f"a = {old_a}",
                        after=f"a = {mid}",
                    )
                )
            x_root = mid

        x_final = float(x_root)
        steps.append(
            Step(
                description=(
                    f"Did NOT converge: reached max_iter={max_iter} with bracket "
                    f"width > tol = {tol}; returning best midpoint."
                ),
                before="",
                after=f"x ≈ {x_final}",
                data={"converged": False},
            )
        )
        return steps, x_final

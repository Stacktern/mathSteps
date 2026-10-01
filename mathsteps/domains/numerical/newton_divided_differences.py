"""Newton's divided-differences interpolation.

Uses the Newton form ``P(x) = f[x_0] + f[x_0,x_1](x-x_0) + ... +
f[x_0,...,x_n](x-x_0)...(x-x_{n-1})``. The divided-difference table
is the natural "step" output for this method.

Problem schema::

    {
        "type": "interpolation",
        "method": "newton_divided_differences",
        "points": [[x0, y0], [x1, y1], ...]
    }
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from mathsteps.core.exact import to_fraction, to_sympy
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step


@register
class NewtonDividedDifferencesSolver(Solver):
    name = "newton_divided_differences"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "interpolation"
            and problem.get("method") == "newton_divided_differences"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        # Exact rationals throughout (no float round-off in the table).
        raw = [(to_fraction(p[0]), to_fraction(p[1])) for p in problem["points"]]
        n = len(raw)
        if n < 2:
            raise ValueError("Need at least 2 points.")
        if len({x for x, _ in raw}) != n:
            raise ValueError("Interpolation needs distinct x-coordinates; found duplicates.")
        pts = [(to_sympy(x), to_sympy(y)) for x, y in raw]

        x_var = sp.symbols("x")
        exact_table = [[y for _, y in raw]]
        for order in range(1, n):
            prev = exact_table[-1]
            exact_table.append([
                (prev[i + 1] - prev[i]) / (raw[i + order][0] - raw[i][0])
                for i in range(n - order)
            ])
        table: list[list[sp.Expr]] = [[to_sympy(v) for v in row] for row in exact_table]

        steps: list[Step] = [
            Step(
                description=(
                    f"Newton divided-differences interpolation through {n} points: "
                    + ", ".join(f"({xi}, {yi})" for xi, yi in pts) + "."
                ),
                before="",
                after=f"order 0 coefficients: {[str(v) for v in table[0]]}",
            )
        ]
        for order, row in enumerate(table[1:], 1):
            steps.append(
                Step(
                    description=(
                        f"Order {order} divided differences: "
                        + ", ".join(f"f[{','.join(['x'+str(k) for k in range(i, i+order+1)])}] = {v}" for i, v in enumerate(row))
                        + "."
                    ),
                    before="",
                    after=f"order {order} = {[str(v) for v in row]}",
                    data={"order": order, "row": [v for v in row]},
                )
            )

        poly = sp.Integer(0)
        term = sp.Integer(1)
        for order, coeff in enumerate([table[o][0] for o in range(n)]):
            if order > 0:
                term = term * (x_var - pts[order - 1][0])
            poly = poly + coeff * term

        steps.append(
            Step(
                description=(
                    f"Newton form: P(x) = "
                    + " + ".join(
                        f"{table[o][0]}*{'1' if o == 0 else '*'.join([f'(x-{pts[k][0]})' for k in range(o)])}"
                        for o in range(n)
                    )
                    + f" = {sp.expand(poly)}."
                ),
                before="",
                after=f"P(x) = {sp.expand(poly)}",
                data={"answer": poly},
            )
        )
        return steps, poly

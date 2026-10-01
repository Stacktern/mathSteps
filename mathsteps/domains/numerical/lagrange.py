"""Lagrange interpolation.

Problem schema::

    {
        "type": "interpolation",
        "method": "lagrange",
        "points": [[x0, y0], [x1, y1], ...]
    }

Returns the unique polynomial through the points as a SymPy
expression, with one ``Step`` per basis polynomial L_i(x).
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from mathsteps.core.exact import to_fraction, to_sympy
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step


@register
class LagrangeInterpolationSolver(Solver):
    name = "lagrange"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "interpolation"
            and problem.get("method", "lagrange") == "lagrange"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        # Exact rationals: expanding the polynomial in floating point loses
        # digits catastrophically for realistic abscissae such as years.
        pts = [(to_sympy(to_fraction(p[0])), to_sympy(to_fraction(p[1]))) for p in problem["points"]]
        n = len(pts)
        if n < 2:
            raise ValueError("Need at least 2 points for Lagrange interpolation.")
        xs = [xi for xi, _ in pts]
        if len(set(xs)) != n:
            raise ValueError("Interpolation needs distinct x-coordinates; found duplicates.")

        x = sp.symbols("x")
        steps: list[Step] = [
            Step(
                description=(
                    f"Lagrange interpolation through {n} points: " +
                    ", ".join(f"({xi}, {yi})" for xi, yi in pts) + "."
                ),
                before="",
                after="n = " + str(n),
            )
        ]

        basis_polys = []
        for i, (xi, _) in enumerate(pts):
            num = sp.Integer(1)
            den = sp.Integer(1)
            factors_num = []
            for j, (xj, _) in enumerate(pts):
                if j == i:
                    continue
                num = num * (x - xj)
                den = den * (xi - xj)
                factors_num.append(f"(x - {xj})")
            Li = sp.expand(num / den)
            factors_str = " * ".join(factors_num)
            steps.append(
                Step(
                    description=(
                        f"Basis polynomial L_{i}(x) = ({factors_str}) / ({den}); "
                        f"= {Li}."
                    ),
                    before="",
                    after=f"L_{i}(x) = {Li}",
                    data={"i": i, "Li": Li},
                )
            )
            basis_polys.append(Li)

        poly = sp.expand(sum(yi * Li for (_, yi), Li in zip(pts, basis_polys)))
        steps.append(
            Step(
                description=(
                    "Final interpolating polynomial P(x) = sum_i y_i * L_i(x) = " + str(poly) + "."
                ),
                before="",
                after=f"P(x) = {poly}",
                data={"answer": poly},
            )
        )
        return steps, poly

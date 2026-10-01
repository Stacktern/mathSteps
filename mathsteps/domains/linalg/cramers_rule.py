"""Cramer's rule for solving Ax = b.

Problem schema::

    {
        "type": "cramers_rule",
        "A": [[...], ...],
        "b": [...]
    }

Builds ``det(A)`` and ``det(A_i)`` (A with column i replaced by b),
then computes ``x_i = det(A_i) / det(A)``.
"""
from __future__ import annotations

from typing import Any

import sympy as sp

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .cofactor_determinant import _det_cofactor


@register
class CramersRuleSolver(Solver):
    name = "cramers_rule"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "cramers_rule"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = sp.Matrix(problem["A"])
        b = sp.Matrix(problem["b"])
        n = A.rows
        if n != A.cols or n != b.rows:
            raise ValueError("Cramer's rule requires square A and matching b.")

        det_steps: list[Step] = []
        det_A = _det_cofactor(A, det_steps)
        steps: list[Step] = [
            Step(
                description=(
                    f"Cramer's rule: det(A) = {det_A}; "
                    f"for each i, x_i = det(A_i) / det(A)."
                ),
                before=f"A = {A.tolist()}, b = {b.tolist()}",
                after=f"det(A) = {det_A}",
            )
        ]
        if det_A == 0:
            raise ValueError("det(A) = 0; Cramer's rule not applicable.")

        x = []
        for i in range(n):
            Ai = A.copy()
            Ai[:, i] = b
            det_steps_local: list[Step] = []
            det_Ai = _det_cofactor(Ai, det_steps_local)
            xi = det_Ai / det_A
            x.append(xi)
            steps.append(
                Step(
                    description=(
                        f"det(A_{i}) = {det_Ai}; x_{i} = det(A_{i})/det(A) = {xi}."
                    ),
                    before=f"A_{i} = {Ai.tolist()}",
                    after=f"x_{i} = {xi}",
                    data={"i": i, "det_Ai": det_Ai, "xi": xi},
                )
            )

        steps.append(
            Step(
                description=f"Solution: x = {x}.",
                before="",
                after=f"x = {x}",
                data={"answer": x},
            )
        )
        return steps, x

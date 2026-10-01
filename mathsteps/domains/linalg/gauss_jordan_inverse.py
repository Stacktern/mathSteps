"""Gauss-Jordan elimination: matrix inverse.

Problem schema::

    {
        "type": "matrix_inverse",
        "A": [[...], ...],
        "detail": "full" | "summary" | "none"      # optional
    }

Matrices larger than ``AUTO_FULL_LIMIT`` are summarised by default (one step
per pivot column).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from mathsteps.core.detail import resolve_detail
from mathsteps.core.exact import to_fraction_matrix, to_sympy, to_sympy_matrix
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

AUTO_FULL_LIMIT = 10


@register
class GaussJordanInverseSolver(Solver):
    name = "gauss_jordan_inverse"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "matrix_inverse"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = to_fraction_matrix(problem["A"])
        n = len(A)
        if any(len(row) != n for row in A):
            raise ValueError("matrix_inverse requires a square matrix.")
        mode = resolve_detail(problem, n, AUTO_FULL_LIMIT)
        full = mode == "full"
        aug = [
            row + [Fraction(int(i == j)) for j in range(n)]
            for i, row in enumerate(A)
        ]

        steps: list[Step] = [
            Step(
                description=f"Form augmented matrix [A | I{n}]."
                + ("" if full else f" (detail='{mode}')"),
                before="",
                after="augmented matrix built",
            )
        ]

        for col in range(n):
            pivot_row = col
            best = abs(aug[col][col])
            for r in range(col + 1, n):
                if abs(aug[r][col]) > best:
                    best = abs(aug[r][col])
                    pivot_row = r
            if best == 0:
                raise ValueError("Matrix is singular; no inverse exists.")
            swapped = pivot_row != col
            if swapped:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                if full:
                    steps.append(
                        Step(
                            description=f"Swap row {col} with row {pivot_row} (pivoting).",
                            before="",
                            after="",
                            data={"swap": (col, pivot_row)},
                        )
                    )
            pivot = aug[col][col]
            aug[col] = [v / pivot for v in aug[col]]
            if full:
                steps.append(
                    Step(
                        description=f"Scale row {col} by 1/{pivot} to make pivot = 1.",
                        before="",
                        after="",
                        data={"scale": (col, to_sympy(1 / pivot))},
                    )
                )
            cleared = 0
            for r in range(n):
                if r == col or aug[r][col] == 0:
                    continue
                factor = aug[r][col]
                aug[r] = [a - factor * c for a, c in zip(aug[r], aug[col])]
                cleared += 1
                if full:
                    steps.append(
                        Step(
                            description=(
                                f"Eliminate column {col} in row {r}: "
                                f"R{r} = R{r} - ({factor}) * R{col}."
                            ),
                            before="",
                            after="",
                            data={"row": r, "col": col, "factor": to_sympy(factor)},
                        )
                    )
            if mode == "summary":
                steps.append(
                    Step(
                        description=(
                            f"Column {col}: pivot {pivot}"
                            + (f" (swapped with row {pivot_row})" if swapped else "")
                            + f"; scaled to 1 and cleared {cleared} other row(s)."
                        ),
                        before="",
                        after="",
                        data={"col": col, "pivot": to_sympy(pivot)},
                    )
                )

        inv = to_sympy_matrix([row[n:] for row in aug])
        steps.append(
            Step(
                description="Left side is now I; the right side is A^{-1}.",
                before="",
                after=f"A^-1 = {inv.tolist()}" if full else "",
                data={"answer": inv},
            )
        )
        return steps, inv

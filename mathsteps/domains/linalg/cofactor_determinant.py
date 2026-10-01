"""Determinant via cofactor expansion.

Problem schema::

    {
        "type": "determinant",
        "method": "cofactor",
        "A": [[...], ...]
    }
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

import sympy as sp

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from mathsteps.core.exact import to_fraction_matrix, to_sympy


def _laplace_det(rows: list[list[Fraction]]) -> Fraction:
    """Cofactor (Laplace) expansion along the first remaining row.

    The minors obtained by deleting the same set of columns coincide, so
    they are memoised on the set of remaining columns (a bitmask). That
    keeps the classic expansion but costs O(n * 2**n) instead of O(n!).
    """
    n = len(rows)
    if any(len(r) != n for r in rows):
        raise ValueError("Determinant requires a square matrix.")
    memo: dict[int, Fraction] = {}

    def minor(row: int, mask: int) -> Fraction:
        if row == n:
            return Fraction(1)
        cached = memo.get(mask)
        if cached is not None:
            return cached
        total = Fraction(0)
        sign = 1
        for j in range(n):
            if not mask & (1 << j):
                continue
            entry = rows[row][j]
            if entry != 0:
                total += sign * entry * minor(row + 1, mask & ~(1 << j))
            sign = -sign
        memo[mask] = total
        return total

    return minor(0, (1 << n) - 1)


def _det_cofactor(M, steps: list[Step], prefix: str = "") -> sp.Expr:
    """Determinant of ``M`` (nested sequence or ``sympy.Matrix``) by cofactor expansion."""
    rows = to_fraction_matrix(M)
    if not rows:
        raise ValueError("Determinant requires a non-empty square matrix.")
    return to_sympy(_laplace_det(rows))


@register
class CofactorDeterminantSolver(Solver):
    name = "cofactor_determinant"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "determinant"
            and problem.get("method", "cofactor") == "cofactor"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = sp.Matrix(problem["A"])
        steps: list[Step] = [
            Step(
                description=(
                    f"Cofactor expansion of det(A) along row 0; matrix A = {A.tolist()}."
                ),
                before="",
                after="",
            )
        ]
        det = _det_cofactor(A, steps)
        steps.append(
            Step(
                description=f"Final det(A) = {det}.",
                before="",
                after=f"det(A) = {det}",
                data={"answer": det},
            )
        )
        return steps, det

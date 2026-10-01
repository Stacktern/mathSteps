"""Gaussian elimination solver for linear systems Ax = b.

Problem schema::

    {
        "type": "linear_system",
        "A": [[a11, a12, ...], ...],
        "b": [b1, b2, ...],
        "detail": "full" | "summary" | "none"      # optional
    }

Elimination runs on exact ``fractions.Fraction`` values, so every
factor / result displayed in a step is exact and the cost stays
polynomial in the matrix size. The final answer is returned as a list
of SymPy rationals; verification (``verify.py``) cross-checks against
``numpy.linalg.solve`` (floating-point ground truth).

Matrices larger than ``AUTO_FULL_LIMIT`` x ``AUTO_FULL_LIMIT`` are recorded
in ``summary`` mode by default (one step per pivot column, no matrix
snapshots); see :mod:`mathsteps.core.detail`.
"""
from __future__ import annotations

from typing import Any

from mathsteps.core.detail import resolve_detail
from mathsteps.core.exact import (
    fmt_matrix,
    to_fraction_matrix,
    to_fraction_vector,
    to_sympy,
)
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

AUTO_FULL_LIMIT = 10  # n x n systems above this are summarised by default


@register
class GaussianEliminationSolver(Solver):
    name = "gaussian_elimination"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "linear_system"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = to_fraction_matrix(problem["A"])
        b = to_fraction_vector(problem["b"])
        n_rows = len(A)
        n_cols = len(A[0]) if A else 0
        if n_rows != n_cols:
            raise ValueError(
                f"Gaussian elimination requires a square matrix; got "
                f"{n_rows}x{n_cols}."
            )
        if n_rows != len(b):
            raise ValueError(f"b has length {len(b)} but A has {n_rows} rows.")

        aug = [row + [bi] for row, bi in zip(A, b)]
        n = n_rows
        mode = resolve_detail(problem, n, AUTO_FULL_LIMIT)
        full = mode == "full"
        steps: list[Step] = []

        steps.append(
            Step(
                description="Form the augmented matrix [A | b]."
                + ("" if full else f" ({n} x {n + 1}; detail='{mode}', matrix snapshots omitted)"),
                before=fmt_matrix(aug) if full else "",
                after=fmt_matrix(aug) if full else "",
            )
        )

        for col in range(n):
            pivot_row = col
            best = abs(aug[col][col])
            for r in range(col + 1, n):
                if abs(aug[r][col]) > best:
                    best = abs(aug[r][col])
                    pivot_row = r
            if best == 0:
                raise ValueError(
                    f"Matrix is singular (zero pivot in column {col}); "
                    f"no unique solution exists."
                )
            swapped = pivot_row != col
            if swapped:
                before = fmt_matrix(aug) if full else ""
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
                if full:
                    steps.append(
                        Step(
                            description=(
                                f"Swap row {col} with row {pivot_row} "
                                f"(partial pivoting, |{aug[col][col]}| is largest in column {col})."
                            ),
                            before=before,
                            after=fmt_matrix(aug),
                            data={"swap": (col, pivot_row)},
                        )
                    )
            pivot = aug[col][col]
            eliminated = 0
            for r in range(col + 1, n):
                if aug[r][col] == 0:
                    continue
                factor = aug[r][col] / pivot
                before = fmt_matrix(aug) if full else ""
                aug[r] = [a - factor * c for a, c in zip(aug[r], aug[col])]
                eliminated += 1
                if full:
                    steps.append(
                        Step(
                            description=(
                                f"Eliminate entry in row {r}, column {col}: "
                                f"R{r} = R{r} - ({factor}) * R{col}."
                            ),
                            before=before,
                            after=fmt_matrix(aug),
                            data={"row": r, "col": col, "factor": to_sympy(factor)},
                        )
                    )
            if mode == "summary":
                steps.append(
                    Step(
                        description=(
                            f"Column {col}: pivot {pivot}"
                            + (f" (swapped with row {pivot_row})" if swapped else "")
                            + f"; cleared {eliminated} entr{'y' if eliminated == 1 else 'ies'} below it."
                        ),
                        before="",
                        after="",
                        data={"col": col, "pivot": to_sympy(pivot), "swap": (col, pivot_row) if swapped else None},
                    )
                )

        if full:
            steps.append(
                Step(
                    description="Augmented matrix is now in (upper) row echelon form.",
                    before=fmt_matrix(aug),
                    after=fmt_matrix(aug),
                )
            )
        elif mode == "none":
            steps.append(
                Step(
                    description=f"Eliminated below all {n} pivots (per-step records omitted, detail='none').",
                    before="",
                    after="",
                )
            )

        x_exact: list = [None] * n
        x: list = [None] * n
        for i in reversed(range(n)):
            rhs = aug[i][-1] - sum(aug[i][j] * x_exact[j] for j in range(i + 1, n))
            x_exact[i] = rhs / aug[i][i]
            x[i] = to_sympy(x_exact[i])
            if full:
                steps.append(
                    Step(
                        description=(
                            f"Back-substitution: solve row {i} for x{i} "
                            f"({aug[i][-1]} - sum_of_above) / {aug[i][i]}."
                        ),
                        before="",
                        after=f"x{i} = {x[i]}",
                        data={"variable": i, "value": x[i]},
                    )
                )
        if not full:
            steps.append(
                Step(
                    description=f"Back-substitution from row {n - 1} up to row 0.",
                    before="",
                    after="",
                )
            )

        answer = list(x)
        steps.append(
            Step(
                description="Solution vector x.",
                before="",
                after="x = [" + ", ".join(str(v) for v in answer) + "]",
                data={"answer": answer},
            )
        )
        return steps, answer

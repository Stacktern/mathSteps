"""LU decomposition with partial pivoting (Doolittle form).

Problem schema::

    {
        "type": "lu_decomposition",
        "A": [[...], ...],
        "detail": "full" | "summary" | "none"      # optional
    }

Returns ``(P, L, U)`` such that ``P @ A = L @ U``. Matrices larger than
``AUTO_FULL_LIMIT`` are summarised by default (one step per pivot column).
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
class LUSolver(Solver):
    name = "lu_decomposition"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "lu_decomposition"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = to_fraction_matrix(problem["A"])
        n = len(A)
        if any(len(row) != n for row in A):
            raise ValueError("LU requires a square matrix.")
        mode = resolve_detail(problem, n, AUTO_FULL_LIMIT)
        full = mode == "full"

        U = [row[:] for row in A]
        L = [[Fraction(0)] * n for _ in range(n)]
        P = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        steps: list[Step] = [
            Step(
                description=f"LU decomposition (Doolittle with partial pivoting) of {n}x{n} A."
                + ("" if full else f" (detail='{mode}')"),
                before="",
                after=f"A = {to_sympy_matrix(A).tolist()}" if full else "",
            )
        ]

        for k in range(n):
            pivot_row = k
            best = abs(U[k][k])
            for r in range(k + 1, n):
                if abs(U[r][k]) > best:
                    best = abs(U[r][k])
                    pivot_row = r
            if best == 0:
                raise ValueError("Singular matrix; LU decomposition impossible.")
            swapped = pivot_row != k
            if swapped:
                U[k], U[pivot_row] = U[pivot_row], U[k]
                P[k], P[pivot_row] = P[pivot_row], P[k]
                L[k], L[pivot_row] = L[pivot_row], L[k]
                if full:
                    steps.append(
                        Step(
                            description=f"Swap row {k} with row {pivot_row} (pivoting).",
                            before="",
                            after="",
                            data={"swap": (k, pivot_row)},
                        )
                    )
            for r in range(k + 1, n):
                factor = U[r][k] / U[k][k]
                L[r][k] = factor
                U[r] = [a - factor * c for a, c in zip(U[r], U[k])]
                if full:
                    steps.append(
                        Step(
                            description=(
                                f"Eliminate row {r}: multiplier = {factor}; "
                                f"U[{r}, :] -= {factor} * U[{k}, :]."
                            ),
                            before="",
                            after=f"L[{r}, {k}] = {factor}",
                            data={"row": r, "k": k, "factor": to_sympy(factor)},
                        )
                    )
            if mode == "summary":
                steps.append(
                    Step(
                        description=(
                            f"Column {k}: pivot {U[k][k]}"
                            + (f" (swapped with row {pivot_row})" if swapped else "")
                            + f"; computed {n - k - 1} multiplier(s) of L."
                        ),
                        before="",
                        after="",
                        data={"col": k, "pivot": to_sympy(U[k][k])},
                    )
                )

        for i in range(n):
            L[i][i] = Fraction(1)

        P_sym, L_sym, U_sym = to_sympy_matrix(P), to_sympy_matrix(L), to_sympy_matrix(U)
        final = (
            f"Done. P*A = L*U; P = {P_sym.tolist()}; L = {L_sym.tolist()}; U = {U_sym.tolist()}."
            if full
            else f"Done. P*A = L*U ({n}x{n}; matrices omitted from the text, see the answer)."
        )
        steps.append(
            Step(
                description=final,
                before="",
                after=f"P = {P_sym.tolist()}; L = {L_sym.tolist()}; U = {U_sym.tolist()}" if full else "",
                data={"answer": (P_sym, L_sym, U_sym)},
            )
        )
        return steps, (P_sym, L_sym, U_sym)

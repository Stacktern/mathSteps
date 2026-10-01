"""Eigenvalues / eigenvectors via the characteristic polynomial.

Problem schema::

    {
        "type": "eigenvalues",
        "A": [[...], ...]
    }

The matrix is converted to exact rationals and the characteristic polynomial
``p(λ) = det(A - λI)`` is factored over the rationals.

* Factors of degree <= 2 give exact closed-form eigenvalues (integers,
  fractions, ``a ± sqrt(b)``), and exact eigenvectors from ``(A - λI)v = 0``.
* An irreducible factor of degree >= 3 has no practical closed form (Cardano's
  formula yields nested complex radicals that are unusable, which is what real
  data such as covariance matrices produce), so its roots are computed
  numerically to 15 significant digits from the exact polynomial, and the
  eigenvectors from a numerical null space. Those eigenvalues/eigenvectors are
  returned as SymPy floats.

Each distinct eigenvalue is reported once. The native answer is
``(eigenvalues, {eigenvalue: [basis vectors of its eigenspace]})``: a repeated
eigenvalue with a multi-dimensional eigenspace (e.g. the identity) keeps every
independent eigenvector, and a defective one reports fewer vectors than its
multiplicity. The public API turns this into NumPy arrays (one column per
independent eigenvector, like ``numpy.linalg.eig``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from mathsteps.core.exact import to_fraction_matrix, to_sympy_matrix
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

_NUMERIC_DIGITS = 15


def _exact_eigenvectors(A: sp.Matrix, ev: sp.Expr, I: sp.Matrix) -> list[sp.Matrix] | None:
    """Exact basis of the eigenspace, or ``None`` if SymPy cannot establish one."""
    try:
        basis = (A - ev * I).nullspace(simplify=True)
    except Exception:
        return None
    if not basis:
        return None
    vectors = [b.applyfunc(sp.simplify) for b in basis]
    # Guard against a radical SymPy failed to simplify to zero: A v must equal ev v.
    A_np = np.asarray(A.tolist(), dtype=complex)
    for v in vectors:
        vn = np.array([complex(x) for x in v])
        if np.linalg.norm(A_np @ vn - complex(ev) * vn) > 1e-8 * max(1.0, abs(complex(ev))) * max(
            np.linalg.norm(vn), 1e-300
        ):
            return None
    return vectors


def _numeric_eigenvectors(A: sp.Matrix, ev: sp.Expr) -> list[sp.Matrix]:
    """Basis of the null space of ``A - ev*I`` via SVD, each vector scaled so its largest entry is 1."""
    A_np = np.asarray(A.tolist(), dtype=complex)
    n = A_np.shape[0]
    _, s, vh = np.linalg.svd(A_np - complex(ev) * np.eye(n))
    k = max(1, int(np.sum(s <= 1e-8 * max(1.0, s[0]))))
    out: list[sp.Matrix] = []
    for row in vh[-k:]:
        v = row.conj()
        v = v / v[np.argmax(np.abs(v))]
        v[np.abs(v) < 1e-14] = 0
        if np.all(np.abs(v.imag) < 1e-12):
            entries = [sp.Float(float(x.real), _NUMERIC_DIGITS) for x in v]
        else:
            entries = [
                sp.Float(x.real, _NUMERIC_DIGITS) + sp.I * sp.Float(x.imag, _NUMERIC_DIGITS)
                for x in v
            ]
        out.append(sp.Matrix(entries))
    return out


@register
class EigenvalueSolver(Solver):
    name = "eigenvalues"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "eigenvalues"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        A = to_sympy_matrix(to_fraction_matrix(problem["A"]))
        if A.rows != A.cols:
            raise ValueError("eigenvalues requires a square matrix.")

        lam = sp.symbols("λ")
        I = sp.eye(A.rows)
        poly = A.charpoly(lam)
        char = sp.expand(poly.as_expr())
        steps: list[Step] = [
            Step(
                description=f"Characteristic polynomial: det(A - λI) = {char}.",
                before=f"A = {A.tolist()}",
                after=f"p(λ) = {char}",
            )
        ]

        _, factors = sp.factor_list(char, lam)
        eigs: list[sp.Expr] = []
        numeric: set[sp.Expr] = set()
        for factor, _mult in factors:
            if not factor.has(lam):
                continue
            if sp.degree(factor, lam) <= 2:
                eigs.extend(sp.solve(factor, lam))
            else:
                roots = sp.Poly(factor, lam).nroots(n=_NUMERIC_DIGITS, maxsteps=500)
                for root in roots:
                    re_part, im_part = sp.re(root), sp.im(root)
                    value = sp.Float(re_part, _NUMERIC_DIGITS)
                    if abs(float(im_part)) > 1e-12:
                        value += sp.I * sp.Float(im_part, _NUMERIC_DIGITS)
                    eigs.append(value)
                    numeric.add(value)
        eigs = sorted(eigs, key=lambda e: (complex(e).real, complex(e).imag))

        note = (
            " (irreducible factor of degree >= 3: no practical closed form, "
            "roots computed numerically to 15 digits)"
            if numeric else ""
        )
        steps.append(
            Step(
                description=f"Eigenvalues (roots of p): {eigs}{note}.",
                before="",
                after=f"eigenvalues = {eigs}",
            )
        )

        eigvecs: dict[sp.Expr, list[sp.Matrix]] = {}
        for ev in eigs:
            basis = None if ev in numeric else _exact_eigenvectors(A, ev, I)
            if basis is None:
                basis = _numeric_eigenvectors(A, ev)
            eigvecs[ev] = basis
            steps.append(
                Step(
                    description=(
                        f"For λ = {ev}: solve (A - {ev}I) v = 0; eigenspace dimension {len(basis)}, "
                        f"eigenvector(s) {[v.T.tolist()[0] for v in basis]}."
                    ),
                    before="",
                    after=f"v_{ev} = {[v.T.tolist()[0] for v in basis]}",
                )
            )

        steps.append(
            Step(
                description=f"Summary: eigenvalues = {eigs}; eigenvectors = {eigvecs}.",
                before="",
                after=f"eigenvalues = {eigs}",
                data={"answer": (eigs, eigvecs)},
            )
        )
        return steps, (eigs, eigvecs)

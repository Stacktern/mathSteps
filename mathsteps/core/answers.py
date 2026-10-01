"""Turn solver-native (exact / SymPy) answers into plain Python / NumPy results.

Solvers work in exact arithmetic where they can and return SymPy objects.
The public API hands users ``numpy`` arrays and Python floats instead, and
keeps the exact form in ``Result.exact``.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

import numpy as np
import sympy as sp

from .exact import to_fraction
from .step import Step


class InterpolatingPolynomial:
    """A polynomial returned by the interpolation solvers.

    Callable on scalars or arrays, evaluated in exact rational arithmetic so
    realistic abscissae (years like 1950..2000) do not lose digits to
    cancellation::

        p = mathsteps.lagrange([(1950, 151.3), (1960, 179.3), (1970, 203.3)]).answer
        p(1965)          # float
        p([1955, 1965])  # ndarray
        p.expr           # SymPy expression
        p.coefficients   # ascending powers, as floats
    """

    def __init__(self, expr: sp.Expr, symbol: sp.Symbol | None = None):
        self.symbol = symbol if symbol is not None else sp.Symbol("x")
        self.expr = sp.expand(sp.sympify(expr))
        poly = sp.Poly(self.expr, self.symbol)
        self._coeffs_desc: list[Fraction] = [to_fraction(c) for c in poly.all_coeffs()]

    @property
    def degree(self) -> int:
        return len(self._coeffs_desc) - 1

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients in ascending powers ``[c0, c1, ...]`` as floats."""
        return np.array([float(c) for c in reversed(self._coeffs_desc)])

    def _eval(self, value: float) -> float:
        x = to_fraction(value)
        acc = Fraction(0)
        for c in self._coeffs_desc:
            acc = acc * x + c
        return float(acc)

    def __call__(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return self._eval(float(arr))
        return np.array([self._eval(float(v)) for v in arr.ravel()]).reshape(arr.shape)

    def _sympy_(self) -> sp.Expr:  # lets ``sympy.sympify(poly)`` work
        return self.expr

    def __str__(self) -> str:
        return str(self.expr)

    def __repr__(self) -> str:
        return f"InterpolatingPolynomial({self.expr})"


def _scalar(value: Any) -> float | complex:
    """A real ``float`` when the value is real, otherwise a ``complex``."""
    if isinstance(value, (int, float)):
        return float(value)
    real = getattr(value, "is_real", None)
    if real:
        return float(value)
    c = complex(value)
    return c.real if c.imag == 0 else c


def to_plain(value: Any) -> Any:
    """Convert SymPy numbers / matrices / lists to Python scalars and ``numpy`` arrays."""
    if isinstance(value, sp.MatrixBase):
        return _array([[_scalar(v) for v in row] for row in value.tolist()])
    if isinstance(value, (list, tuple)):
        return _array([_scalar(v) for v in value])
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, str)) or value is None:
        return value
    return _scalar(value)


def _array(data: Sequence) -> np.ndarray:
    flat = np.array(data, dtype=object).ravel()
    is_complex = any(isinstance(v, complex) for v in flat)
    return np.array(data, dtype=complex if is_complex else float)


def _unit_vector(vec: sp.Matrix) -> np.ndarray:
    """Unit-norm numpy vector with a canonical phase (largest entry real, positive)."""
    v = np.array([complex(x) for x in vec], dtype=complex)
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) > 0:
        v = v * (abs(v[k]) / v[k])
    v = v / np.linalg.norm(v)
    return v.real if np.all(np.abs(v.imag) < 1e-12) else v


def details_from_steps(steps: Sequence[Step]) -> dict[str, Any]:
    """Extra outputs (grids, trajectories, ...) a solver attached to its final step."""
    for step in reversed(steps):
        if "details" in step.data:
            return dict(step.data["details"])
    return {}


def finalize(problem: dict, raw: Any, steps: Sequence[Step]) -> tuple[Any, Any, dict[str, Any]]:
    """Return ``(answer, exact, details)`` for a solver's native answer ``raw``."""
    ptype = problem.get("type")
    details = details_from_steps(steps)
    if ptype in ("linear_system", "cramers_rule"):
        return to_plain(list(raw)), list(raw), details
    if ptype == "determinant":
        return float(raw), raw, details
    if ptype == "matrix_inverse":
        return to_plain(raw), raw, details
    if ptype == "lu_decomposition":
        return tuple(to_plain(m) for m in raw), tuple(raw), details
    if ptype == "eigenvalues":
        eigs, vecs = raw
        values: list = []
        columns: list[np.ndarray] = []
        for ev in eigs:
            for vec in vecs[ev]:
                values.append(_scalar(ev))
                columns.append(_unit_vector(vec))
        answer = (_array(values), np.column_stack(columns) if columns else np.zeros((0, 0)))
        return answer, raw, details
    if ptype == "interpolation":
        return InterpolatingPolynomial(raw), raw, details
    return raw, None, details

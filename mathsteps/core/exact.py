"""Exact rational arithmetic helpers shared by the linear-algebra and interpolation solvers.

The solvers do their elimination on plain ``fractions.Fraction`` values
(fast, exact, no expression-tree growth) and convert to SymPy rationals
only for the answers and ``Step.data`` payloads exposed to callers.
"""
from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Any, Sequence

import sympy as sp


def to_fraction(x: Any) -> Fraction:
    """Convert an int / float / SymPy number to an exact ``Fraction``.

    Floats are converted through their shortest decimal representation,
    so ``0.1`` becomes ``1/10`` (not the binary expansion of the double).
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, numbers.Integral):
        return Fraction(int(x))
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot use {x!r} as a matrix entry; expected a real number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Matrix entries must be finite; got {x!r}.")
    return Fraction(str(value))


def to_fraction_matrix(rows: Any) -> list[list[Fraction]]:
    """Convert a nested sequence (or ``sympy.Matrix``) to a list of Fraction rows."""
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    return [[to_fraction(v) for v in row] for row in rows]


def to_fraction_vector(values: Any) -> list[Fraction]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    flat: list[Fraction] = []
    for v in values:
        if isinstance(v, (list, tuple)):  # column vector given as [[b0], [b1], ...]
            (v,) = v
        flat.append(to_fraction(v))
    return flat


def to_sympy(x: Fraction) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator)


def to_sympy_matrix(rows: Sequence[Sequence[Fraction]]) -> sp.Matrix:
    return sp.Matrix([[to_sympy(v) for v in row] for row in rows])


def fmt_matrix(rows: Sequence[Sequence[Fraction]]) -> str:
    """Pretty-print a Fraction matrix, one bracketed row per line."""
    return "\n".join("[" + "  ".join(str(v) for v in row) + "]" for row in rows)

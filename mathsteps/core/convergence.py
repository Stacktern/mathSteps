"""Convergence reporting for iterative solvers.

Iterative solvers (Newton, secant, fixed-point, bisection, shooting) stop
either because they met their tolerance or because they ran out of
iterations. They record which on their *final* step as
``Step.data["converged"]`` (``True`` / ``False``); non-iterative solvers
leave it out. The public API surfaces it as ``Result.converged`` and
emits a :class:`ConvergenceWarning` when it is ``False``.
"""
from __future__ import annotations

from typing import Sequence

from .step import Step


class ConvergenceWarning(UserWarning):
    """An iterative solver hit ``max_iter`` before meeting its tolerance."""


def converged_from_steps(steps: Sequence[Step]) -> bool | None:
    """Return the solver's convergence flag, or ``None`` if it is not iterative."""
    for step in reversed(steps):
        if "converged" in step.data:
            return bool(step.data["converged"])
    return None

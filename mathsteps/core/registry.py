"""Solver registry: maps problem type -> solver instance.

Solvers register themselves on import, so ``cli.py`` simply imports the
domain modules to trigger registration.
"""
from __future__ import annotations

from .solver_base import Solver

_SOLVERS: list[Solver] = []


def register(solver_cls_or_instance) -> Solver:
    """Register a solver.

    Accepts either a ``Solver`` instance or a ``Solver`` *subclass* (in
    which case one instance is created and stored).
    """
    if isinstance(solver_cls_or_instance, type) and issubclass(
        solver_cls_or_instance, Solver
    ):
        instance = solver_cls_or_instance()
    elif isinstance(solver_cls_or_instance, Solver):
        instance = solver_cls_or_instance
    else:
        raise TypeError(
            f"register() expects a Solver subclass or instance, "
            f"got {type(solver_cls_or_instance).__name__}"
        )
    _SOLVERS.append(instance)
    return instance


def all_solvers() -> list[Solver]:
    return list(_SOLVERS)


def find_solver(problem: dict) -> Solver:
    """Return the first registered solver that claims this problem."""
    for s in _SOLVERS:
        if s.can_solve(problem):
            return s
    raise ValueError(
        f"No solver registered for problem type "
        f"{problem.get('type')!r}. "
        f"Registered: {[s.name for s in _SOLVERS]}"
    )

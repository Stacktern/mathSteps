"""Abstract Solver interface implemented by every domain."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .step import Step


class Solver(ABC):
    """Base class for every step-by-step solver.

    A Solver consumes a *problem* (a plain dict in v1, structured per
    domain) and produces ``(steps, final_answer)``.
    """

    name: str = ""

    @abstractmethod
    def can_solve(self, problem: dict) -> bool:
        """Return True if this solver handles the given problem."""

    @abstractmethod
    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        """Produce ``(steps, final_answer)`` for the problem."""

"""Step dataclass shared by every solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """A single transformation in a step-by-step solution.

    Attributes
    ----------
    description : str
        Human-readable explanation of what was done.
    before : str
        State before the step (matrix snapshot, equation, etc.).
    after : str
        State after the step.
    data : dict
        Optional structured payload (e.g. row indices swapped, factors used).
    """

    description: str
    before: str
    after: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.description}\n  {self.before} -> {self.after}"

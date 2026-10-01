"""How much per-step output a solver should record.

Large problems (a 40x40 elimination, an ODE with 10 000 steps) produce a
flood of steps that costs time and memory and that nobody reads. Solvers that
can flood honour a ``detail`` setting in the problem dict:

``"full"``     every step (the default for small problems)
``"summary"``  the first few and last few steps plus one note saying how many
               were omitted (the default for large problems)
``"none"``     only the introduction and the final step

The answer is identical in every mode.
"""
from __future__ import annotations

from .step import Step

HEAD = 3  # steps kept at the start in "summary" mode
TAIL = 2  # steps kept at the end in "summary" mode

_MODES = ("full", "summary", "none")


def resolve_detail(problem: dict, size: int, limit: int) -> str:
    """Return ``"full"``, ``"summary"`` or ``"none"``.

    An explicit ``problem["detail"]`` wins; otherwise a problem with more than
    ``limit`` steps (``size``) is summarised.
    """
    detail = problem.get("detail")
    if detail is None:
        return "full" if size <= limit else "summary"
    if detail not in _MODES:
        raise ValueError(f"detail must be one of {_MODES}; got {detail!r}.")
    return detail


def keep_step(k: int, n: int, mode: str) -> bool:
    """Should step ``k`` (1-based) of ``n`` be recorded in ``mode``?"""
    if mode == "full":
        return True
    if mode == "none":
        return False
    return k <= HEAD or k > n - TAIL


def omission_note(k: int, n: int, mode: str) -> Step | None:
    """The one-off step announcing omitted steps; emit it when ``k`` first becomes omitted."""
    if mode == "summary" and n > HEAD + TAIL and k == HEAD + 1:
        omitted = n - HEAD - TAIL
        return Step(
            description=(
                f"... {omitted} intermediate step{'s' if omitted != 1 else ''} omitted "
                f"(pass detail='full' to record every step) ..."
            ),
            before="",
            after="",
            data={"omitted": omitted},
        )
    if mode == "none" and k == 1:
        return Step(
            description=f"Per-step records omitted for all {n} steps (detail='none').",
            before="",
            after="",
            data={"omitted": n},
        )
    return None

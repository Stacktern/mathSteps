"""Step planning shared by the fixed-step ODE solvers and their verifier."""
from __future__ import annotations

import math


def fixed_step_plan(x0: float, x_end: float, h: float) -> tuple[int, float]:
    """Return ``(n, h_eff)``: ``n`` equal steps from ``x0`` to ``x_end``.

    ``h`` is the requested step *size* (always positive). The returned
    ``h_eff`` is signed (negative when ``x_end < x0``, i.e. integrating
    backwards) and satisfies ``|h_eff| <= h``: the requested ``h`` rarely
    divides the interval exactly (``h = 0.01`` on ``[0, pi]``), so the step is
    shrunk slightly to land exactly on ``x_end`` instead of stopping short.

    Raises ``ValueError`` for ``h <= 0`` or ``x_end == x0``.
    """
    if not h > 0:
        raise ValueError(f"Step size h must be positive; got h = {h}.")
    span = x_end - x0
    if not math.isfinite(span) or span == 0:
        raise ValueError(f"x_end ({x_end}) must differ from x0 ({x0}).")
    n = max(1, math.ceil(abs(span) / h - 1e-9))
    return n, span / n

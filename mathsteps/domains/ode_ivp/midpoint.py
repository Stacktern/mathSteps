"""Midpoint method (a 2nd-order Runge-Kutta variant) for IVPs."""
from __future__ import annotations

from typing import Any

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import solve_fixed_step


def _step(f, x, y, h):
    k1 = f(x, y)
    y_mid = y + 0.5 * h * k1
    k2 = f(x + h / 2, y_mid)
    return y + h * k2, {"k1": k1, "y_mid": y_mid, "k2": k2}


def _describe(k, x, y, x_new, y_new, h, info):
    return (
        f"Step {k}: x = {x}, y = {y}; k1 = {info['k1']}, y_mid = {info['y_mid']}, "
        f"k2 = {info['k2']}; y_new = y + h*k2 = {y_new}; x_new = {x_new}."
    )


@register
class MidpointSolver(Solver):
    name = "midpoint"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "ivp" and problem.get("method") == "midpoint"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        return solve_fixed_step(
            problem, title="Midpoint method", step_fn=_step, describe=_describe
        )

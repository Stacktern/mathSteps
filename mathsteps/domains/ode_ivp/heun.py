"""Improved Euler / Heun's method for IVPs.

Predictor-corrector: predict with one Euler step, then correct using
the average of the slopes at the start and predicted end-points.
"""
from __future__ import annotations

from typing import Any

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import solve_fixed_step


def _step(f, x, y, h):
    k1 = f(x, y)
    y_pred = y + h * k1
    k2 = f(x + h, y_pred)
    return y + 0.5 * h * (k1 + k2), {"k1": k1, "y_pred": y_pred, "k2": k2}


def _describe(k, x, y, x_new, y_new, h, info):
    return (
        f"Step {k}: at x = {x}, y = {y}; k1 = {info['k1']}, y_pred = {info['y_pred']}, "
        f"k2 = {info['k2']}; y_new = y + h/2*(k1+k2) = {y_new}; x_new = {x_new}."
    )


@register
class HeunSolver(Solver):
    name = "heun"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "ivp" and problem.get("method") == "heun"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        return solve_fixed_step(
            problem, title="Improved Euler (Heun)", step_fn=_step, describe=_describe
        )

"""Classical 4th-order Runge-Kutta (RK4) for IVPs.

The workhorse ODE solver: each step records k1, k2, k3, k4.
"""
from __future__ import annotations

from typing import Any

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import solve_fixed_step


def _step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), {"k1": k1, "k2": k2, "k3": k3, "k4": k4}


def _describe(k, x, y, x_new, y_new, h, info):
    return (
        f"Step {k}: x = {x}, y = {y}; "
        f"k1 = {info['k1']}, k2 = {info['k2']}, k3 = {info['k3']}, k4 = {info['k4']}; "
        f"y_new = y + h/6*(k1+2k2+2k3+k4) = {y_new}; x_new = {x_new}."
    )


@register
class RK4Solver(Solver):
    name = "rk4"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "ivp" and problem.get("method") == "rk4"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        return solve_fixed_step(problem, title="RK4", step_fn=_step, describe=_describe)

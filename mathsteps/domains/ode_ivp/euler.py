"""Euler's method for IVPs.

Solves ``dy/dx = f(x, y)`` (a scalar equation or a system) with explicit
forward Euler. Emits one ``Step`` per step (see ``mathsteps.core.detail``).
"""
from __future__ import annotations

from typing import Any

from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import solve_fixed_step


def _step(f, x, y, h):
    slope = f(x, y)
    return y + h * slope, {"slope": slope}


def _describe(k, x, y, x_new, y_new, h, info):
    return (
        f"Step {k}: x = {x}, y = {y}; "
        f"f(x, y) = {info['slope']}; y_new = y + h*f = {y_new}; x_new = {x_new}."
    )


@register
class EulerSolver(Solver):
    name = "euler"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "ivp" and problem.get("method") == "euler"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        return solve_fixed_step(problem, title="Euler's method", step_fn=_step, describe=_describe)

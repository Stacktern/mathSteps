"""Shooting method for second-order two-point BVPs.

Solves ``y'' = f(x, y, y')`` on ``[x0, x_end]`` with
``y(x0) = y_left`` and ``y(x_end) = y_right``. Reduces to the
first-order system ``y1' = y2, y2' = f(x, y1, y2)``, then guesses the
missing initial slope ``s = y'(x0)`` and adjusts via secant until the
right boundary is satisfied.

This is the project's showcase for *composability*: it reuses both
the IVP machinery (RK4 integrator) and the root-finding machinery
(secant on the boundary residual).

The answer is the solution ``y`` on the RK4 grid (``n + 1`` points); the grid,
``y'`` and the converged slope ``y'(x0)`` are attached as ``details``.

Problem schema::

    {
        "type": "bvp",
        "method": "shooting",
        "f_expr": "-y",                # second derivative f(x, y, y')
        "variable": "x",
        "y_left": 0.0,
        "y_right": 0.0,
        "x0": 0.0,
        "x_end": 3.141592653589793,
        "h": 0.01,
        "s0": 0.5,                     # initial slope guesses
        "s1": 1.5,
        "tol": 1e-6,
        "max_iter": 30
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from mathsteps.core.expr import parse_expr
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step
from mathsteps.core.stepping import fixed_step_plan


@register
class ShootingSolver(Solver):
    name = "shooting"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "bvp" and problem.get("method", "shooting") == "shooting"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        x = sp.symbols(problem["variable"])
        y = sp.symbols("y")
        yp = sp.symbols("yp")
        f_expr = parse_expr(
            problem["f_expr"],
            {problem["variable"]: x, "y": y, "yp": yp},
        )
        f = sp.lambdify((x, y, yp), f_expr, modules=["numpy"])

        x0 = float(problem["x0"])
        x_end = float(problem["x_end"])
        h = float(problem["h"])
        y_left = float(problem["y_left"])
        y_right_target = float(problem["y_right"])
        s0 = float(problem.get("s0", 0.0))
        s1 = float(problem.get("s1", 1.0))
        tol = float(problem.get("tol", 1e-6))
        max_iter = int(problem.get("max_iter", 30))

        n, h = fixed_step_plan(x0, x_end, h)

        def rk4_path(s: float):
            """Integrate from x0 with slope ``s``; return the grid, y and y'."""
            y1, y2 = y_left, s
            xs, y1s, y2s = [x0], [y1], [y2]
            for i in range(n):
                xv = xs[-1]
                k1a, k1b = y2, float(f(xv, y1, y2))
                k2a, k2b = y2 + h * k1b / 2, float(f(xv + h / 2, y1 + h * k1a / 2, y2 + h * k1b / 2))
                k3a, k3b = y2 + h * k2b / 2, float(f(xv + h / 2, y1 + h * k2a / 2, y2 + h * k2b / 2))
                k4a, k4b = y2 + h * k3b, float(f(xv + h, y1 + h * k3a, y2 + h * k3b))
                y1 = y1 + (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
                y2 = y2 + (h / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
                xs.append(x0 + (i + 1) * h if i + 1 < n else x_end)
                y1s.append(y1)
                y2s.append(y2)
            return np.array(xs), np.array(y1s), np.array(y2s)

        def rk4_endpoint(s: float) -> float:
            y1, y2 = y_left, s
            for i in range(n):
                xv = x0 + i * h
                k1a, k1b = y2, float(f(xv, y1, y2))
                k2a, k2b = y2 + h * k1b / 2, float(f(xv + h / 2, y1 + h * k1a / 2, y2 + h * k1b / 2))
                k3a, k3b = y2 + h * k2b / 2, float(f(xv + h / 2, y1 + h * k2a / 2, y2 + h * k2b / 2))
                k4a, k4b = y2 + h * k3b, float(f(xv + h, y1 + h * k3a, y2 + h * k3b))
                y1 = y1 + (h / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
                y2 = y2 + (h / 6.0) * (k1b + 2 * k2b + 2 * k3b + k4b)
            return y1

        steps: list[Step] = [
            Step(
                description=(
                    f"Shooting method for y'' = {f_expr} (system y1' = y2, y2' = f); "
                    f"y({x0}) = {y_left}, target y({x_end}) = {y_right_target}, h = {h}."
                ),
                before="",
                after=f"guess s ∈ [{s0}, {s1}]",
            )
        ]

        s_prev, s_curr = s0, s1
        r_prev = rk4_endpoint(s_prev) - y_right_target
        r_curr = rk4_endpoint(s_curr) - y_right_target
        steps.append(
            Step(
                description=(
                    f"Initial residuals at s0={s0}, s1={s1}: "
                    f"y({x_end}) - target = [{r_prev + y_right_target}, {r_curr + y_right_target}]; "
                    f"residuals = [{r_prev}, {r_curr}]."
                ),
                before="",
                after=f"r = [{r_prev}, {r_curr}]",
            )
        )

        s_star = s_curr
        converged = False
        for k in range(1, max_iter + 1):
            denom = r_curr - r_prev
            if denom == 0:
                raise ValueError("Shooting secant: residual difference is zero.")
            s_new = s_curr - r_curr * (s_curr - s_prev) / denom
            r_new = rk4_endpoint(s_new) - y_right_target
            steps.append(
                Step(
                    description=(
                        f"Iteration {k}: secant update s = {s_new}; "
                        f"integrated endpoint gives residual = {r_new}."
                    ),
                    before=f"s = {s_curr}, r = {r_curr}",
                    after=f"s = {s_new}, r = {r_new}",
                    data={"iter": k, "s": s_new, "r": r_new},
                )
            )
            if abs(r_new) <= tol:
                converged = True
                s_star = s_new
                steps.append(
                    Step(
                        description=f"Converged: residual |{r_new}| <= tol {tol}.",
                        before="",
                        after=f"s* = {s_star}",
                    )
                )
                break
            s_prev, r_prev = s_curr, r_curr
            s_curr, r_curr = s_new, r_new
            s_star = s_new
        else:
            steps.append(
                Step(
                    description=(
                        f"Did NOT converge: reached max_iter={max_iter} with "
                        f"|residual| > tol = {tol}; best s* = {s_star}."
                    ),
                    before="",
                    after=f"s* ≈ {s_star}",
                )
            )

        xs, y_path, dy_path = rk4_path(s_star)
        y_final = float(y_path[-1])
        steps.append(
            Step(
                description=(
                    f"Shooting {'converged' if converged else 'stopped without converging'}: "
                    f"shooting parameter s* = y'({x0}) = {s_star}; "
                    f"integrated endpoint y({x_end}) ≈ {y_final}. "
                    f"The solution is returned on the {n + 1}-point grid (step h = {h})."
                ),
                before="",
                after=f"y({x_end}) ≈ {y_final}",
                data={
                    "answer": y_path,
                    "s_star": s_star,
                    "converged": converged,
                    "details": {"x": xs, "y": y_path, "dy": dy_path, "slope": s_star},
                },
            )
        )
        return steps, y_path

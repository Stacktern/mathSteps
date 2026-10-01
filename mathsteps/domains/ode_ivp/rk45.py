"""Adaptive step-size RK45 (Dormand-Prince) for IVPs.

Uses scipy.integrate.solve_ivp with RK45 under the hood for the actual
integration, but emits one ``Step`` per accepted step so the output looks like
every other step-by-step solver. Works for scalar equations and systems, and
forwards or backwards in ``x``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from mathsteps.core.detail import keep_step, omission_note, resolve_detail
from mathsteps.core.registry import register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step

from .common import (
    AUTO_FULL_LIMIT,
    _final_text,
    _initial_condition,
    _state_label,
    fv,
    get_float,
    initial_state,
    parse_ivp,
    solution_details,
)


@register
class RK45Solver(Solver):
    name = "rk45"

    def can_solve(self, problem: dict) -> bool:
        return problem.get("type") == "ivp" and problem.get("method") == "rk45"

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        system = parse_ivp(problem)
        x0 = get_float(problem, "x0")
        x_end = get_float(problem, "x_end")
        y0 = initial_state(problem, system)
        if x_end == x0 or not np.isfinite(x_end - x0):
            raise ValueError(f"x_end ({x_end}) must differ from x0 ({x0}).")
        atol = float(problem.get("atol", 1e-8))
        rtol = float(problem.get("rtol", 1e-8))
        max_step = float(problem.get("max_step", abs(x_end - x0) / 100.0))
        sc = system.scalar
        direction = "" if x_end > x0 else " backwards"

        sol = solve_ivp(
            lambda xv, yv: system.f(xv, yv),
            t_span=(x0, x_end),
            y0=y0,
            method="RK45",
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        count = len(sol.t) - 1
        mode = resolve_detail(problem, count, AUTO_FULL_LIMIT)
        steps: list[Step] = [
            Step(
                description=(
                    f"Adaptive RK45 (Dormand-Prince) for {system.equations()}, "
                    f"{_initial_condition(system, x0, y0)}, integrating{direction} to x = {x_end}; "
                    f"atol = {atol}, rtol = {rtol}, max_step = {max_step}."
                ),
                before="",
                after=f"x0 = {x0}, y0 = {fv(y0, sc)}",
            )
        ]

        for k in range(1, count + 1):
            xk, yk = float(sol.t[k]), sol.y[:, k]
            step_size = float(sol.t[k] - sol.t[k - 1])
            if keep_step(k, count, mode):
                steps.append(
                    Step(
                        description=(
                            f"Step {k}: accepted step of size h = {step_size}; "
                            f"x = {xk}, y = {fv(yk, sc)}."
                        ),
                        before=f"x = {float(sol.t[k - 1])}",
                        after=f"x = {xk}, y = {fv(yk, sc)}",
                        data={"step": k, "x": xk, "y": fv(yk, sc), "h": step_size},
                    )
                )
            else:
                note = omission_note(k, count, mode)
                if note is not None:
                    steps.append(note)

        y_final = sol.y[:, -1]
        answer: Any = float(y_final[0]) if sc else y_final.copy()
        steps.append(
            Step(
                description=(
                    f"Final state after {count} accepted RK45 steps: "
                    + _final_text(system, x_end, y_final).removeprefix("Approximate ")
                ),
                before="",
                after=f"{_state_label(system)}({x_end}) ≈ {fv(y_final, sc)}",
                data={
                    "answer": answer,
                    "details": solution_details(system, list(sol.t), [sol.y[:, i] for i in range(len(sol.t))]),
                },
            )
        )
        return steps, answer

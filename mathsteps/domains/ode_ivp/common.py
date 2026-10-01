"""Shared machinery for the IVP solvers (scalar equations *and* systems).

Problem schema shared by every IVP solver::

    {
        "type": "ivp",
        "method": "euler" | "heun" | "midpoint" | "rk4" | "rk45",
        "variable": "x",           # independent variable
        "function": "y",           # dependent variable name (or a list of names for a system)
        "f_expr": "-2*x*y",        # right-hand side dy/dx = f(x, y) (or a list, one per equation)
        "y0": 1.0,                 # initial value at x0 (or a list for a system)
        "x0": 0.0,
        "x_end": 2.0,              # may be smaller than x0: integrates backwards
        "h": 0.1,                  # step size (fixed-step methods); shrunk to land on x_end
        "detail": "full" | "summary" | "none"   # optional, see mathsteps.core.detail
    }

A system such as the harmonic oscillator ``x'' = -4x`` is written as the
first-order pair ``function = ["x", "v"]``, ``f_expr = ["v", "-4*x"]``,
``y0 = [1, 0]``. Adaptive RK45 takes ``atol`` / ``rtol`` instead of ``h``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import sympy as sp

from mathsteps.core.detail import keep_step, omission_note, resolve_detail
from mathsteps.core.expr import parse_expr
from mathsteps.core.step import Step
from mathsteps.core.stepping import fixed_step_plan

# More than this many steps and the per-step records are summarised by default.
AUTO_FULL_LIMIT = 200


@dataclass
class ODESystem:
    """A parsed first-order system ``y' = f(x, y)``."""

    exprs: list[sp.Expr]
    x: sp.Symbol
    ys: list[sp.Symbol]
    names: list[str]
    variable: str
    f: Callable[[float, np.ndarray], np.ndarray]

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def scalar(self) -> bool:
        return self.size == 1

    def equations(self) -> str:
        return ", ".join(
            f"d{n}/d{self.variable} = {e}" for n, e in zip(self.names, self.exprs)
        )


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]


def parse_ivp(problem: dict) -> ODESystem:
    """Parse the equation(s) of an IVP into a numeric right-hand side ``f(x, y_vector)``."""
    variable = problem["variable"]
    names = [str(n) for n in _as_list(problem["function"])]
    rhs = [str(r) for r in _as_list(problem["f_expr"])]
    if len(names) != len(rhs):
        raise ValueError(
            f"Got {len(rhs)} right-hand side(s) for {len(names)} unknown(s) {names}; "
            f"a system needs one expression per unknown."
        )
    if len(set(names + [variable])) != len(names) + 1:
        raise ValueError(f"Variable names must be distinct; got {[variable] + names}.")
    x = sp.symbols(variable)
    ys = [sp.symbols(n) for n in names]
    symbols = {variable: x, **dict(zip(names, ys))}
    exprs = [parse_expr(r, symbols) for r in rhs]
    lam = sp.lambdify((x, *ys), exprs, modules=["numpy"])
    m = len(names)

    def f(xv: float, yv: np.ndarray) -> np.ndarray:
        out = np.asarray(lam(xv, *yv), dtype=float)
        return out.reshape(m)

    return ODESystem(exprs=exprs, x=x, ys=ys, names=names, variable=variable, f=f)


def initial_state(problem: dict, system: ODESystem) -> np.ndarray:
    y0 = np.atleast_1d(np.asarray(problem["y0"], dtype=float))
    if y0.shape != (system.size,):
        raise ValueError(
            f"y0 has {y0.size} value(s) but there are {system.size} unknown(s) {system.names}."
        )
    return y0


def get_float(problem: dict, key: str, default: float | None = None) -> float:
    if key not in problem:
        if default is None:
            raise KeyError(f"Missing required key {key!r} in problem.")
        return float(default)
    return float(problem[key])


def fv(value: Any, scalar: bool) -> Any:
    """Display form of a state/slope: a float for one equation, a list of floats otherwise."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if scalar else [float(v) for v in arr]


def solution_details(system: ODESystem, xs: list[float], ys: list[np.ndarray]) -> dict[str, Any]:
    return {"x": np.array(xs), "y": np.vstack(ys), "names": list(system.names)}


def solve_fixed_step(
    problem: dict,
    *,
    title: str,
    step_fn: Callable[[Callable, float, np.ndarray, float], tuple[np.ndarray, dict]],
    describe: Callable[..., str],
) -> tuple[list[Step], Any]:
    """Run a fixed-step one-step method and record it.

    ``step_fn(f, x, y, h)`` advances one step and returns ``(y_new, info)``,
    where ``info`` holds the intermediate slopes (``k1``...). ``describe`` turns
    one step into text: ``describe(k, x, y, x_new, y_new, h, info)`` with
    every value already formatted for display.
    """
    system = parse_ivp(problem)
    x0 = get_float(problem, "x0")
    x_end = get_float(problem, "x_end")
    y = initial_state(problem, system)
    n, h = fixed_step_plan(x0, x_end, get_float(problem, "h"))
    mode = resolve_detail(problem, n, AUTO_FULL_LIMIT)
    sc = system.scalar
    direction = "" if x_end > x0 else " backwards"

    steps: list[Step] = [
        Step(
            description=(
                f"{title} for {system.equations()}, {_initial_condition(system, x0, y)}, "
                f"integrating{direction} to x = {x_end} with h = {h} ({n} steps)."
            ),
            before="",
            after=f"x0 = {x0}, y0 = {fv(y, sc)}",
        )
    ]

    xs = [x0]
    states = [y.copy()]
    trajectory: list[tuple[float, Any]] = [(x0, fv(y, sc))]
    x = x0
    for k in range(1, n + 1):
        x_new = x0 + k * h if k < n else x_end
        y_new, info = step_fn(system.f, x, y, h)
        if not np.all(np.isfinite(y_new)):
            raise ValueError(
                f"The solution became non-finite at step {k} (x = {x_new}); the step size "
                f"h = {h} is probably too large for this equation (stiffness / instability)."
            )
        xs.append(x_new)
        states.append(y_new.copy())
        trajectory.append((x_new, fv(y_new, sc)))
        if keep_step(k, n, mode):
            shown = {name: fv(val, sc) for name, val in info.items()}
            steps.append(
                Step(
                    description=describe(k, x, fv(y, sc), x_new, fv(y_new, sc), h, shown),
                    before=f"x = {x}, y = {fv(y, sc)}",
                    after=f"x = {x_new}, y = {fv(y_new, sc)}",
                    data={"step": k, "x": x_new, "y": fv(y_new, sc), **shown},
                )
            )
        else:
            note = omission_note(k, n, mode)
            if note is not None:
                steps.append(note)
        x, y = x_new, y_new

    answer: Any = float(y[0]) if sc else y.copy()
    steps.append(
        Step(
            description=_final_text(system, x_end, y),
            before="",
            after=f"{_state_label(system)}({x_end}) ≈ {fv(y, sc)}",
            data={
                "answer": answer,
                "trajectory": trajectory,
                "details": solution_details(system, xs, states),
            },
        )
    )
    return steps, answer


def _initial_condition(system: ODESystem, x0: float, y: np.ndarray) -> str:
    if system.scalar:
        return f"{system.names[0]}({x0}) = {float(y[0])}"
    return ", ".join(f"{n}({x0}) = {float(v)}" for n, v in zip(system.names, y))


def _state_label(system: ODESystem) -> str:
    return system.names[0] if system.scalar else "(" + ", ".join(system.names) + ")"


def _final_text(system: ODESystem, x_end: float, y: np.ndarray) -> str:
    return f"Approximate {_state_label(system)}({x_end}) ≈ {fv(y, system.scalar)}."

"""Finite-difference method for two-point linear BVPs.

Solves ``-y'' + p(x) y' + q(x) y = r(x)`` on ``[a, b]`` with
``y(a) = alpha``, ``y(b) = beta`` by discretising into an ``(n-1) x
(n-1)`` linear system (n interior points), then delegating to the
Gaussian elimination solver.

The answer is the array of ``y`` values on the grid including both boundary
points; the grid itself is attached as ``details["x"]``.

Problem schema::

    {
        "type": "bvp",
        "method": "finite_difference",
        "p_expr": "0",            # coefficient of y'
        "q_expr": "1",            # coefficient of y
        "r_expr": "x",            # right-hand side
        "variable": "x",
        "a": 0.0, "b": 1.0,
        "alpha": 0.0, "beta": 1.0,
        "n": 10
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

import mathsteps.domains.linalg.gaussian_elimination  # noqa: F401  (registers solver)
from mathsteps.core.expr import parse_expr
from mathsteps.core.registry import find_solver, register
from mathsteps.core.solver_base import Solver
from mathsteps.core.step import Step


@register
class FiniteDifferenceBVPSolver(Solver):
    name = "finite_difference"

    def can_solve(self, problem: dict) -> bool:
        return (
            problem.get("type") == "bvp"
            and problem.get("method") == "finite_difference"
        )

    def solve(self, problem: dict) -> tuple[list[Step], Any]:
        x = sp.symbols(problem["variable"])
        p = parse_expr(problem["p_expr"], {problem["variable"]: x})
        q = parse_expr(problem["q_expr"], {problem["variable"]: x})
        r = parse_expr(problem["r_expr"], {problem["variable"]: x})
        a = float(problem["a"])
        b = float(problem["b"])
        alpha = float(problem["alpha"])
        beta = float(problem["beta"])
        n = int(problem["n"])
        if n < 2:
            raise ValueError("n must be >= 2 for a meaningful BVP.")

        p_num = sp.lambdify(x, p, modules=["numpy"])
        q_num = sp.lambdify(x, q, modules=["numpy"])
        r_num = sp.lambdify(x, r, modules=["numpy"])

        h = (b - a) / (n + 1)
        xs = np.linspace(a, b, n + 2)
        A = np.zeros((n, n), dtype=float)
        rhs = np.zeros(n, dtype=float)

        steps: list[Step] = [
            Step(
                description=(
                    f"Finite-difference BVP solver: -y'' + ({p}) y' + ({q}) y = ({r}); "
                    f"y({a}) = {alpha}, y({b}) = {beta}; n = {n}, h = {h}."
                ),
                before="",
                after=f"interior grid: {n} points",
            )
        ]

        for i in range(n):
            xi = xs[i + 1]
            pi = float(p_num(xi))
            qi = float(q_num(xi))
            ri = float(r_num(xi))

            diag_main = 2.0 / (h * h) + qi
            diag_up = -1.0 / (h * h) + pi / (2 * h)
            diag_down = -1.0 / (h * h) - pi / (2 * h)

            if i > 0:
                A[i, i - 1] = diag_down
            if i < n - 1:
                A[i, i + 1] = diag_up
            A[i, i] = diag_main
            rhs[i] = ri

            if i == 0:
                rhs[i] -= diag_down * alpha
            if i == n - 1:
                rhs[i] -= diag_up * beta

        steps.append(
            Step(
                description=(
                    f"Built (n={n}) x (n={n}) tridiagonal system. "
                    f"Boundary contributions added to rhs: alpha={alpha}, beta={beta}."
                ),
                before="",
                after=f"system size = {n}",
            )
        )

        ge = find_solver({"type": "linear_system", "A": A.tolist(), "b": rhs.tolist()})
        _, y_interior = ge.solve({"type": "linear_system", "A": A.tolist(), "b": rhs.tolist()})
        y_interior = [float(v) for v in y_interior]

        ys = np.array([alpha] + list(y_interior) + [beta])
        steps.append(
            Step(
                description=(
                    f"Solved linear system via Gaussian elimination. "
                    f"Approximate solution on the {n + 2}-point grid: y ≈ {ys.tolist()}."
                ),
                before="",
                after=f"y ≈ {ys.tolist()}",
                data={
                    "answer": ys,
                    "x_grid": xs.tolist(),
                    "y": ys.tolist(),
                    "details": {"x": xs, "y": ys},
                },
            )
        )
        return steps, ys

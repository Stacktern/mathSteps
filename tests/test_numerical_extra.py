"""Tests for the additional numerical solvers."""
import json
from pathlib import Path

import numpy as np
import sympy as sp

import mathsteps.domains.numerical.secant  # noqa: F401
import mathsteps.domains.numerical.fixed_point  # noqa: F401
import mathsteps.domains.numerical.numerical_diff  # noqa: F401
import mathsteps.domains.numerical.numerical_integration  # noqa: F401
import mathsteps.domains.numerical.lagrange  # noqa: F401
import mathsteps.domains.numerical.newton_divided_differences  # noqa: F401
from mathsteps.core.registry import find_solver

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_secant_cubic_example():
    problem = json.loads((EXAMPLES / "secant_cubic.json").read_text())
    _, answer = _solve(problem)
    assert abs(float(answer) - 1.5213797068049677) < 1e-6


def test_fixed_point_dottie_number():
    problem = json.loads((EXAMPLES / "fixed_point_dottie.json").read_text())
    _, answer = _solve(problem)
    assert abs(float(answer) - 0.7390851332151607) < 1e-9


def test_numerical_diff_central():
    problem = json.loads((EXAMPLES / "numdiff_sin_central.json").read_text())
    _, answer = _solve(problem)
    assert abs(float(answer) - np.cos(1.0)) < 1e-5


def test_simpson_integration_of_sin():
    problem = json.loads((EXAMPLES / "int_simpson_sin.json").read_text())
    _, answer = _solve(problem)
    assert abs(float(answer) - 2.0) < 1e-6


def test_lagrange_three_points():
    problem = json.loads((EXAMPLES / "lagrange_three_points.json").read_text())
    _, answer = _solve(problem)
    poly = sp.sympify(answer)
    x = sp.symbols("x")
    for px, py in [(0, 1), (1, 2), (2, 5)]:
        assert abs(float(poly.subs(x, px)) - float(py)) < 1e-9


def test_newton_divided_differences_matches_lagrange():
    """Both interpolators must produce the same polynomial for the same points."""
    newt = json.loads((EXAMPLES / "newton_dd_four_points.json").read_text())
    _, p_newt = _solve(newt)
    poly_newt = sp.expand(sp.sympify(p_newt))

    lag_problem = {
        "type": "interpolation", "method": "lagrange",
        "points": newt["points"],
    }
    _, p_lag = _solve(lag_problem)
    poly_lag = sp.expand(sp.sympify(p_lag))

    x = sp.symbols("x")
    diff = sp.simplify(poly_newt - poly_lag)
    coeffs = sp.Poly(diff, x).all_coeffs() if diff != 0 else [0]
    assert all(abs(float(c)) < 1e-9 for c in coeffs)


def test_newton_divided_differences_pass_through_points():
    problem = json.loads((EXAMPLES / "newton_dd_four_points.json").read_text())
    _, answer = _solve(problem)
    poly = sp.sympify(answer)
    x = sp.symbols("x")
    for px, py in problem["points"]:
        assert abs(float(poly.subs(x, px)) - float(py)) < 1e-9

"""Tests for bisection and Newton-Raphson root finders."""
import json
from pathlib import Path

import sympy as sp

import mathsteps.domains.numerical.bisection  # noqa: F401  (registers solver)
import mathsteps.domains.numerical.newton_raphson  # noqa: F401
from mathsteps.core.registry import find_solver
from mathsteps.verify import verify_root, verify_root_against_scipy

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_bisection_cubic_example():
    problem = json.loads((EXAMPLES / "bisection_cubic.json").read_text())
    steps, answer = _solve(problem)
    x = float(answer)
    assert verify_root(problem["function"], problem["variable"], x, tol=1e-8)
    assert verify_root_against_scipy(
        problem["function"], problem["variable"], x,
        bracket=(problem["a"], problem["b"]),
    )
    assert any("midpoint" in s.description.lower() for s in steps)


def test_newton_cos_minus_x_example():
    problem = json.loads((EXAMPLES / "newton_cos_minus_x.json").read_text())
    steps, answer = _solve(problem)
    x = float(answer)
    assert verify_root(problem["function"], problem["variable"], x, tol=1e-10)
    assert verify_root_against_scipy(
        problem["function"], problem["variable"], x, x0=problem["x0"],
    )
    assert any("f'(x)" in s.description or "f'(x) =" in s.description for s in steps)


def test_bisection_initial_bracket_validation():
    import pytest

    problem = {
        "type": "root_finding", "method": "bisection",
        "function": "x**2 + 1", "variable": "x",
        "a": -1.0, "b": 1.0,
    }
    with pytest.raises(ValueError, match="opposite signs"):
        _solve(problem)


def test_bisection_exact_endpoint_root():
    problem = {
        "type": "root_finding", "method": "bisection",
        "function": "x - 3", "variable": "x",
        "a": 3.0, "b": 5.0,
    }
    steps, answer = _solve(problem)
    assert float(answer) == 3.0
    assert any("already a root" in s.description.lower() for s in steps)


def test_newton_zero_derivative_raises():
    import pytest

    problem = {
        "type": "root_finding", "method": "newton_raphson",
        "function": "x**3", "variable": "x",
        "x0": 0.0,
    }
    with pytest.raises(ValueError, match="f'.*= 0"):
        _solve(problem)


def test_newton_raphson_quadratic_exact():
    problem = {
        "type": "root_finding", "method": "newton_raphson",
        "function": "x**2 - 2", "variable": "x",
        "x0": 1.0, "tol": 1e-14, "max_iter": 50,
    }
    _, answer = _solve(problem)
    assert abs(float(answer) - float(sp.sqrt(2))) < 1e-12


def test_bisection_emits_step_data():
    problem = {
        "type": "root_finding", "method": "bisection",
        "function": "x**3 - 10", "variable": "x",
        "a": 0.0, "b": 5.0, "tol": 1e-6, "max_iter": 50,
    }
    steps, _ = _solve(problem)
    iter_steps = [s for s in steps if s.data.get("iter") is not None]
    assert len(iter_steps) >= 10
    assert iter_steps[0].data["a"] == 0.0
    assert iter_steps[0].data["b"] == 5.0

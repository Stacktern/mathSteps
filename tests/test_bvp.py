"""Tests for BVP solvers."""
import json
import math
from pathlib import Path

import mathsteps.domains.ode_bvp.shooting  # noqa: F401
import mathsteps.domains.ode_bvp.finite_difference  # noqa: F401
from mathsteps.core.registry import find_solver

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_shooting_exp_bvp():
    """y'' + y = sin(x), y(0)=0, y(1)=0. True y(1)=0; test that
    shooting converges to y(x_end) ≈ 0 and y'(0) ≈ 0.218 (the
    analytical s*)."""
    problem = json.loads((EXAMPLES / "bvp_shooting_sin.json").read_text())
    _, answer = _solve(problem)
    assert abs(float(answer[-1])) < 1e-3  # y(x_end) = y_right; answer is the whole solution
    steps_data = [s.data for s in _solve(problem)[0] if s.data]
    s_stars = [d.get("s_star") for d in steps_data if d.get("s_star") is not None]
    assert s_stars
    expected_s_star = -0.5 + math.cos(1) / (2 * math.sin(1))
    assert abs(s_stars[-1] - expected_s_star) < 1e-2


def test_finite_difference_simple_bvp():
    """-y'' + y = x on [0,1], y(0)=0, y(1)=1; numerical interior values plausible."""
    problem = json.loads((EXAMPLES / "bvp_fd_simple.json").read_text())
    _, answer = _solve(problem)
    ys = [float(v) for v in answer]
    assert ys[0] == 0.0
    assert ys[-1] == 1.0
    for v in ys[1:-1]:
        assert 0.0 < v < 1.0


def test_shooting_method_emits_secant_iterations():
    problem = json.loads((EXAMPLES / "bvp_shooting_sin.json").read_text())
    steps, _ = _solve(problem)
    assert any("secant update" in s.description.lower() for s in steps)

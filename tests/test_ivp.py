"""Tests for IVP solvers."""
import json
from pathlib import Path

import mathsteps.domains.ode_ivp.euler  # noqa: F401
import mathsteps.domains.ode_ivp.heun  # noqa: F401
import mathsteps.domains.ode_ivp.rk4  # noqa: F401
import mathsteps.domains.ode_ivp.midpoint  # noqa: F401
import mathsteps.domains.ode_ivp.rk45  # noqa: F401
from mathsteps.core.registry import find_solver

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_euler_exp_decay():
    import math
    problem = json.loads((EXAMPLES / "ivp_euler_exp_decay.json").read_text())
    _, answer = _solve(problem)
    expected = math.exp(-2 ** 2 + 0 ** 2)
    assert abs(float(answer) - expected) < 0.1


def test_heun_exp_decay():
    import math
    problem = json.loads((EXAMPLES / "ivp_heun_exp_decay.json").read_text())
    _, answer = _solve(problem)
    expected = math.exp(-(2 ** 2))
    assert abs(float(answer) - expected) < 5e-3


def test_rk4_exp_decay_high_accuracy():
    import math
    problem = json.loads((EXAMPLES / "ivp_rk4_exp_decay.json").read_text())
    _, answer = _solve(problem)
    expected = math.exp(-(2 ** 2))
    assert abs(float(answer) - expected) < 1e-4


def test_midpoint_exp_decay():
    import math
    problem = json.loads((EXAMPLES / "ivp_midpoint_exp_decay.json").read_text())
    _, answer = _solve(problem)
    expected = math.exp(-(2 ** 2))
    assert abs(float(answer) - expected) < 1e-2


def test_rk45_exp_decay_high_accuracy():
    import math
    problem = json.loads((EXAMPLES / "ivp_rk45_exp_decay.json").read_text())
    _, answer = _solve(problem)
    expected = math.exp(-(2 ** 2))
    assert abs(float(answer) - expected) < 1e-6


def test_rk4_steps_have_k1_k2_k3_k4():
    problem = json.loads((EXAMPLES / "ivp_rk4_exp_decay.json").read_text())
    steps, _ = _solve(problem)
    iter_steps = [s for s in steps if s.data.get("step")]
    assert iter_steps
    for s in iter_steps:
        assert all(k in s.data for k in ("k1", "k2", "k3", "k4"))

"""Smoke tests for the solver registry."""
import mathsteps.domains.linalg.gaussian_elimination  # noqa: F401  (registers solver)
import mathsteps.domains.numerical.bisection  # noqa: F401
import mathsteps.domains.numerical.newton_raphson  # noqa: F401
from mathsteps.core.registry import all_solvers, find_solver


def test_core_solvers_are_registered():
    names = {s.name for s in all_solvers()}
    assert {"gaussian_elimination", "bisection", "newton_raphson"} <= names


def test_find_solver_picks_correct_one():
    solver = find_solver({"type": "linear_system", "A": [[1]], "b": [1]})
    assert solver.name == "gaussian_elimination"
    solver = find_solver(
        {"type": "root_finding", "method": "bisection",
         "function": "x", "variable": "x", "a": -1, "b": 1}
    )
    assert solver.name == "bisection"
    solver = find_solver(
        {"type": "root_finding", "method": "newton_raphson",
         "function": "x", "variable": "x", "x0": 0}
    )
    assert solver.name == "newton_raphson"


def test_find_solver_raises_on_unknown_type():
    import pytest

    with pytest.raises(ValueError):
        find_solver({"type": "definitely_not_supported"})

"""Tests for the Gaussian elimination solver."""
import json
from pathlib import Path

import numpy as np
import sympy as sp

from mathsteps.core.registry import find_solver
from mathsteps.domains.linalg.gaussian_elimination import GaussianEliminationSolver
from mathsteps.verify import (
    verify_linear_system,
    verify_linear_system_against_numpy,
)


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_2x2_example():
    problem = json.loads((EXAMPLES / "linear_system_2x2.json").read_text())
    steps, answer = _solve(problem)
    assert len(steps) >= 4
    assert verify_linear_system(problem["A"], problem["b"], answer)
    assert verify_linear_system_against_numpy(problem["A"], problem["b"], answer)


def test_3x3_example():
    problem = json.loads((EXAMPLES / "linear_system_3x3.json").read_text())
    steps, answer = _solve(problem)
    assert len(steps) >= 6
    assert verify_linear_system(problem["A"], problem["b"], answer)
    assert verify_linear_system_against_numpy(problem["A"], problem["b"], answer)


def test_fractional_solution_is_exact():
    problem = {
        "type": "linear_system",
        "A": [[2, 1], [1, 1]],
        "b": [1, 0],
    }
    steps, answer = _solve(problem)
    assert sp.nsimplify(answer[0]) == 1
    assert sp.nsimplify(answer[1]) == -1


def test_partial_pivoting_triggers_swap():
    problem = {
        "type": "linear_system",
        "A": [[0, 1], [1, 1]],
        "b": [1, 2],
    }
    steps, _ = _solve(problem)
    swap_step = next(s for s in steps if "Swap" in s.description)
    assert "partial pivoting" in swap_step.description.lower()
    assert swap_step.data["swap"] == (0, 1)


def test_steps_have_descriptions():
    problem = {
        "type": "linear_system",
        "A": [[3, 2, -1], [2, -2, 4], [1, 0, 2]],
        "b": [1, -2, 3],
    }
    steps, _ = _solve(problem)
    assert all(s.description for s in steps)
    assert any("augmented matrix" in s.description.lower() for s in steps)
    assert any("back-substitution" in s.description.lower() for s in steps)


def test_cannot_solve_other_types():
    solver = GaussianEliminationSolver
    assert not solver.can_solve({"type": "root_finding"})
    assert solver.can_solve({"type": "linear_system"})

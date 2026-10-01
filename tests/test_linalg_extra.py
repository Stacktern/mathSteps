"""Tests for the additional linear-algebra solvers."""
import json
from pathlib import Path

import numpy as np

import mathsteps.domains.linalg.gauss_jordan_inverse  # noqa: F401
import mathsteps.domains.linalg.cofactor_determinant  # noqa: F401
import mathsteps.domains.linalg.lu_decomposition  # noqa: F401
import mathsteps.domains.linalg.eigenvalues  # noqa: F401
import mathsteps.domains.linalg.cramers_rule  # noqa: F401
from mathsteps.core.registry import find_solver

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _solve(problem: dict):
    solver = find_solver(problem)
    return solver.solve(problem)


def test_gauss_jordan_inverse_3x3():
    problem = json.loads((EXAMPLES / "inverse_3x3.json").read_text())
    _, answer = _solve(problem)
    inv = np.asarray(answer.tolist() if hasattr(answer, "tolist") else answer, dtype=float)
    A = np.asarray(problem["A"], dtype=float)
    assert np.allclose(A @ inv, np.eye(3), atol=1e-9)


def test_cofactor_determinant():
    problem = json.loads((EXAMPLES / "determinant_3x3.json").read_text())
    _, answer = _solve(problem)
    A = np.asarray(problem["A"], dtype=float)
    assert abs(float(answer) - np.linalg.det(A)) < 1e-9


def test_lu_decomposition():
    problem = json.loads((EXAMPLES / "lu_3x3.json").read_text())
    _, answer = _solve(problem)
    P, L, U = (np.asarray(v.tolist() if hasattr(v, "tolist") else v, dtype=float) for v in answer)
    A = np.asarray(problem["A"], dtype=float)
    assert np.allclose(P @ A, L @ U, atol=1e-9)


def test_eigenvalues_match_numpy():
    problem = json.loads((EXAMPLES / "eigenvalues_3x3.json").read_text())
    _, answer = _solve(problem)
    eigs, _ = answer
    ours = sorted([float(ev) for ev in eigs])
    A = np.asarray(problem["A"], dtype=float)
    ref = sorted(np.linalg.eigvals(A).real.tolist())
    assert np.allclose(ours, ref, atol=1e-9)


def test_cramers_rule_matches_gaussian():
    problem = json.loads((EXAMPLES / "cramers_3x3.json").read_text())
    _, answer = _solve(problem)
    A = np.asarray(problem["A"], dtype=float)
    b = np.asarray(problem["b"], dtype=float)
    ref = np.linalg.solve(A, b)
    ours = np.asarray([float(v) for v in answer], dtype=float)
    assert np.allclose(ours, ref, atol=1e-9)

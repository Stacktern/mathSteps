"""Tests for the public Python API (mathsteps.api)."""
import math

import numpy as np
import sympy as sp


def test_linear_system_returns_correct_answer():
    r = __import__("mathsteps").linear_system([[1, 2], [3, 4]], [5, 11])
    assert r.solver == "gaussian_elimination"
    assert r.verified
    assert [float(v) for v in r.answer] == [1.0, 2.0]
    assert len(r.steps) >= 4


def test_linear_system_3x3():
    r = __import__("mathsteps").linear_system(
        [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
        [8, -11, -3],
    )
    assert [float(v) for v in r.answer] == [2.0, 3.0, -1.0]
    assert r.verified


def test_root_newton_finds_dottie_number():
    r = __import__("mathsteps").root("cos(x) - x", method="newton", x0=0.0)
    assert r.solver == "newton_raphson"
    assert r.verified
    assert abs(float(r.answer) - 0.7390851332151607) < 1e-9


def test_root_newton_alias():
    r = __import__("mathsteps").root(
        "cos(x) - x", method="newton_raphson", x0=0.0,
    )
    assert r.solver == "newton_raphson"


def test_root_bisection():
    r = __import__("mathsteps").root("x**3 - x - 2", method="bisection", a=1, b=2)
    assert r.solver == "bisection"
    assert abs(float(r.answer) - 1.5213797068049677) < 1e-6


def test_root_secant():
    r = __import__("mathsteps").root(
        "x**3 - x - 2", method="secant", x0=1.0, x1=2.0,
    )
    assert r.solver == "secant"
    assert abs(float(r.answer) - 1.5213797068049677) < 1e-6


def test_root_bisection_missing_bracket():
    import pytest

    with pytest.raises(ValueError, match="bracket"):
        __import__("mathsteps").root("x**3 - x - 2", method="bisection")


def test_root_secant_missing_x1():
    import pytest

    with pytest.raises(ValueError, match="x1"):
        __import__("mathsteps").root(
            "x**3 - x - 2", method="secant", x0=1.0,
        )


def test_ivp_rk4():
    r = __import__("mathsteps").ivp(
        "-2*x*y", y0=1, x_end=2, h=0.1, method="rk4",
    )
    assert r.solver == "rk4"
    assert r.verified
    assert abs(float(r.answer) - math.exp(-4)) < 1e-3


def test_ivp_euler():
    r = __import__("mathsteps").ivp(
        "-2*x*y", y0=1, x_end=1, h=0.05, method="euler",
    )
    assert r.solver == "euler"
    assert r.verified


def test_ivp_rk45_no_h_required():
    r = __import__("mathsteps").ivp(
        "-2*x*y", y0=1, x_end=2, method="rk45",
    )
    assert r.solver == "rk45"


def test_ivp_requires_h_for_fixed_step():
    import pytest

    with pytest.raises(ValueError, match="requires `h`"):
        __import__("mathsteps").ivp("-2*x*y", y0=1, x_end=1, method="euler")


def test_bvp_shooting():
    r = __import__("mathsteps").bvp_shooting(
        "sin(x) - y", y_left=0, y_right=0, x_end=1,
    )
    assert r.solver == "shooting"
    assert r.converged is True
    assert r.verified is True  # checked against scipy.integrate.solve_bvp
    # The answer is the whole solution on the grid, not just the boundary value.
    assert r.answer.shape == r.details["x"].shape
    assert r.answer[0] == 0.0 and abs(r.answer[-1]) < 1e-3


def test_bvp_finite_difference():
    r = __import__("mathsteps").bvp_finite_difference(
        q_expr="1", r_expr="x", n=10,
    )
    assert r.solver == "finite_difference"
    ys = r.answer
    assert ys[0] == 0.0
    assert ys[-1] == 0.0


def test_determinant():
    r = __import__("mathsteps").determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    assert r.solver == "cofactor_determinant"
    assert r.verified
    assert float(r.answer) == -306


def test_inverse():
    A = [[1, 2], [3, 4]]
    r = __import__("mathsteps").inverse(A)
    assert r.solver == "gauss_jordan_inverse"
    assert r.verified
    inv = np.asarray(r.answer.tolist(), dtype=float)
    assert np.allclose(np.array(A) @ inv, np.eye(2), atol=1e-9)


def test_lu():
    A = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
    r = __import__("mathsteps").lu(A)
    assert r.solver == "lu_decomposition"
    P, L, U = (
        np.asarray(v.tolist(), dtype=float) if hasattr(v, "tolist")
        else np.asarray(v, dtype=float)
        for v in r.answer
    )
    assert np.allclose(P @ np.array(A, dtype=float), L @ U, atol=1e-9)


def test_eigenvalues():
    A = [[2, 0, 0], [0, 3, 4], [0, 4, 9]]
    r = __import__("mathsteps").eigenvalues(A)
    eigs, _ = r.answer
    ours = sorted(float(ev) for ev in eigs)
    ref = sorted(np.linalg.eigvals(A).real.tolist())
    assert np.allclose(ours, ref, atol=1e-9)


def test_cramers_rule():
    r = __import__("mathsteps").cramers_rule(
        [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3],
    )
    assert r.solver == "cramers_rule"
    ours = np.array([float(v) for v in r.answer])
    ref = np.linalg.solve(np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]),
                          np.array([8, -11, -3]))
    assert np.allclose(ours, ref, atol=1e-9)


def test_lagrange():
    r = __import__("mathsteps").lagrange([[0, 1], [1, 2], [2, 5]])
    assert r.solver == "lagrange"
    poly = sp.sympify(r.answer)
    x = sp.symbols("x")
    for px, py in [(0, 1), (1, 2), (2, 5)]:
        assert abs(float(poly.subs(x, px)) - float(py)) < 1e-9


def test_newton_divided_differences():
    r = __import__("mathsteps").newton_divided_differences(
        [[0, 1], [1, 2], [2, 5], [3, 10]],
    )
    assert r.solver == "newton_divided_differences"


def test_integrate_simpson():
    r = __import__("mathsteps").integrate(
        "sin(x)", a=0, b=math.pi, n=100, method="simpson",
    )
    assert r.solver == "numerical_integration"
    assert abs(float(r.answer) - 2.0) < 1e-6


def test_integrate_trapezoidal():
    r = __import__("mathsteps").integrate(
        "sin(x)", a=0, b=math.pi, n=1000, method="trapezoidal",
    )
    assert abs(float(r.answer) - 2.0) < 1e-3


def test_differentiate_central():
    r = __import__("mathsteps").differentiate(
        "sin(x)", x=1.0, h=0.001, method="central",
    )
    assert abs(float(r.answer) - math.cos(1.0)) < 1e-5


def test_differentiate_forward():
    r = __import__("mathsteps").differentiate(
        "sin(x)", x=1.0, h=0.001, method="forward",
    )
    assert abs(float(r.answer) - math.cos(1.0)) < 1e-3


def test_solve_problem_generic_root():
    r = __import__("mathsteps").solve_problem({
        "type": "root_finding", "method": "newton_raphson",
        "function": "x**2 - 2", "variable": "x", "x0": 1.0,
        "tol": 1e-12, "max_iter": 50,
    })
    assert abs(float(r.answer) - math.sqrt(2)) < 1e-10


def test_solve_problem_linear_system():
    r = __import__("mathsteps").solve_problem({
        "type": "linear_system", "A": [[1, 2], [3, 4]], "b": [5, 11],
    })
    assert [float(v) for v in r.answer] == [1.0, 2.0]


def test_available_solvers_includes_core_set():
    names = set(__import__("mathsteps").available_solvers())
    for required in (
        "gaussian_elimination", "newton_raphson", "rk4", "lagrange",
    ):
        assert required in names


def test_result_str_is_human_readable():
    r = __import__("mathsteps").linear_system([[1, 2], [3, 4]], [5, 11])
    text = str(r)
    assert "Solver: gaussian_elimination" in text
    assert "Final answer" in text
    assert "Verification" in text


def test_result_repr_is_concise():
    r = __import__("mathsteps").root("x**2 - 2", method="newton", x0=1.0)
    rep = repr(r)
    assert "Result(" in rep
    assert "verified" in rep


def test_verify_can_be_disabled():
    r = __import__("mathsteps").linear_system(
        [[1, 2], [3, 4]], [5, 11], verify=False,
    )
    assert r.verified is None  # not checked, which is different from "failed"
    assert [float(v) for v in r.answer] == [1.0, 2.0]


def test_importing_mathsteps_registers_all_solvers():
    """A bare ``import mathsteps`` should make every solver callable."""
    import mathsteps

    r = mathsteps.linear_system([[1, 1], [1, 2]], [3, 5])
    assert r.verified

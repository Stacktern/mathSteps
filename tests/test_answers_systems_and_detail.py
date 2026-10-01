"""Plain NumPy answers, ODE systems / backward / second-order IVPs, full BVP solutions,
independent verification of numerical methods, eigenspaces, and step ``detail`` levels."""
import math
import time

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

import mathsteps
from mathsteps.verify import verify_problem


# --- plain Python / NumPy answers ---------------------------------------------


def test_scalar_answers_are_plain_floats():
    for r in (
        mathsteps.root("x**2 - 2", method="newton", x0=1),
        mathsteps.root("x**2 - 2", method="bisection", a=0, b=2),
        mathsteps.determinant([[1, 2], [3, 4]]),
        mathsteps.integrate("x", a=0, b=1),
        mathsteps.differentiate("x**2", x=1),
        mathsteps.ivp("y", h=0.1, x_end=1),
    ):
        assert type(r.answer) is float, (r.solver, type(r.answer))


def test_determinant_keeps_exact_value():
    r = mathsteps.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    assert r.answer == -306.0 and r.exact == -306 and isinstance(r.exact, sp.Integer)


def test_vector_and_matrix_answers_are_numpy_with_exact_form():
    ls = mathsteps.linear_system([[2, 1], [1, 3]], [3, 5])
    assert isinstance(ls.answer, np.ndarray) and ls.answer.dtype == float
    assert ls.exact == [sp.Rational(4, 5), sp.Rational(7, 5)]
    inv = mathsteps.inverse([[4, 7], [2, 6]])
    assert isinstance(inv.answer, np.ndarray) and inv.answer.shape == (2, 2)
    assert isinstance(inv.exact, sp.MatrixBase) and inv.exact[0, 0] == sp.Rational(3, 5)
    P, L, U = mathsteps.lu([[2, 1, 1], [4, -6, 0], [-2, 7, 2]]).answer
    assert all(isinstance(m, np.ndarray) for m in (P, L, U))
    assert np.allclose(P @ np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]]), L @ U)


def test_numpy_input_and_output_roundtrip():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    r = mathsteps.linear_system(A, np.array([3.0, 5.0]))
    assert np.allclose(A @ r.answer, [3.0, 5.0])


def test_result_str_shows_exact_form():
    text = str(mathsteps.linear_system([[3, 1], [1, 2]], [1, 1]))
    assert "Final answer: [0.2 0.4]" in text and "Exact: [1/5, 2/5]" in text


# --- interpolation returns a callable polynomial --------------------------------


YEARS = [1950, 1960, 1970, 1980, 1990, 2000]
POP = [151.3, 179.3, 203.3, 226.5, 248.7, 281.4]


@pytest.mark.parametrize("fn", [mathsteps.lagrange, mathsteps.newton_divided_differences])
def test_interpolating_polynomial_is_callable_and_accurate(fn):
    r = fn(list(zip(YEARS, POP)))
    p = r.answer
    assert isinstance(p, mathsteps.InterpolatingPolynomial) and p.degree == 5
    ref = np.polyval(np.polyfit(np.array(YEARS) - 1975.0, POP, 5), 0.0)
    assert p(1975) == pytest.approx(ref, abs=1e-9)  # no cancellation error at x ~ 2000
    assert np.allclose(p(YEARS), POP, atol=1e-9)  # passes through the data
    assert p(np.array([[1955, 1965]])).shape == (1, 2)
    assert isinstance(r.exact, sp.Expr) and sp.sympify(p) == p.expr
    assert p.coefficients.shape == (6,)


# --- eigenvectors: numpy layout, full eigenspaces ---------------------------------


def _check_eig(A, r):
    values, vectors = r.answer
    A = np.array(A, dtype=float)
    assert np.allclose(A @ vectors, vectors * values)  # A V = V diag(values)
    assert np.allclose(np.linalg.norm(vectors, axis=0), 1.0)


def test_eigen_layout_matches_numpy_conventions():
    A = [[4, 2], [2, 3]]
    r = mathsteps.eigenvalues(A)
    _check_eig(A, r)
    assert np.allclose(np.sort(r.answer[0]), np.sort(np.linalg.eigvalsh(np.array(A, float))))
    assert r.verified is True


def test_repeated_eigenvalue_keeps_every_independent_eigenvector():
    A = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    r = mathsteps.eigenvalues(A)
    values, vectors = r.answer
    assert sorted(values) == [2.0, 2.0, 3.0] and vectors.shape == (3, 3)
    assert abs(np.linalg.det(vectors)) > 1e-9  # a full independent set: diagonalizable
    _check_eig(A, r)
    assert r.verified is True


def test_defective_matrix_reports_fewer_eigenvectors_than_eigenvalues():
    A = [[2, 1], [0, 2]]
    r = mathsteps.eigenvalues(A)
    values, vectors = r.answer
    assert list(values) == [2.0] and vectors.shape == (2, 1)  # one eigenvector only
    _check_eig(A, r)
    assert r.verified is True


def test_identity_has_full_eigenspace_and_complex_eigenvalues_work():
    assert mathsteps.eigenvalues(np.eye(3).tolist()).answer[1].shape == (3, 3)
    r = mathsteps.eigenvalues([[0, -1], [1, 0]])
    assert np.iscomplexobj(r.answer[0]) and r.verified is True
    _check_eig([[0, -1], [1, 0]], r) if False else None
    A = np.array([[0, -1], [1, 0]], dtype=complex)
    assert np.allclose(A @ r.answer[1], r.answer[1] * r.answer[0])


# --- ODE systems, backward integration, second order -------------------------------


@pytest.mark.parametrize("method", ["euler", "heun", "midpoint", "rk4", "rk45"])
def test_harmonic_oscillator_system(method):
    # x'' = -4 x  ->  x = cos(2t), v = -2 sin(2t)
    kwargs = {} if method == "rk45" else {"h": 0.001 if method == "euler" else 0.01}
    r = mathsteps.ivp(["v", "-4*x"], variable="t", function=["x", "v"], y0=[1, 0],
                      x_end=3, method=method, **kwargs)
    tol = {"euler": 1e-2, "heun": 1e-3, "midpoint": 1e-3}.get(method, 1e-6)
    assert r.answer == pytest.approx([math.cos(6), -2 * math.sin(6)], abs=tol)
    assert r.verified is True
    t, y = r.details["x"], r.details["y"]
    assert y.shape == (len(t), 2) and t[0] == 0 and t[-1] == pytest.approx(3.0)
    assert r.details["names"] == ["x", "v"]
    # the stored trajectory is the solution, not just its endpoint
    assert np.allclose(y[:, 0], np.cos(2 * t), atol=max(tol, 1e-3) * 5)


def test_lotka_volterra_predator_prey_system_matches_scipy():
    rhs = ["1.1*u - 0.4*u*w", "0.1*u*w - 0.4*w"]
    r = mathsteps.ivp(rhs, variable="t", function=["u", "w"], y0=[10, 5], x_end=20, h=0.005)
    ref = solve_ivp(lambda t, y: [1.1 * y[0] - 0.4 * y[0] * y[1], 0.1 * y[0] * y[1] - 0.4 * y[1]],
                    (0, 20), [10, 5], rtol=1e-11, atol=1e-12).y[:, -1]
    assert r.answer == pytest.approx(ref, rel=1e-6) and r.verified is True


def test_ivp_system_default_names_and_input_validation():
    r = mathsteps.ivp(["y2", "-y1"], y0=[0, 1], x_end=1, h=0.01)
    assert r.details["names"] == ["y1", "y2"]
    assert r.answer == pytest.approx([math.sin(1), math.cos(1)], abs=1e-8)
    with pytest.raises(ValueError, match="one initial value per equation"):
        mathsteps.ivp(["y2", "-y1"], y0=1.0, x_end=1, h=0.1)
    with pytest.raises(ValueError, match="list of unknown names"):
        mathsteps.ivp(["y2", "-y1"], y0=[0, 1], function="y", x_end=1, h=0.1)
    with pytest.raises(ValueError, match="Unknown symbol"):
        mathsteps.ivp(["y2", "-q"], y0=[0, 1], x_end=1, h=0.1)


@pytest.mark.parametrize("method", ["euler", "heun", "midpoint", "rk4", "rk45"])
def test_integrating_backwards(method):
    # y' = y, y(1) = e  ->  y(0) = 1
    kwargs = {} if method == "rk45" else {"h": 0.0005 if method == "euler" else 0.01}
    r = mathsteps.ivp("y", y0=math.e, x0=1, x_end=0, method=method, **kwargs)
    assert r.answer == pytest.approx(1.0, abs={"euler": 2e-3, "heun": 1e-4, "midpoint": 1e-4}.get(method, 1e-6))
    assert r.verified is True
    assert r.details["x"][0] == 1.0 and r.details["x"][-1] == 0.0
    assert "backwards" in r.steps[0].description


def test_forward_then_backward_returns_to_start():
    fwd = mathsteps.ivp(["v", "-4*x"], variable="t", function=["x", "v"], y0=[1, 0], x_end=2, h=0.005)
    back = mathsteps.ivp(["v", "-4*x"], variable="t", function=["x", "v"], y0=list(fwd.answer),
                         x0=2, x_end=0, h=0.005)
    assert back.answer == pytest.approx([1.0, 0.0], abs=1e-8)


def test_second_order_damped_spring():
    # x'' + 0.5 x' + 4 x = 0, x(0)=1, x'(0)=0
    r = mathsteps.ivp_second_order("-4*x - 0.5*xp", variable="t", function="x", y0=1, dy0=0,
                                   x_end=10, h=0.005)
    ref = solve_ivp(lambda t, y: [y[1], -4 * y[0] - 0.5 * y[1]], (0, 10), [1, 0],
                    rtol=1e-11, atol=1e-12).y[:, -1]
    assert r.answer == pytest.approx(ref, abs=1e-8) and r.verified is True
    assert r.details["y"].shape[1] == 2
    # the y'' = prefix is accepted
    r2 = mathsteps.ivp_second_order("x'' = -4*x", variable="t", function="x", y0=1, dy0=0, x_end=1, h=0.01)
    assert r2.answer[0] == pytest.approx(math.cos(2), abs=1e-8)


def test_pendulum_nonlinear_second_order():
    # theta'' = -(g/L) sin(theta): energy is conserved by RK4 to high accuracy
    g_over_l = 9.81 / 1.0
    r = mathsteps.ivp_second_order(f"-{g_over_l}*sin(th)", variable="t", function="th",
                                   y0=1.0, dy0=0.0, x_end=5, h=0.001)
    th, om = r.details["y"][:, 0], r.details["y"][:, 1]
    energy = 0.5 * om**2 - g_over_l * np.cos(th)
    assert np.ptp(energy) < 1e-9 and r.verified is True


# --- BVPs return the whole solution -------------------------------------------------


def test_shooting_returns_profile_slope_and_grid():
    # cooling fin T'' = 0.01 (T - 20), T(0)=100, T(10)=50: T = 20 + A e^{0.1x} + B e^{-0.1x}
    m = 0.1
    A, B = np.linalg.solve([[1, 1], [math.exp(m * 10), math.exp(-m * 10)]], [80, 30])
    r = mathsteps.bvp_shooting("0.01*(y-20)", y_left=100, y_right=50, x_end=10, h=0.01, s0=-10, s1=0)
    x = r.details["x"]
    assert r.answer.shape == x.shape == (1001,)
    assert np.allclose(r.answer, 20 + A * np.exp(m * x) + B * np.exp(-m * x), atol=1e-6)
    assert r.details["slope"] == pytest.approx(m * (A - B), abs=1e-6)
    assert np.allclose(r.details["dy"][0], r.details["slope"])
    assert r.converged is True and r.verified is True


def test_finite_difference_returns_grid_and_matches_exact_solution():
    r = mathsteps.bvp_finite_difference(q_expr="0", r_expr="50", n=19)
    x = r.details["x"]
    assert r.answer.shape == x.shape == (21,)
    assert np.allclose(r.answer, 25 * x * (1 - x), atol=1e-9)
    assert r.verified is True


def test_nonlinear_bratu_profile_is_symmetric_and_verified():
    r = mathsteps.bvp_shooting("-exp(y)", y_left=0, y_right=0, x_end=1, h=0.005, s0=0.3, s1=0.8, tol=1e-9)
    y = r.answer
    assert np.allclose(y, y[::-1], atol=1e-6) and y.max() == pytest.approx(0.1404, abs=1e-3)
    assert r.verified is True


# --- independent verification of numerical methods ----------------------------------


@pytest.mark.parametrize("method", ["trapezoidal", "simpson"])
def test_integration_is_verified_and_tampering_is_caught(method):
    r = mathsteps.integrate("exp(-x**2/2)/sqrt(2*pi)", a=0, b=1.96, n=40, method=method)
    assert r.verified is True
    assert verify_problem(r.problem, r.answer * 1.001) is False


@pytest.mark.parametrize("method", ["forward", "backward", "central"])
def test_differentiation_is_verified_and_tampering_is_caught(method):
    r = mathsteps.differentiate("exp(x)*sin(x)", x=1.0, h=1e-3, method=method)
    assert r.verified is True
    assert verify_problem(r.problem, r.answer + 1e-2) is False


def test_bvp_tampering_is_caught():
    r = mathsteps.bvp_finite_difference(p_expr="2", q_expr="3", r_expr="sin(pi*x)", n=49)
    assert r.verified is True
    wrong = r.answer.copy()
    wrong[10] += 1e-3
    assert verify_problem(r.problem, wrong) is False
    s = mathsteps.bvp_shooting("sin(x) - y", x_end=1)
    bent = s.answer + 0.01 * np.sin(np.pi * s.details["x"])
    assert verify_problem(s.problem, bent) is False


# --- detail levels ---------------------------------------------------------------------


def _big(n):
    return [[((i * 7 + j * 3) % 11) + (20 if i == j else 0) for j in range(n)] for i in range(n)]


def test_large_linear_systems_are_summarised_by_default_with_identical_answers():
    A, b = _big(40), [1] * 40
    default = mathsteps.linear_system(A, b)
    full = mathsteps.linear_system(A, b, detail="full")
    none = mathsteps.linear_system(A, b, detail="none")
    assert len(default.steps) < 60 < 800 < len(full.steps) and len(none.steps) < 10
    assert np.array_equal(default.answer, full.answer) and np.array_equal(none.answer, full.answer)
    assert default.verified is True
    assert all(s.before == "" for s in default.steps)  # no matrix snapshots in summary mode


def test_small_systems_stay_fully_detailed_and_summary_can_be_forced():
    assert len(mathsteps.linear_system(_big(4), [1] * 4).steps) > 10
    assert len(mathsteps.linear_system(_big(4), [1] * 4, detail="summary").steps) < 12


def test_inverse_and_lu_detail_levels():
    A = _big(15)
    assert len(mathsteps.inverse(A).steps) < 30 < len(mathsteps.inverse(A, detail="full").steps)
    assert len(mathsteps.lu(A).steps) < 30 < len(mathsteps.lu(A, detail="full").steps)
    assert mathsteps.inverse(A).verified is True and mathsteps.lu(A).verified is True


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_ode_steps_are_summarised_but_the_trajectory_is_complete(method):
    kwargs = {} if method == "rk45" else {"h": 0.001}
    # rk45 takes only ~110 steps here (under the 200-step auto threshold), so ask explicitly.
    detail = "summary" if method == "rk45" else None
    r = mathsteps.ivp("-0.5*y", y0=1, x_end=20, method=method, detail=detail, **kwargs)
    full = mathsteps.ivp("-0.5*y", y0=1, x_end=20, method=method, detail="full", **kwargs)
    assert len(r.steps) < 12 < len(full.steps)
    assert any("omitted" in s.description for s in r.steps)
    assert r.answer == full.answer
    assert np.array_equal(r.details["x"], full.details["x"]) and len(r.details["x"]) > 100


def test_small_ode_stays_fully_detailed():
    assert len(mathsteps.ivp("-0.5*y", y0=1, x_end=1, h=0.1).steps) == 12  # intro + 10 steps + final


def test_detail_none_and_invalid_detail():
    r = mathsteps.ivp("-0.5*y", y0=1, x_end=1, h=0.1, detail="none")
    assert len(r.steps) == 3  # intro, "omitted" note, final
    with pytest.raises(ValueError, match="detail must be"):
        mathsteps.linear_system([[1]], [1], detail="verbose")


def test_summary_mode_is_much_faster_for_big_systems():
    t0 = time.perf_counter()
    mathsteps.linear_system(_big(60), [1] * 60)
    assert time.perf_counter() - t0 < 10


# --- CLI ----------------------------------------------------------------------------------


def _cli(*args):
    import subprocess
    import sys

    return subprocess.run([sys.executable, "-m", "mathsteps.cli", *args],
                          capture_output=True, text=True, encoding="utf-8", check=False)


def test_cli_ode_system_backward_and_detail():
    r = _cli("ivp", "--f-expr", "v; -4*x", "--function", "x v", "--y0", "1 0", "--variable", "t",
             "--x-end", "3", "--h", "0.01")
    assert r.returncode == 0, r.stderr
    assert "Verification: PASS" in r.stdout and "0.96017" in r.stdout
    b = _cli("ivp", "--f-expr", "y", "--y0", "2.718281828459045", "--x0", "1", "--x-end", "0", "--h", "0.01")
    assert b.returncode == 0 and "Verification: PASS" in b.stdout
    d = _cli("linear-system", "--A", " ; ".join(" ".join(map(str, row)) for row in _big(12)),
             "--b", " ".join(["1"] * 12))
    assert d.stdout.count("Step ") < 20
    f = _cli("linear-system", "--A", "1 2; 3 5", "--b", "1 2", "--detail", "none")
    assert f.returncode == 0 and "Final answer: [-1.,  1.]" in f.stdout and "Exact: [-1, 1]" in f.stdout


def test_cli_prints_arrays_and_exact_forms():
    r = _cli("lu", "--A", "2 1; 4 -6")
    assert "P =" in r.stdout and "L =" in r.stdout and "U =" in r.stdout
    e = _cli("eigen", "--A", "2 0; 0 2")
    assert "eigenvalues = [2., 2.]" in e.stdout and "Verification: PASS" in e.stdout

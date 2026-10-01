"""Exact linear algebra at scale, honest ``verified`` values, convergence reporting."""
import math
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest
import sympy as sp

import mathsteps
from mathsteps import ConvergenceWarning
from mathsteps.verify import (
    verdict,
    verify_ivp_against_scipy,
    verify_problem,
)


def _diag_dominant(n):
    return [[((i * 7 + j * 3) % 11) + (20 if i == j else 0) for j in range(n)] for i in range(n)]


# --- performance / exactness ------------------------------------------------


def test_gaussian_elimination_scales_past_8x8():
    n = 12
    A = _diag_dominant(n)
    b = list(range(1, n + 1))
    t0 = time.perf_counter()
    r = mathsteps.linear_system(A, b)
    assert time.perf_counter() - t0 < 10  # used to take minutes at n=10
    assert r.verified is True
    assert np.allclose([float(v) for v in r.answer], np.linalg.solve(A, b))


def test_gaussian_elimination_answer_is_exact():
    r = mathsteps.linear_system([[3, 1], [1, 2]], [1, 1])
    assert r.exact == [sp.Rational(1, 5), sp.Rational(2, 5)]
    assert isinstance(r.answer, np.ndarray) and r.answer.tolist() == [0.2, 0.4]


def test_decimal_inputs_are_converted_exactly():
    r = mathsteps.linear_system([[0.1, 0.2], [0.3, 0.4]], [0.5, 1.1])
    assert r.exact == [1, 2]
    assert r.answer.tolist() == [1.0, 2.0]


def test_cofactor_determinant_10x10_matches_numpy():
    A = [[(i * j + 1) if i != j else 5 for j in range(10)] for i in range(10)]
    r = mathsteps.determinant(A)
    assert r.verified is True
    assert float(r.answer) == pytest.approx(np.linalg.det(np.array(A, float)), rel=1e-9)


def test_cofactor_determinant_known_value_and_non_square():
    assert mathsteps.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).answer == -306
    with pytest.raises(ValueError, match="square"):
        mathsteps.determinant([[1, 2, 3], [4, 5, 6]])


def test_lu_and_inverse_large():
    A = _diag_dominant(15)
    assert mathsteps.inverse(A).verified is True
    r = mathsteps.lu(A)
    assert r.verified is True
    P, L, U = r.answer
    assert np.allclose(
        np.array(P.tolist(), float) @ np.array(A, float),
        np.array(L.tolist(), float) @ np.array(U.tolist(), float),
    )


def test_finite_difference_bvp_fine_grid():
    r = mathsteps.bvp_finite_difference(q_expr="0", r_expr="2", n=40)
    # -y'' = 2, y(0)=y(1)=0  ->  y = x(1-x)
    mid = len(r.answer) // 2
    assert r.answer[mid] == pytest.approx(0.25, abs=1e-3)


# --- honest `verified` ------------------------------------------------------


def test_verified_is_none_when_verification_disabled_or_unavailable():
    assert mathsteps.linear_system([[1, 0], [0, 1]], [1, 2], verify=False).verified is None
    assert verify_problem({"type": "some_custom_solver"}, 1.0) is None  # no reference: not a pass
    assert "NOT CHECKED" in str(mathsteps.integrate("x", a=0, b=1, verify=False))


def test_verdict_labels():
    assert [verdict(True), verdict(False), verdict(None)] == ["PASS", "FAIL", "NOT CHECKED"]


def test_wrong_answers_fail_verification():
    assert verify_problem({"type": "linear_system", "A": [[1, 0], [0, 1]], "b": [1, 2]}, [1, 3]) is False
    assert verify_problem({"type": "determinant", "A": [[1, 2], [3, 4]]}, 5) is False
    assert verify_problem({"type": "root_finding", "method": "newton_raphson",
                           "function": "x**2 - 2", "variable": "x", "x0": 1.0}, 1.5) is False


def test_verification_that_cannot_run_is_none_not_a_pass():
    assert verify_problem({"type": "linear_system"}, [1, 2]) is None  # malformed problem


def test_complex_and_repeated_eigenvalues_verify():
    assert mathsteps.eigenvalues([[0, -1], [1, 0]]).verified is True
    assert mathsteps.eigenvalues([[2, 0], [0, 2]]).verified is True
    assert mathsteps.eigenvalues([[1, 1], [1, 0]]).verified is True


def test_fixed_point_is_verified_against_g_x_minus_x():
    r = mathsteps.root("cos(x)", method="fixed_point", x0=0.5, max_iter=200)
    assert r.converged is True
    assert r.verified is True
    assert verify_problem(
        {"type": "root_finding", "method": "fixed_point", "function": "cos(x)",
         "variable": "x"}, 0.9,
    ) is False


@pytest.mark.parametrize("method", ["euler", "heun", "midpoint", "rk4", "rk45"])
def test_ivp_methods_verify_with_truncation_aware_tolerance(method):
    kwargs = {} if method == "rk45" else {"h": 0.05}
    r = mathsteps.ivp("-2*x*y", y0=1, x_end=2, method=method, **kwargs)
    assert r.verified is True


def test_ivp_verifier_rejects_wrong_answers():
    # Exact answer is exp(-4) = 0.0183; an RK4 result of 0.05 is not a truncation error.
    assert verify_ivp_against_scipy(
        "-2*x*y", "x", "y", 1.0, 0.0, 2.0, 0.05, method="rk4", h=0.1,
    ) is False


def test_ivp_step_size_not_dividing_interval_still_lands_on_x_end():
    # h=0.3 does not divide [0, 1]; the solver shrinks it to 0.25 instead of stopping at x=0.9.
    r = mathsteps.ivp("y", y0=1, x_end=1, h=0.3, method="rk4")
    assert r.answer == pytest.approx(math.e, rel=1e-3)
    assert r.verified is True
    assert "h = 0.25" in r.steps[0].description


@pytest.mark.parametrize("method", ["euler", "heun", "midpoint", "rk4"])
def test_ivp_rejects_non_positive_h_and_empty_interval(method):
    with pytest.raises(ValueError, match="positive"):
        mathsteps.ivp("y", h=-0.1, x_end=1, method=method)
    with pytest.raises(ValueError, match="positive"):
        mathsteps.ivp("y", h=0, x_end=1, method=method)
    with pytest.raises(ValueError, match="must differ"):
        mathsteps.ivp("y", x0=1, x_end=1, h=0.1, method=method)
    with pytest.raises(ValueError, match="must differ"):
        mathsteps.ivp("y", x0=1, x_end=1, method="rk45")


def test_unstable_ivp_result_is_not_reported_as_verified():
    # Euler on stiff y' = -1000 y with h = 0.01 explodes (|1 - 10| > 1 each step).
    r = mathsteps.ivp("-1000*y", y0=1, x_end=1, h=0.01, method="euler")
    assert r.verified is False


def test_lu_cramer_interpolation_now_have_real_checks():
    assert mathsteps.cramers_rule([[2, 1], [1, 3]], [3, 5]).verified is True
    assert mathsteps.lagrange([(0, 1), (1, 2), (2, 5)]).verified is True
    assert mathsteps.newton_divided_differences([(0, 1), (1, 2), (2, 5)]).verified is True
    x = sp.Symbol("x")
    assert verify_problem(
        {"type": "interpolation", "points": [[0, 1], [1, 2]]}, x + 2,
    ) is False


# --- eigenvalues on realistic (irrational-eigenvalue) matrices ---------------


def _check_eigenpairs(A, r):
    values, vectors = r.answer
    A_np = np.array(A, dtype=float)
    assert vectors.shape == (len(A), len(values))
    assert len(values) == len(A)  # these test matrices are diagonalizable
    for j, lam in enumerate(values):
        v = vectors[:, j]
        assert abs(np.linalg.norm(v) - 1.0) < 1e-12, "eigenvectors are unit-norm"
        assert np.linalg.norm(A_np @ v - lam * v) < 1e-8 * max(1, abs(lam))


@pytest.mark.parametrize(
    "A",
    [
        np.round(np.cov(np.random.default_rng(1).normal(size=(3, 50))), 2).tolist(),
        np.round(np.cov(np.random.default_rng(1).normal(size=(4, 50))), 3).tolist(),
        np.random.default_rng(1).integers(-5, 6, (5, 5)).tolist(),  # non-symmetric: complex pairs
        (lambda S: (S + S.T).tolist())(np.random.default_rng(1).integers(-3, 4, (6, 6))),
    ],
    ids=["cov3-decimals", "cov4-decimals", "random5-nonsymmetric", "random6-symmetric"],
)
def test_eigenvalues_of_realistic_matrices_are_fast_and_correct(A):
    t0 = time.perf_counter()
    r = mathsteps.eigenvalues(A)
    assert time.perf_counter() - t0 < 10  # used to hang for minutes (Cardano radicals)
    assert r.verified is True
    _check_eigenpairs(A, r)


def test_eigenvalues_decimal_input_gives_nonzero_eigenvectors():
    # Markov chain with decimal transition probabilities: stationary vector is [5, 1] / 6.
    r = mathsteps.eigenvalues([[0.9, 0.5], [0.1, 0.5]])
    values, vectors = r.answer
    v = vectors[:, int(np.argmin(np.abs(values - 1.0)))]
    assert v / v.sum() == pytest.approx([5 / 6, 1 / 6])
    assert r.verified is True


def test_eigenvector_verification_rejects_zero_vectors():
    from mathsteps.verify import verify_eigenvectors

    assert verify_eigenvectors([[2, 0], [0, 3]], [2], {2: [sp.Matrix([0, 0])]}) is False
    assert verify_eigenvectors([[2, 0], [0, 3]], [2], {2: [sp.Matrix([1, 0])]}) is True
    # a missing basis vector (eigenspace of the identity is 2-dimensional) is caught too
    assert verify_eigenvectors([[2, 0], [0, 2]], [2], {2: [sp.Matrix([1, 0])]}) is False


# --- convergence reporting --------------------------------------------------


def test_converged_solvers_report_true_and_stay_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert mathsteps.root("cos(x) - x", method="newton", x0=0.0).converged is True
        assert mathsteps.root("x**3 - x - 2", method="bisection", a=1, b=2).converged is True
        assert mathsteps.root("x**3 - x - 2", method="secant", x0=1, x1=2).converged is True
        assert mathsteps.bvp_shooting("-y", y_right=0, x_end=np.pi, s0=0.5, s1=1.5).converged is True


def test_slow_fixed_point_with_default_max_iter_is_flagged_not_silent():
    # |g'| ~ 0.67 near the fixed point, so 50 iterations cannot reach tol=1e-10.
    with pytest.warns(ConvergenceWarning):
        r = mathsteps.root("cos(x)", method="fixed_point", x0=0.5)
    assert r.converged is False
    assert r.answer == pytest.approx(0.7390851332, abs=1e-6)


def test_non_iterative_solvers_have_converged_none():
    assert mathsteps.linear_system([[1, 0], [0, 1]], [1, 2]).converged is None
    assert mathsteps.ivp("y", h=0.1, method="rk4").converged is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: mathsteps.root("cos(x) - x", method="newton", x0=0.0, max_iter=2),
        lambda: mathsteps.root("x**3 - x - 2", method="bisection", a=1, b=2, max_iter=3),
        lambda: mathsteps.root("x**3 - x - 2", method="secant", x0=1, x1=2, max_iter=2),
        lambda: mathsteps.root("cos(x)", method="fixed_point", x0=0.5, max_iter=3),
        lambda: mathsteps.bvp_shooting("sin(x) - y", x_end=1, tol=1e-30, max_iter=1),
    ],
)
def test_hitting_max_iter_warns_and_sets_converged_false(call):
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        r = call()
    assert r.converged is False
    assert "NOT converge" in " ".join(s.description for s in r.steps) or "without converging" in r.steps[-1].description
    assert "Converged: NO" in str(r)


# --- CLI --------------------------------------------------------------------


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mathsteps.cli", *args],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


def test_cli_verify_exit_codes(monkeypatch):
    # Numerical integration now has an independent check, so it passes ...
    r = _cli("verify", "examples/int_simpson_sin.json")
    assert r.returncode == 0 and "PASS" in r.stdout

    # ... and a result that cannot be checked exits 2 (simulated in-process).
    from typer.testing import CliRunner

    import mathsteps.cli as cli

    unchecked = mathsteps.Result(solver="x", steps=[], answer=0.0, verified=None)
    monkeypatch.setattr(cli, "_solve", lambda problem, verify: unchecked)
    out = CliRunner().invoke(cli.app, ["verify", "examples/int_simpson_sin.json"])
    assert out.exit_code == 2 and "NOT CHECKED" in out.output


def test_cli_solve_reports_non_convergence_on_stderr():
    r = _cli("solve", "cos(x) - x", "--method", "newton_raphson", "--x0", "0",
             "--max-iter", "2", "--no-verify")
    assert r.returncode == 0
    assert "did not converge" in r.stderr


def test_cli_accepts_method_alias_newton():
    r = _cli("root", "--function", "cos(x) - x", "--method", "newton", "--x0", "0")
    assert r.returncode == 0, r.stderr
    assert "Verification: PASS" in r.stdout


def test_cli_finite_difference_bvp_is_reachable():
    # -y'' = 50, y(0) = y(1) = 0  ->  y = 25 x (1 - x); midpoint value 6.25
    r = _cli("bvp", "--method", "finite_difference", "--q-expr", "0", "--r-expr", "50",
             "--alpha", "0", "--beta", "0", "--n", "19")
    assert r.returncode == 0, r.stderr
    import ast

    text = r.stdout.split("Final answer:", 1)[1].split("Verification:", 1)[0]
    ys = ast.literal_eval(text.strip())
    xs = [i / 20 for i in range(21)]
    assert max(abs(y - 25 * x * (1 - x)) for x, y in zip(xs, ys)) < 1e-9


@pytest.mark.parametrize(
    "args, needle",
    [
        (["linear-system", "--A", "1 2; 2 4", "--b", "1 2"], "singular"),
        (["root", "--function", "x**2+1", "--method", "bisection", "--a", "0", "--c", "1"], "opposite signs"),
        (["root", "--function", "k*x - 3", "--x0", "1"], "Unknown symbol"),
        (["ivp", "--f-expr", "y", "--x0", "1", "--x-end", "1", "--h", "0.1"], "must differ"),
    ],
)
def test_cli_user_errors_are_one_line_not_tracebacks(args, needle):
    r = _cli(*args)
    assert r.returncode == 1
    assert "Traceback" not in r.stdout + r.stderr
    assert r.stderr.startswith("Error:") and needle in r.stderr


# --- interactive `ask` wizard ---------------------------------------------------


def _ask(first_lines: str):
    from typer.testing import CliRunner

    from mathsteps.cli import app

    return CliRunner().invoke(app, ["ask"], input=first_lines + "\n" * 40)


@pytest.mark.parametrize("choice", [str(n) for n in range(1, 11)])
def test_ask_wizard_every_menu_entry_solves_with_defaults(choice):
    r = _ask(choice)
    assert r.exit_code == 0, r.output
    assert "Final answer" in r.output


def test_ask_wizard_accepts_domain_name_and_method_specific_prompts():
    assert "Final answer" in _ask("determinant").output
    bis = _ask("2\nx**2-2\nbisection\n1\n2")
    assert bis.exit_code == 0 and "Verification: PASS" in bis.output
    sec = _ask("2\nx**2-2\nsecant\n1\n2")
    assert sec.exit_code == 0 and "Verification: PASS" in sec.output


def test_ask_wizard_bad_input_is_a_message_not_a_traceback():
    bad_menu = _ask("banana")
    assert bad_menu.exit_code != 0 and "Choice must be" in bad_menu.output
    singular = _ask("1\n1 2; 2 4\n1 2")
    assert singular.exit_code == 1
    assert "singular" in singular.output and "Traceback" not in singular.output


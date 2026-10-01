"""Smoke tests for the new CLI subcommands."""
import json
import subprocess
import sys

import mathsteps.domains.linalg.gaussian_elimination  # noqa: F401
import mathsteps.domains.numerical.bisection  # noqa: F401
import mathsteps.domains.numerical.newton_raphson  # noqa: F401
import mathsteps.domains.numerical.secant  # noqa: F401
import mathsteps.domains.numerical.numerical_diff  # noqa: F401
import mathsteps.domains.numerical.numerical_integration  # noqa: F401
import mathsteps.domains.numerical.lagrange  # noqa: F401
import mathsteps.domains.ode_ivp.rk4  # noqa: F401
import mathsteps.domains.ode_bvp.shooting  # noqa: F401
import mathsteps.domains.linalg.cofactor_determinant  # noqa: F401
import mathsteps.domains.linalg.gauss_jordan_inverse  # noqa: F401
import mathsteps.domains.linalg.lu_decomposition  # noqa: F401
import mathsteps.domains.linalg.eigenvalues  # noqa: F401
import mathsteps.domains.linalg.cramers_rule  # noqa: F401


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mathsteps.cli", *args],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )


def test_linear_system_subcommand():
    r = _run(["linear-system", "--A", "1 2; 3 4", "--b", "5 11", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: gaussian_elimination" in r.stdout
    assert "x = [1, 2]" in r.stdout


def test_root_subcommand_bisection():
    r = _run([
        "root", "--function", "x**3 - x - 2",
        "--method", "bisection", "--a", "1", "--c", "2", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: bisection" in r.stdout


def test_root_subcommand_newton():
    r = _run([
        "root", "--function", "cos(x) - x",
        "--method", "newton_raphson", "--x0", "0", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: newton_raphson" in r.stdout


def test_root_subcommand_secant():
    r = _run([
        "root", "--function", "x**3 - x - 2",
        "--method", "secant", "--x0", "1", "--x1", "2", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: secant" in r.stdout


def test_ivp_subcommand_rk4():
    r = _run([
        "ivp", "--f-expr", "-2*x*y",
        "--y0", "1", "--x-end", "2", "--h", "0.1", "--method", "rk4",
        "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: rk4" in r.stdout


def test_bvp_subcommand_shooting():
    r = _run([
        "bvp", "--f-expr", "sin(x) - y",
        "--a", "0", "--c", "0", "--x-end", "1", "--s0", "0", "--s1", "1",
        "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: shooting" in r.stdout


def test_determinant_subcommand():
    r = _run(["determinant", "--A", "6 1 1; 4 -2 5; 2 8 7", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: cofactor_determinant" in r.stdout
    assert "-306" in r.stdout


def test_inverse_subcommand():
    r = _run(["inverse", "--A", "1 2; 3 4", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: gauss_jordan_inverse" in r.stdout


def test_lu_subcommand():
    r = _run(["lu", "--A", "2 1 1; 4 -6 0; -2 7 2", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: lu_decomposition" in r.stdout


def test_eigen_subcommand():
    r = _run(["eigen", "--A", "2 0 0; 0 3 4; 0 4 9", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: eigenvalues" in r.stdout


def test_cramer_subcommand():
    r = _run([
        "cramer", "--A", "2 1 -1; -3 -1 2; -2 1 2",
        "--b", "8 -11 -3", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: cramers_rule" in r.stdout


def test_interp_subcommand():
    r = _run([
        "interp", "--points", "(0,1); (1,2); (2,5)",
        "--method", "lagrange", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: lagrange" in r.stdout


def test_integrate_subcommand():
    r = _run([
        "integrate", "--function", "sin(x)",
        "--a", "0", "--c", "3.141592653589793",
        "--n", "100", "--method", "simpson", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: numerical_integration" in r.stdout


def test_diff_subcommand():
    r = _run([
        "diff", "--function", "sin(x)",
        "--x", "1", "--h", "0.001", "--method", "central", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: numerical_diff" in r.stdout


def test_solve_free_form_root():
    r = _run(["solve", "x**3 - x - 2", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: newton_raphson" in r.stdout


def test_solve_free_form_with_equals():
    r = _run(["solve", "x**2 - 4 = 0", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: newton_raphson" in r.stdout


def test_solve_free_form_ivp():
    r = _run([
        "solve", "y' = -2*x*y", "--y0", "1", "--x-end", "2",
        "--h", "0.1", "--method", "rk4", "--no-verify",
    ])
    assert r.returncode == 0
    assert "Solver: rk4" in r.stdout


def test_solve_json_file_still_works():
    r = _run(["solve", "examples/linear_system_2x2.json", "--no-verify"])
    assert r.returncode == 0
    assert "Solver: gaussian_elimination" in r.stdout


def test_list_solvers():
    r = _run(["list-solvers"])
    assert r.returncode == 0
    assert "gaussian_elimination" in r.stdout
    assert "rk4" in r.stdout


def test_pretty_flag():
    r = _run(["linear-system", "--A", "1 2; 3 4", "--b", "5 11", "--no-verify", "--pretty"])
    assert r.returncode == 0
    assert "MathSteps" in r.stdout


def test_verify_command_exit_code():
    r = _run(["verify", "examples/linear_system_2x2.json"])
    assert r.returncode == 0
    assert "PASS" in r.stdout

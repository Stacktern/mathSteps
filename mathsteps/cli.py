"""MathSteps CLI entry point.

Three ways to use it:

1. Direct subcommands (easiest for new users)::

       mathsteps linear-system --A "1 2; 3 4" --b "5 11"
       mathsteps root "x**3 - x - 2" --method newton --x0 1.5
       mathsteps ivp "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
       mathsteps diff "sin(x)" --x 1 --h 0.001
       mathsteps integrate "sin(x)" --a 0 --b 3.14 --n 100 --method simpson

2. Free-form equation (auto-detected)::

       mathsteps solve "x**3 - x - 2"
       mathsteps solve "x**2 - 4 = 0"
       mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2

3. JSON file (power-user / scripted)::

       mathsteps solve path/to/problem.json

4. Interactive wizard::

       mathsteps ask
"""
from __future__ import annotations

import json
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

# Force UTF-8 stdout so we can print λ, ≈, Δ, ∫ on Windows consoles that
# default to cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import typer

import mathsteps.domains.linalg.gaussian_elimination  # noqa: F401  (registers solver)
import mathsteps.domains.linalg.gauss_jordan_inverse  # noqa: F401
import mathsteps.domains.linalg.cofactor_determinant  # noqa: F401
import mathsteps.domains.linalg.lu_decomposition  # noqa: F401
import mathsteps.domains.linalg.eigenvalues  # noqa: F401
import mathsteps.domains.linalg.cramers_rule  # noqa: F401
import mathsteps.domains.numerical.bisection  # noqa: F401  (registers solver)
import mathsteps.domains.numerical.newton_raphson  # noqa: F401
import mathsteps.domains.numerical.secant  # noqa: F401
import mathsteps.domains.numerical.fixed_point  # noqa: F401
import mathsteps.domains.numerical.numerical_diff  # noqa: F401
import mathsteps.domains.numerical.numerical_integration  # noqa: F401
import mathsteps.domains.numerical.lagrange  # noqa: F401
import mathsteps.domains.numerical.newton_divided_differences  # noqa: F401
import mathsteps.domains.ode_ivp.euler  # noqa: F401
import mathsteps.domains.ode_ivp.heun  # noqa: F401
import mathsteps.domains.ode_ivp.rk4  # noqa: F401
import mathsteps.domains.ode_ivp.midpoint  # noqa: F401
import mathsteps.domains.ode_ivp.rk45  # noqa: F401
import mathsteps.domains.ode_bvp.shooting  # noqa: F401
import mathsteps.domains.ode_bvp.finite_difference  # noqa: F401
from mathsteps.core.inputs import build_problem_from_inputs, detect_and_build, normalize_method
from mathsteps.api import Result, solve_problem
from mathsteps.core.answers import InterpolatingPolynomial
from mathsteps.core.registry import all_solvers
from mathsteps.core.render import render_panel, render_steps
from mathsteps.core.convergence import ConvergenceWarning
from mathsteps.verify import verdict

app = typer.Typer(
    help="MathSteps: step-by-step math solver.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(show: bool) -> None:
    if show:
        from mathsteps import __version__

        typer.echo(f"mathsteps {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """MathSteps: step-by-step math solver."""


def _load_problem(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"Problem file not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in {path}: {exc}") from None


@contextmanager
def _user_errors():
    """Turn bad-input failures into a one-line message + exit code 1 (no traceback)."""
    try:
        yield
    except typer.Exit:
        raise
    except (ValueError, KeyError, RuntimeError, ZeroDivisionError) as exc:
        message = f"missing required input {exc}" if isinstance(exc, KeyError) else str(exc)
        typer.echo(f"Error: {message}", err=True)
        raise typer.Exit(code=1) from None


def _fmt_array(value) -> str:
    import numpy as np

    return np.array2string(
        np.asarray(value), precision=10, suppress_small=True, separator=", ", threshold=60
    )


def _format_answer(result: Result) -> str:
    """Plain-text rendering of ``result.answer`` (arrays summarised when long)."""
    import numpy as np

    answer = result.answer
    kind = result.problem.get("type")
    if kind == "lu_decomposition":
        return "\n".join(f"{name} =\n{_fmt_array(m)}" for name, m in zip("PLU", answer))
    if kind == "eigenvalues":
        values, vectors = answer
        return f"eigenvalues = {_fmt_array(values)}\neigenvectors (columns) =\n{_fmt_array(vectors)}"
    if isinstance(answer, np.ndarray):
        return _fmt_array(answer)
    return str(answer)


def _print_result(result: Result, pretty: bool) -> None:
    final = _format_answer(result)
    exact = None
    if result.exact is not None and not isinstance(result.answer, InterpolatingPolynomial):
        text = str(result.exact)
        exact = text if len(text) <= 400 else text[:400] + " ..."
    if pretty:
        from rich.console import Console

        console = Console()
        render_panel("MathSteps", f"Solver: {result.solver}", console=console)
        render_steps(result.steps, console=console)
        render_panel("Final answer", final + (f"\n\nExact: {exact}" if exact else ""), console=console)
    else:
        typer.echo(f"Solver: {result.solver}\n")
        for i, step in enumerate(result.steps, 1):
            typer.echo(f"Step {i}: {step.description}")
            if step.before:
                typer.echo(f"  before: {step.before}")
            if step.after and step.after != step.before:
                typer.echo(f"  after:  {step.after}")
            typer.echo("")
        typer.echo(f"Final answer: {final}")
        if exact:
            typer.echo(f"Exact: {exact}")


def _solve(problem: dict, *, verify: bool) -> Result:
    # The CLI reports non-convergence itself, so silence the library warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return solve_problem(problem, verify=verify)


def _solve_and_print(problem: dict, *, no_verify: bool, pretty: bool) -> None:
    result = _solve(problem, verify=not no_verify)
    _print_result(result, pretty)
    if result.converged is False:
        typer.echo(
            "Warning: did not converge within max_iter; the answer above is the last iterate.",
            err=True,
        )
    if not no_verify:
        typer.echo(f"Verification: {verdict(result.verified)}")


# Common kwargs reused across the per-domain subcommands.
def _domain_kwargs(
    method: Optional[str] = typer.Option(None, "--method", help="Method name (e.g. newton, rk4, simpson)."),
    A: Optional[str] = typer.Option(None, "--A", help='Matrix, e.g. "1 2; 3 4".'),
    b: Optional[str] = typer.Option(None, "--b", help='Vector, e.g. "5 11".'),
    function: Optional[str] = typer.Option(None, "--function", help="SymPy expression."),
    variable: Optional[str] = typer.Option(None, "--variable", help="Variable name (default x)."),
    a: Optional[str] = typer.Option(None, "--a", help="Left bracket / boundary / interval start."),
    c: Optional[str] = typer.Option(None, "--c", help="Right bracket / boundary / interval end (alias for --b where ambiguous)."),
    x0: Optional[str] = typer.Option(None, "--x0", help="Initial guess / initial x / boundary alpha."),
    x1: Optional[str] = typer.Option(None, "--x1", help="Second guess / boundary beta."),
    y0: Optional[str] = typer.Option(None, "--y0", help="Initial y for IVPs."),
    x_end: Optional[str] = typer.Option(None, "--x-end", help="End of integration interval."),
    h: Optional[str] = typer.Option(None, "--h", help="Step size."),
    tol: Optional[str] = typer.Option(None, "--tol", help="Convergence tolerance."),
    max_iter: Optional[str] = typer.Option(None, "--max-iter", help="Iteration cap."),
    s0: Optional[str] = typer.Option(None, "--s0", help="Initial shooting slope guess 1."),
    s1: Optional[str] = typer.Option(None, "--s1", help="Initial shooting slope guess 2."),
    points: Optional[str] = typer.Option(None, "--points", help="Points '(x0,y0); (x1,y1); ...'."),
    p_expr: Optional[str] = typer.Option(None, "--p-expr", help="BVP: p(x) coefficient of y'."),
    q_expr: Optional[str] = typer.Option(None, "--q-expr", help="BVP: q(x) coefficient of y."),
    r_expr: Optional[str] = typer.Option(None, "--r-expr", help="BVP: r(x) RHS."),
    alpha: Optional[str] = typer.Option(None, "--alpha", help="BVP: y(a) value."),
    beta: Optional[str] = typer.Option(None, "--beta", help="BVP: y(b) value."),
    n: Optional[str] = typer.Option(None, "--n", help="Number of sub-intervals / interior points."),
    f_expr: Optional[str] = typer.Option(None, "--f-expr", help="IVP/BVP right-hand-side expression."),
    target_function: Optional[str] = typer.Option(None, "--target-function", help="Underlying f(x) for fixed-point verification."),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> dict[str, Any]:
    """Build a problem dict from the shared kwargs."""
    locals_dict = locals()
    locals_dict.pop("no_verify", None)
    locals_dict.pop("pretty", None)
    return locals_dict


def _run_from_kwargs(
    domain: str, kwargs: dict[str, Any], *, no_verify: bool, pretty: bool, detail: str | None = None
) -> None:
    with _user_errors():
        problem = build_problem_from_inputs(domain, **kwargs)
        if detail is not None:
            problem["detail"] = detail
        _solve_and_print(problem, no_verify=no_verify, pretty=pretty)


# --- Per-domain subcommands -------------------------------------------------


@app.command("linear-system")
def linear_system_cmd(
    A: str = typer.Option(..., "--A", help='Augmented-coefficient matrix (without b).'),
    b: str = typer.Option(..., "--b", help='Right-hand side vector, e.g. "5 11".'),
    detail: Optional[str] = typer.Option(None, "--detail", help="Steps to record: full|summary|none (default: summary for large problems)."),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Solve a linear system Ax = b via Gaussian elimination.

    Example::

        mathsteps linear-system --A "1 2; 3 4" --b "5 11"
    """
    _run_from_kwargs(
        "linear_system",
        {"A": A, "b": b},
        no_verify=no_verify,
        pretty=pretty,
    )


@app.command("root")
def root_cmd(
    function: str = typer.Option(..., "--function", help="f(x) expression."),
    variable: str = typer.Option("x", "--variable"),
    method: str = typer.Option("newton_raphson", "--method", help="bisection|newton_raphson|secant|fixed_point"),
    a: Optional[str] = typer.Option(None, "--a", help="Bisection: left bracket."),
    c: Optional[str] = typer.Option(None, "--c", help="Bisection: right bracket."),
    x0: Optional[str] = typer.Option(None, "--x0", help="Newton / secant / fixed-point: initial guess."),
    x1: Optional[str] = typer.Option(None, "--x1", help="Secant: second guess."),
    tol: Optional[str] = typer.Option(None, "--tol"),
    max_iter: Optional[str] = typer.Option(None, "--max-iter"),
    target_function: Optional[str] = typer.Option(None, "--target-function"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Find a root of f(x) = 0.

    Examples::

        mathsteps root --function "x**3 - x - 2" --method bisection --a 1 --c 2
        mathsteps root --function "cos(x) - x" --method newton --x0 0
        mathsteps root --function "x**3 - x - 2" --method secant --x0 1 --x1 2
    """
    _run_from_kwargs(
        "root",
        {
            "function": function, "variable": variable, "method": method,
            "a": a, "c": c, "x0": x0, "x1": x1, "tol": tol, "max_iter": max_iter,
            "target_function": target_function,
        },
        no_verify=no_verify, pretty=pretty,
    )


@app.command("ivp")
def ivp_cmd(
    f_expr: str = typer.Option(..., "--f-expr", help='dy/dx = f(x, y), e.g. "-2*x*y" or "y\' = -2*x*y".'),
    y0: str = typer.Option("1.0", "--y0"),
    x0: str = typer.Option("0.0", "--x0"),
    x_end: str = typer.Option(..., "--x-end"),
    h: Optional[str] = typer.Option(None, "--h", help="Step size (not needed for rk45)."),
    method: str = typer.Option("rk4", "--method", help="euler|heun|midpoint|rk4|rk45"),
    variable: str = typer.Option("x", "--variable"),
    function: Optional[str] = typer.Option(None, "--function", help='Unknown name(s), e.g. "T" or, for a system, "x v".'),
    detail: Optional[str] = typer.Option(None, "--detail", help="Steps to record: full|summary|none (default: summary for large problems)."),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Solve an IVP: dy/dx = f(x, y), y(x0) = y0 (one equation or a system; x_end may be < x0).

    Example::

        mathsteps ivp --f-expr "-2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
    """
    _run_from_kwargs(
        "ivp",
        {"f_expr": f_expr, "y0": y0, "x0": x0, "x_end": x_end,
         "h": h, "method": method, "variable": variable, "function": function},
        no_verify=no_verify, pretty=pretty, detail=detail,
    )


@app.command("bvp")
def bvp_cmd(
    f_expr: Optional[str] = typer.Option(None, "--f-expr", help="Shooting: y'' = f(x, y, y')."),
    p_expr: Optional[str] = typer.Option(None, "--p-expr", help="Finite difference: p(x), coefficient of y'."),
    q_expr: Optional[str] = typer.Option(None, "--q-expr", help="Finite difference: q(x), coefficient of y."),
    r_expr: Optional[str] = typer.Option(None, "--r-expr", help="Finite difference: r(x), right-hand side."),
    alpha: Optional[str] = typer.Option(None, "--alpha", help="Finite difference: y(x0)."),
    beta: Optional[str] = typer.Option(None, "--beta", help="Finite difference: y(x_end)."),
    n: Optional[str] = typer.Option(None, "--n", help="Finite difference: number of interior points."),
    a: str = typer.Option("0.0", "--a", help="y_left."),
    c: str = typer.Option("0.0", "--c", help="y_right."),
    x0: str = typer.Option("0.0", "--x0", help="Left boundary x."),
    x_end: str = typer.Option("1.0", "--x-end", help="Right boundary x."),
    h: str = typer.Option("0.01", "--h"),
    s0: str = typer.Option("0.0", "--s0"),
    s1: str = typer.Option("1.0", "--s1"),
    tol: str = typer.Option("1e-6", "--tol"),
    method: str = typer.Option("shooting", "--method", help="shooting|finite_difference"),
    variable: str = typer.Option("x", "--variable"),
    max_iter: str = typer.Option("30", "--max-iter"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Solve a two-point BVP.

    Examples::

        # shooting: y'' = f(x, y, y'), y(x0) = --a, y(x_end) = --c
        mathsteps bvp --f-expr "sin(x) - y" --a 0 --c 0 --x0 0 --x-end 1 --s0 0 --s1 1

        # finite difference: -y'' + p y' + q y = r on [x0, x_end]
        mathsteps bvp --method finite_difference --q-expr 0 --r-expr 50 --alpha 0 --beta 0 --n 19
    """
    _run_from_kwargs(
        "bvp",
        {"f_expr": f_expr, "a": a, "c": c, "x0": x0, "x_end": x_end,
         "h": h, "s0": s0, "s1": s1, "tol": tol, "method": method,
         "variable": variable, "max_iter": max_iter,
         "p_expr": p_expr, "q_expr": q_expr, "r_expr": r_expr,
         "alpha": alpha, "beta": beta, "n": n},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("determinant")
def determinant_cmd(
    A: str = typer.Option(..., "--A", help='Square matrix, e.g. "6 1 1; 4 -2 5; 2 8 7".'),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Compute the determinant of a square matrix."""
    _run_from_kwargs(
        "determinant",
        {"A": A, "method": "cofactor"},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("inverse")
def inverse_cmd(
    A: str = typer.Option(..., "--A", help='Square matrix.'),
    detail: Optional[str] = typer.Option(None, "--detail", help="Steps to record: full|summary|none (default: summary for large problems)."),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Compute the inverse of a square matrix via Gauss-Jordan elimination."""
    _run_from_kwargs(
        "matrix_inverse",
        {"A": A},
        no_verify=no_verify, pretty=pretty, detail=detail,
    )


@app.command("lu")
def lu_cmd(
    A: str = typer.Option(..., "--A", help='Square matrix.'),
    detail: Optional[str] = typer.Option(None, "--detail", help="Steps to record: full|summary|none (default: summary for large problems)."),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """LU decomposition (with partial pivoting)."""
    _run_from_kwargs(
        "lu",
        {"A": A},
        no_verify=no_verify, pretty=pretty, detail=detail,
    )


@app.command("eigen")
def eigen_cmd(
    A: str = typer.Option(..., "--A", help='Square matrix.'),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Eigenvalues / eigenvectors via the characteristic polynomial."""
    _run_from_kwargs(
        "eigen",
        {"A": A},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("cramer")
def cramer_cmd(
    A: str = typer.Option(..., "--A", help='Square coefficient matrix.'),
    b: str = typer.Option(..., "--b", help='Right-hand side vector.'),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Solve Ax = b via Cramer's rule."""
    _run_from_kwargs(
        "cramer",
        {"A": A, "b": b},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("interp")
def interp_cmd(
    points: str = typer.Option(..., "--points", help='Points "(x0,y0); (x1,y1); ...".'),
    method: str = typer.Option("lagrange", "--method", help="lagrange|newton_divided_differences"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Polynomial interpolation through given points."""
    _run_from_kwargs(
        "interp",
        {"points": points, "method": method},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("integrate")
def integrate_cmd(
    function: str = typer.Option(..., "--function"),
    variable: str = typer.Option("x", "--variable"),
    a: str = typer.Option(..., "--a"),
    c: str = typer.Option(..., "--c", help="Right interval end."),
    n: str = typer.Option("100", "--n"),
    method: str = typer.Option("trapezoidal", "--method", help="trapezoidal|simpson"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Numerical integration of f(x) over [a, c]."""
    _run_from_kwargs(
        "integrate",
        {"function": function, "variable": variable, "a": a, "c": c,
         "n": n, "method": method},
        no_verify=no_verify, pretty=pretty,
    )


@app.command("diff")
def diff_cmd(
    function: str = typer.Option(..., "--function"),
    x: str = typer.Option("1.0", "--x"),
    h: str = typer.Option("0.001", "--h"),
    variable: str = typer.Option("x", "--variable"),
    method: str = typer.Option("central", "--method", help="forward|backward|central"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    """Numerical derivative of f(x) at x via finite differences."""
    _run_from_kwargs(
        "diff",
        {"function": function, "x": x, "h": h, "variable": variable, "method": method},
        no_verify=no_verify, pretty=pretty,
    )


# --- Generic solve (free-form or JSON file) ---------------------------------


@app.command()
def solve(
    expression: Optional[str] = typer.Argument(
        None, help="Free-form equation (auto-detected) OR a path to a JSON file."
    ),
    pretty: bool = typer.Option(False, "--pretty"),
    no_verify: bool = typer.Option(False, "--no-verify"),
    # Common flags forwarded to detected IVP/root/...
    method: Optional[str] = typer.Option(None, "--method"),
    y0: Optional[str] = typer.Option(None, "--y0"),
    x0: Optional[str] = typer.Option(None, "--x0"),
    x_end: Optional[str] = typer.Option(None, "--x-end"),
    h: Optional[str] = typer.Option(None, "--h"),
    a: Optional[str] = typer.Option(None, "--a"),
    c: Optional[str] = typer.Option(None, "--c"),
    tol: Optional[str] = typer.Option(None, "--tol"),
    max_iter: Optional[str] = typer.Option(None, "--max-iter"),
) -> None:
    """Solve a problem.

    Two modes:

      * Pass a free-form equation as the argument; the solver type is
        auto-detected (root-finding if ``=`` is present and it's an
        algebraic equation; IVP if ``y'`` or ``dy/dx`` is present).
      * Pass a path to a JSON file (``.json``) for full control.

    Examples::

        mathsteps solve "x**3 - x - 2"
        mathsteps solve "x**2 - 4 = 0"
        mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
        mathsteps solve examples/linear_system_3x3.json
    """
    if expression is None:
        raise typer.BadParameter("Pass an expression or a path to a .json problem file.")

    with _user_errors():
        if expression.lower().endswith(".json") or Path(expression).is_file():
            problem = _load_problem(Path(expression))
        else:
            problem = detect_and_build(expression)
            if method:
                problem["method"] = normalize_method(method)
            if y0 is not None:
                problem["y0"] = float(y0)
            if x0 is not None:
                problem["x0"] = float(x0)
            if x_end is not None:
                problem["x_end"] = float(x_end)
            if h is not None:
                problem["h"] = float(h)
            if a is not None and "a" in problem:
                problem["a"] = float(a)
            if c is not None and "b" in problem:
                problem["b"] = float(c)
            if tol is not None:
                problem["tol"] = float(tol)
            if max_iter is not None:
                problem["max_iter"] = int(max_iter)

        _solve_and_print(problem, no_verify=no_verify, pretty=pretty)


@app.command()
def verify(problem_file: Path = typer.Argument(...)) -> None:
    """Solve a JSON problem file and verify it.

    Exit code 0 = PASS, 1 = FAIL, 2 = NOT CHECKED (no independent reference).
    """
    with _user_errors():
        problem = _load_problem(problem_file)
        result = _solve(problem, verify=True)
    verified = result.verified
    typer.echo(verdict(verified))
    # Exit codes: 0 = verified, 1 = verification failed, 2 = could not be checked.
    raise typer.Exit(code={True: 0, False: 1, None: 2}[verified])


@app.command("list-solvers")
def list_solvers() -> None:
    """List every registered solver."""
    for s in all_solvers():
        typer.echo(f"- {s.name}")


# --- Interactive wizard -----------------------------------------------------


@app.command()
def ask() -> None:
    """Interactive wizard: pick a problem type and answer prompts."""
    with _user_errors():
        _ask_wizard()


def _ask_wizard() -> None:
    typer.echo("MathSteps — what do you want to solve?\n")
    menu = [
        ("1", "Linear system  Ax = b", "linear_system"),
        ("2", "Root  f(x) = 0", "root"),
        ("3", "IVP  y' = f(x, y), y(x0) = y0", "ivp"),
        ("4", "BVP  two-point boundary value problem", "bvp"),
        ("5", "Matrix determinant", "determinant"),
        ("6", "Matrix inverse", "inverse"),
        ("7", "Eigenvalues", "eigen"),
        ("8", "Polynomial interpolation", "interp"),
        ("9", "Numerical integration", "integrate"),
        ("10", "Numerical derivative", "diff"),
    ]
    for num, label, _ in menu:
        typer.echo(f"  [{num}] {label}")
    typer.echo("")
    choice = typer.prompt("Pick a number", default="1").strip()
    domain_by_number = {num: dom for num, _, dom in menu}
    domains = set(domain_by_number.values())
    if choice in domain_by_number:
        domain = domain_by_number[choice]
    elif choice.lower().replace("-", "_") in domains:
        domain = choice.lower().replace("-", "_")
    else:
        raise typer.BadParameter(
            f"Choice must be a number 1-{len(menu)} or one of {sorted(domains)}."
        )

    problem: dict[str, Any] = {}

    if domain == "linear_system":
        A = typer.prompt("Matrix A (rows separated by ';')", default="1 2; 3 4")
        b = typer.prompt("Vector b", default="5 11")
        problem = build_problem_from_inputs("linear_system", A=A, b=b)

    elif domain == "root":
        f_expr = typer.prompt("Function f(x)", default="x**3 - x - 2")
        method = typer.prompt("Method (newton_raphson|secant|bisection|fixed_point)", default="newton_raphson")
        method = normalize_method(method)
        guesses: dict[str, str] = {}
        if method == "bisection":
            guesses["a"] = typer.prompt("Bracket left a", default="1")
            guesses["c"] = typer.prompt("Bracket right b", default="2")
        elif method == "secant":
            guesses["x0"] = typer.prompt("Initial guess x0", default="1")
            guesses["x1"] = typer.prompt("Second guess x1", default="2")
        else:
            guesses["x0"] = typer.prompt("Initial guess x0", default="0.5")
        problem = build_problem_from_inputs(
            "root", function=f_expr, method=method, variable="x", **guesses
        )

    elif domain == "ivp":
        f_expr = typer.prompt("dy/dx = ", default="-2*x*y")
        method = typer.prompt("Method (euler|heun|midpoint|rk4|rk45)", default="rk4")
        problem = build_problem_from_inputs(
            "ivp", f_expr=f_expr, method=method,
            y0=typer.prompt("y(x0)", default="1"),
            x0=typer.prompt("x0", default="0"),
            x_end=typer.prompt("x_end", default="2"),
            h=typer.prompt("step h (or empty for rk45)", default="0.1"),
        )

    elif domain == "bvp":
        method = typer.prompt("Method (shooting|finite_difference)", default="shooting")
        problem = build_problem_from_inputs(
            "bvp", method=method,
            f_expr=typer.prompt("y'' = ", default="sin(x) - y"),
            a=typer.prompt("y(a)", default="0"),
            c=typer.prompt("y(b)", default="0"),
            x0=typer.prompt("a (left x)", default="0"),
            x_end=typer.prompt("b (right x)", default="1"),
            h=typer.prompt("step h", default="0.01"),
            s0=typer.prompt("shooting s0", default="0"),
            s1=typer.prompt("shooting s1", default="1"),
        )

    elif domain == "determinant":
        A = typer.prompt("Square matrix A", default="6 1 1; 4 -2 5; 2 8 7")
        problem = build_problem_from_inputs("determinant", A=A, method="cofactor")

    elif domain == "inverse":
        A = typer.prompt("Square matrix A", default="1 2 3; 0 1 4; 5 6 0")
        problem = build_problem_from_inputs("matrix_inverse", A=A)

    elif domain == "eigen":
        A = typer.prompt("Square matrix A", default="2 0 0; 0 3 4; 0 4 9")
        problem = build_problem_from_inputs("eigen", A=A)

    elif domain == "interp":
        pts = typer.prompt("Points '(x0,y0); (x1,y1); ...'", default="(0,1); (1,2); (2,5)")
        method = typer.prompt("Method (lagrange|newton_divided_differences)", default="lagrange")
        problem = build_problem_from_inputs("interp", points=pts, method=method)

    elif domain == "integrate":
        f_expr = typer.prompt("Function f(x)", default="sin(x)")
        method = typer.prompt("Method (trapezoidal|simpson)", default="simpson")
        problem = build_problem_from_inputs(
            "integrate", function=f_expr, method=method,
            a=typer.prompt("Interval start a", default="0"),
            c=typer.prompt("Interval end b", default="3.14"),
            n=typer.prompt("Number of sub-intervals n", default="100"),
        )

    elif domain == "diff":
        f_expr = typer.prompt("Function f(x)", default="sin(x)")
        method = typer.prompt("Method (forward|backward|central)", default="central")
        problem = build_problem_from_inputs(
            "diff", function=f_expr, method=method,
            x=typer.prompt("x", default="1"),
            h=typer.prompt("step h", default="0.001"),
        )

    pretty = typer.confirm("Pretty output?", default=False)
    _solve_and_print(problem, no_verify=False, pretty=pretty)


if __name__ == "__main__":
    app()

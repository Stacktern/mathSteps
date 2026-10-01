"""Public Python API for ``mathsteps``.

This module exposes one function per supported problem type. Every
function returns a :class:`Result` containing the step-by-step
explanation and the final answer as plain Python / NumPy values.

Typical usage::

    import mathsteps

    result = mathsteps.linear_system(A=[[1, 2], [3, 4]], b=[5, 11])
    print(result.answer)        # [1. 2.]   (numpy array)
    print(result.exact)         # [1, 2]    (exact SymPy numbers)
    for step in result.steps:
        print(step.description)

    result = mathsteps.root("x**3 - x - 2", method="newton", x0=1.5)
    print(result.answer, result.verified)   # 1.5213797... True
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .core.answers import InterpolatingPolynomial, finalize
from .core.convergence import ConvergenceWarning, converged_from_steps
from .core.inputs import normalize_method
from .core.registry import find_solver
from .core.step import Step
from .verify import verdict, verify_problem


def _short(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.size > 6:
        return f"array(shape={value.shape})"
    if isinstance(value, tuple):
        return "(" + ", ".join(_short(v) for v in value) + ")"
    return repr(value)


@dataclass
class Result:
    """The outcome of running a solver.

    Attributes
    ----------
    solver : str
        Name of the solver that handled the problem.
    steps : list[Step]
        Step-by-step explanation. Each ``Step`` has ``description``,
        ``before``, ``after``, and a ``data`` dict carrying structured
        payload.
    answer : Any
        The final answer as plain Python / NumPy values: a ``float`` for a
        scalar (root, integral, derivative, determinant), a ``numpy.ndarray``
        for vectors and matrices, a tuple of arrays for ``lu`` and
        ``eigenvalues``, and a callable
        :class:`~mathsteps.core.answers.InterpolatingPolynomial` for the
        interpolation functions.
    exact : Any
        The exact SymPy form of the answer (rationals, matrices, expressions)
        for the solvers that work exactly; ``None`` for numerical methods.
    details : dict
        Extra outputs that go with the answer, e.g. ``details["x"]`` /
        ``details["y"]`` (the grid and solution) for ODE and BVP problems.
    verified : bool or None
        ``True`` if the answer was checked against a trusted reference
        (NumPy / SciPy / SymPy) and matches, ``False`` if it was checked
        and does **not** match, ``None`` if it was not checked (you passed
        ``verify=False``, or the check could not be run). ``None`` is not
        a pass.
    problem : dict
        The problem dict that was passed to the solver. Useful for
        debugging or for re-running with different settings.
    converged : bool or None
        For iterative solvers (Newton, secant, bisection, fixed-point,
        shooting): ``True`` if the tolerance was met, ``False`` if
        ``max_iter`` ran out first (a :class:`ConvergenceWarning` is also
        emitted). ``None`` for solvers that do not iterate to a tolerance.
    """

    solver: str
    steps: list[Step]
    answer: Any
    verified: bool | None
    problem: dict = field(default_factory=dict)
    converged: bool | None = None
    exact: Any = None
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Solver: {self.solver}", ""]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}: {step.description}")
            if step.before:
                lines.append(f"  before: {step.before}")
            if step.after and step.after != step.before:
                lines.append(f"  after:  {step.after}")
            lines.append("")
        lines.append(f"Final answer: {self.answer}")
        if self.exact is not None and not isinstance(self.answer, InterpolatingPolynomial):
            exact = str(self.exact)
            lines.append(f"Exact: {exact if len(exact) <= 400 else exact[:400] + ' ...'}")
        if self.converged is not None:
            lines.append(f"Converged: {'yes' if self.converged else 'NO (max_iter reached)'}")
        lines.append(f"Verification: {verdict(self.verified)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Result(solver={self.solver!r}, answer={_short(self.answer)}, "
            f"verified={self.verified}, converged={self.converged}, "
            f"steps={len(self.steps)})"
        )


def _run(problem: dict, *, verify: bool = True, detail: str | None = None) -> Result:
    """Internal helper: find a solver, run it, optionally verify."""
    if detail is not None:
        problem["detail"] = detail
    solver = find_solver(problem)
    steps, raw = solver.solve(problem)
    # Verify the solver's native (exact) answer: it is the stricter check.
    verified = verify_problem(problem, raw) if verify else None
    converged = converged_from_steps(steps)
    if converged is False:
        warnings.warn(
            f"{solver.name} did not converge within max_iter="
            f"{problem.get('max_iter', 'default')}; the returned answer is the "
            f"last iterate. Increase max_iter, loosen tol, or change the starting point.",
            ConvergenceWarning,
            stacklevel=3,
        )
    answer, exact, details = finalize(problem, raw, steps)
    return Result(
        solver=solver.name,
        steps=steps,
        answer=answer,
        verified=verified,
        problem=problem,
        converged=converged,
        exact=exact,
        details=details,
    )


# --------------------------------------------------------------------------
# Public API: one function per problem type
# --------------------------------------------------------------------------


def linear_system(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    verify: bool = True,
    detail: str | None = None,
) -> Result:
    """Solve a linear system ``Ax = b`` via Gaussian elimination.

    Parameters
    ----------
    A : 2D sequence of numbers
        Square coefficient matrix.
    b : 1D sequence of numbers
        Right-hand side vector.
    verify : bool, default True
        If True, check the answer against ``numpy.linalg.solve``
        (``result.verified`` is ``None`` when False).
    detail : {"full", "summary", "none"}, optional
        How many steps to record. Default: every step for systems up to
        10x10, a per-pivot summary for larger ones.

    Returns
    -------
    Result
        ``result.answer`` is the ``numpy`` vector ``x``; ``result.exact`` the
        exact rationals.

    Example
    -------
    >>> import mathsteps
    >>> r = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
    >>> r.answer
    array([1., 2.])
    """
    return _run(
        {"type": "linear_system", "A": [list(row) for row in A], "b": list(b)},
        verify=verify,
        detail=detail,
    )


def determinant(A: Sequence[Sequence[float]], *, verify: bool = True) -> Result:
    """Compute the determinant of a square matrix via cofactor expansion.

    ``result.answer`` is a ``float``; ``result.exact`` the exact value.

    Example
    -------
    >>> import mathsteps
    >>> mathsteps.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).answer
    -306.0
    """
    return _run(
        {
            "type": "determinant", "method": "cofactor",
            "A": [list(row) for row in A],
        },
        verify=verify,
    )


def inverse(
    A: Sequence[Sequence[float]], *, verify: bool = True, detail: str | None = None
) -> Result:
    """Compute the inverse of a square matrix via Gauss-Jordan elimination.

    ``result.answer`` is a ``numpy`` array; ``result.exact`` a ``sympy.Matrix``.

    Example
    -------
    >>> import mathsteps
    >>> import numpy as np
    >>> r = mathsteps.inverse([[1, 2], [3, 4]])
    >>> np.allclose(r.answer, np.linalg.inv([[1, 2], [3, 4]]))
    True
    """
    return _run(
        {"type": "matrix_inverse", "A": [list(row) for row in A]},
        verify=verify,
        detail=detail,
    )


def lu(
    A: Sequence[Sequence[float]], *, verify: bool = True, detail: str | None = None
) -> Result:
    """LU decomposition with partial pivoting.

    ``result.answer`` is the tuple of arrays ``(P, L, U)`` such that
    ``P @ A = L @ U``.
    """
    return _run(
        {"type": "lu_decomposition", "A": [list(row) for row in A]},
        verify=verify,
        detail=detail,
    )


def eigenvalues(A: Sequence[Sequence[float]], *, verify: bool = True) -> Result:
    """Eigenvalues and eigenvectors via the characteristic polynomial.

    ``result.answer`` is ``(values, vectors)`` in the layout of
    ``numpy.linalg.eig``: ``vectors[:, j]`` is a unit-norm eigenvector for
    ``values[j]``, so ``A @ vectors == vectors * values``. There is one
    column per *independent* eigenvector: an eigenvalue with a
    multi-dimensional eigenspace appears several times, and a defective
    matrix has fewer than ``n`` columns.

    Eigenvalues come out exactly (``result.exact``) when the characteristic
    polynomial factors into degree <= 2 pieces over the rationals, and as
    15-digit numerical roots otherwise.
    """
    return _run(
        {"type": "eigenvalues", "A": [list(row) for row in A]},
        verify=verify,
    )


def cramers_rule(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    *,
    verify: bool = True,
) -> Result:
    """Solve ``Ax = b`` via Cramer's rule."""
    return _run(
        {"type": "cramers_rule", "A": [list(row) for row in A], "b": list(b)},
        verify=verify,
    )


def root(
    function: str,
    *,
    variable: str = "x",
    method: str = "newton_raphson",
    a: float | None = None,
    b: float | None = None,
    x0: float | None = None,
    x1: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    verify: bool = True,
) -> Result:
    """Find a root of ``f(x) = 0``.

    Parameters
    ----------
    function : str
        SymPy-compatible expression in ``variable``.
    variable : str, default "x"
    method : str, default "newton_raphson"
        One of ``"newton_raphson"`` (or alias ``"newton"``),
        ``"bisection"``, ``"secant"``, ``"fixed_point"``.
    a, b : float, optional
        Bisection: bracket endpoints. Required for ``method="bisection"``.
    x0 : float, optional
        Initial guess. Required for Newton / fixed-point.
    x1 : float, optional
        Second guess. Required for secant.
    tol : float, default 1e-10
    max_iter : int, default 50
    verify : bool, default True

    Returns
    -------
    Result
        ``result.answer`` is the root as a float.

    Examples
    --------
    >>> import mathsteps
    >>> r = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
    >>> round(r.answer, 6)
    0.739085
    >>> r = mathsteps.root("x**3 - x - 2", method="bisection", a=1, b=2)
    >>> round(r.answer, 6)
    1.52138
    """
    method = normalize_method(method)
    problem: dict[str, Any] = {
        "type": "root_finding",
        "method": method,
        "function": function,
        "variable": variable,
        "tol": tol,
        "max_iter": max_iter,
    }
    if method == "bisection":
        if a is None or b is None:
            raise ValueError("bisection requires `a` and `b` (bracket endpoints).")
        problem["a"] = float(a)
        problem["b"] = float(b)
    elif method == "secant":
        if x0 is None or x1 is None:
            raise ValueError("secant requires `x0` and `x1`.")
        problem["x0"] = float(x0)
        problem["x1"] = float(x1)
    elif method == "fixed_point":
        if x0 is None:
            raise ValueError("fixed_point requires `x0`.")
        problem["x0"] = float(x0)
    else:
        if x0 is None:
            raise ValueError(f"{method} requires `x0`.")
        problem["x0"] = float(x0)
    return _run(problem, verify=verify)


def ivp(
    f_expr: str | Sequence[str],
    *,
    y0: float | Sequence[float] = 1.0,
    x0: float = 0.0,
    x_end: float = 1.0,
    h: float | None = None,
    method: str = "rk4",
    variable: str = "x",
    function: str | Sequence[str] | None = None,
    verify: bool = True,
    detail: str | None = None,
) -> Result:
    """Solve an initial value problem ``dy/dx = f(x, y)``, ``y(x0) = y0``.

    Works for one equation or a **system**, forwards or backwards (``x_end <
    x0``).

    Parameters
    ----------
    f_expr : str or sequence of str
        Right-hand side(s), e.g. ``"-2*x*y"`` (the ``y' =`` prefix is
        accepted), or one expression per unknown for a system.
    y0 : float or sequence of float
        Initial value(s) at ``x0``; one per unknown.
    x0, x_end : float
        Start and end of the integration interval.
    h : float, optional
        Step size, always positive. Not needed for ``method="rk45"``
        (adaptive). If it does not divide the interval it is shrunk so the
        last step lands exactly on ``x_end``.
    method : str, default "rk4"
        ``"euler"``, ``"heun"``, ``"midpoint"``, ``"rk4"``, ``"rk45"``.
    variable : str
        Name of the independent variable used in ``f_expr``.
    function : str or sequence of str, optional
        Name(s) of the unknown(s): ``"y"`` for one equation, ``y1, y2, ...``
        for a system unless you give them.
    detail : {"full", "summary", "none"}, optional
        How many steps to record (default: all up to 200 steps, a summary
        beyond that). The answer and ``details`` are the same either way.

    Returns
    -------
    Result
        ``result.answer`` is ``y(x_end)``: a float, or an array for a system.
        ``result.details["x"]`` and ``result.details["y"]`` hold the whole
        trajectory (``y`` has shape ``(points, unknowns)``).

    Examples
    --------
    >>> import mathsteps, math
    >>> r = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")
    >>> abs(r.answer - math.exp(-4)) < 1e-4
    True

    A harmonic oscillator ``x'' = -4 x`` as a first-order system:

    >>> r = mathsteps.ivp(["v", "-4*x"], variable="t", function=["x", "v"],
    ...                   y0=[1, 0], x_end=3, h=0.01)
    >>> r.answer.shape
    (2,)
    """
    is_system = not isinstance(f_expr, str)
    if is_system:
        rhs: Any = [str(e) for e in f_expr]
        if function is None:
            names: Any = [f"y{i + 1}" for i in range(len(rhs))]
        elif isinstance(function, str):
            raise ValueError("For a system, `function` must be a list of unknown names.")
        else:
            names = [str(n) for n in function]
        if isinstance(y0, (int, float)) or len(list(y0)) != len(rhs):
            raise ValueError(
                f"y0 must give one initial value per equation ({len(rhs)}); got {y0!r}."
            )
        y0_value: Any = [float(v) for v in y0]
    else:
        names = function if function is not None else "y"
        if not isinstance(names, str):
            raise ValueError("For a single equation, `function` must be a name like 'y'.")
        rhs = _strip_ivp_prefix(f_expr, variable, names)
        y0_value = float(y0)  # type: ignore[arg-type]
    problem: dict[str, Any] = {
        "type": "ivp",
        "method": method,
        "f_expr": rhs,
        "function": names,
        "variable": variable,
        "y0": y0_value,
        "x0": float(x0),
        "x_end": float(x_end),
    }
    if method == "rk45":
        problem["atol"] = 1e-9
        problem["rtol"] = 1e-9
    else:
        if h is None:
            raise ValueError(f"{method} requires `h` (step size).")
        problem["h"] = float(h)
    return _run(problem, verify=verify, detail=detail)


def ivp_second_order(
    f_expr: str,
    *,
    y0: float,
    dy0: float,
    x0: float = 0.0,
    x_end: float = 1.0,
    h: float | None = None,
    method: str = "rk4",
    variable: str = "x",
    function: str = "y",
    verify: bool = True,
    detail: str | None = None,
) -> Result:
    """Solve a second-order IVP ``y'' = f(x, y, y')``, ``y(x0) = y0``, ``y'(x0) = dy0``.

    Write the right-hand side with ``y`` for the unknown and ``yp`` for its
    derivative (``<function>p`` if you rename the unknown). It is solved as
    the system ``y' = yp, yp' = f``.

    ``result.answer`` is the array ``[y(x_end), y'(x_end)]``;
    ``result.details["y"][:, 0]`` is the solution ``y`` along
    ``result.details["x"]``.

    Example
    -------
    A damped spring ``m x'' = -k x - c x'`` with ``m=1, k=4, c=0.5``:

    >>> import mathsteps
    >>> r = mathsteps.ivp_second_order("-4*x - 0.5*xp", variable="t", function="x",
    ...                                y0=1, dy0=0, x_end=10, h=0.01)
    >>> r.answer.shape
    (2,)
    """
    deriv = f"{function}p"
    s = f_expr.strip()
    s = re.sub(rf"^d2{function}\s*/\s*d{variable}2\s*=\s*", "", s)
    s = re.sub(rf"^{function}''\s*=\s*", "", s)
    return ivp(
        [deriv, s.strip()],
        y0=[y0, dy0],
        x0=x0,
        x_end=x_end,
        h=h,
        method=method,
        variable=variable,
        function=[function, deriv],
        verify=verify,
        detail=detail,
    )


def bvp_shooting(
    f_expr: str,
    *,
    y_left: float = 0.0,
    y_right: float = 0.0,
    x0: float = 0.0,
    x_end: float = 1.0,
    h: float = 0.01,
    s0: float = 0.0,
    s1: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 30,
    variable: str = "x",
    verify: bool = True,
) -> Result:
    """Solve a 2nd-order BVP via the shooting method.

    Solves ``y'' = f(x, y, y')`` (write ``y`` and ``yp`` in ``f_expr``) with
    ``y(x0) = y_left``, ``y(x_end) = y_right`` by guessing the initial slope
    and adjusting via secant.

    ``result.answer`` is the solution ``y`` on the grid
    ``result.details["x"]``; ``result.details["slope"]`` is the converged
    ``y'(x0)`` and ``result.details["dy"]`` is ``y'`` along the grid.

    Example
    -------
    >>> import mathsteps
    >>> r = mathsteps.bvp_shooting("sin(x) - y", y_left=0, y_right=0, x_end=1)
    >>> abs(r.answer[-1]) < 1e-3          # hits the right boundary value
    True
    """
    problem = {
        "type": "bvp",
        "method": "shooting",
        "f_expr": f_expr,
        "variable": variable,
        "y_left": float(y_left),
        "y_right": float(y_right),
        "x0": float(x0),
        "x_end": float(x_end),
        "h": float(h),
        "s0": float(s0),
        "s1": float(s1),
        "tol": float(tol),
        "max_iter": int(max_iter),
    }
    return _run(problem, verify=verify)


def bvp_finite_difference(
    *,
    p_expr: str = "0",
    q_expr: str = "1",
    r_expr: str = "0",
    a: float = 0.0,
    b: float = 1.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    n: int = 10,
    variable: str = "x",
    verify: bool = True,
) -> Result:
    """Solve a 2nd-order linear BVP via finite differences.

    Discretises ``-y'' + p(x) y' + q(x) y = r(x)`` on ``[a, b]`` with
    ``y(a) = alpha``, ``y(b) = beta`` into an ``n x n`` tridiagonal system
    (``n`` interior points) and solves it.

    ``result.answer`` is ``y`` on the ``n + 2`` grid points (both boundary
    points included); the grid is ``result.details["x"]``.
    """
    problem = {
        "type": "bvp",
        "method": "finite_difference",
        "p_expr": p_expr,
        "q_expr": q_expr,
        "r_expr": r_expr,
        "variable": variable,
        "a": float(a),
        "b": float(b),
        "alpha": float(alpha),
        "beta": float(beta),
        "n": int(n),
    }
    return _run(problem, verify=verify)


def lagrange(points: Iterable[Sequence[float]], *, verify: bool = True) -> Result:
    """Polynomial interpolation through the given points (Lagrange form).

    ``result.answer`` is a callable polynomial: ``p = r.answer; p(2.5)``
    (scalars or arrays; exact internally). ``p.expr`` is the SymPy form.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")
    return _run(
        {"type": "interpolation", "method": "lagrange", "points": pts},
        verify=verify,
    )


def newton_divided_differences(
    points: Iterable[Sequence[float]], *, verify: bool = True
) -> Result:
    """Polynomial interpolation via Newton's divided differences.

    ``result.answer`` is a callable polynomial, as for :func:`lagrange`.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")
    return _run(
        {
            "type": "interpolation",
            "method": "newton_divided_differences",
            "points": pts,
        },
        verify=verify,
    )


def integrate(
    function: str,
    *,
    a: float,
    b: float,
    n: int = 100,
    method: str = "trapezoidal",
    variable: str = "x",
    verify: bool = True,
) -> Result:
    """Numerical integration of ``f(x)`` over ``[a, b]``.

    Parameters
    ----------
    function : str
        SymPy-compatible expression in ``variable``.
    a, b : float
        Interval endpoints.
    n : int, default 100
        Number of sub-intervals (must be even for Simpson's rule).
    method : str, default "trapezoidal"
        ``"trapezoidal"`` or ``"simpson"``.
    variable : str, default "x"
    verify : bool, default True
        Check against ``scipy.integrate.quad`` within the rule's own
        truncation error.

    Example
    -------
    >>> import mathsteps, math
    >>> r = mathsteps.integrate("sin(x)", a=0, b=math.pi, n=100, method="simpson")
    >>> abs(r.answer - 2.0) < 1e-6
    True
    """
    return _run(
        {
            "type": "numerical_integration",
            "function": function,
            "variable": variable,
            "a": float(a),
            "b": float(b),
            "n": int(n),
            "method": normalize_method(method),
        },
        verify=verify,
    )


def differentiate(
    function: str,
    *,
    x: float = 1.0,
    h: float = 0.001,
    method: str = "central",
    variable: str = "x",
    verify: bool = True,
) -> Result:
    """Numerical derivative of ``f(x)`` at ``x`` via finite differences.

    Parameters
    ----------
    function : str
    x : float, default 1.0
    h : float, default 0.001
    method : str, default "central"
        ``"forward"``, ``"backward"``, or ``"central"``.
    variable : str, default "x"
    verify : bool, default True
        Check against the exact SymPy derivative within the formula's own
        truncation + rounding error.

    Example
    -------
    >>> import mathsteps, math
    >>> r = mathsteps.differentiate("sin(x)", x=1, h=1e-3, method="central")
    >>> abs(r.answer - math.cos(1)) < 1e-5
    True
    """
    return _run(
        {
            "type": "numerical_diff",
            "function": function,
            "variable": variable,
            "x": float(x),
            "h": float(h),
            "method": method,
        },
        verify=verify,
    )


def solve_problem(problem: dict, *, verify: bool = True) -> Result:
    """Generic entry point: pass a problem dict, get a ``Result`` back.

    This is the lowest-level public API. It dispatches to the right
    solver based on ``problem["type"]`` and ``problem["method"]``.

    Example
    -------
    >>> import mathsteps
    >>> r = mathsteps.solve_problem({
    ...     "type": "root_finding", "method": "newton_raphson",
    ...     "function": "x**2 - 2", "variable": "x", "x0": 1.0,
    ...     "tol": 1e-12, "max_iter": 50,
    ... })
    >>> abs(r.answer - 2**0.5) < 1e-10
    True
    """
    return _run(problem, verify=verify)


def available_solvers() -> list[str]:
    """Return the names of every registered solver."""
    from .core.registry import all_solvers

    return [s.name for s in all_solvers()]


def _strip_ivp_prefix(f_expr: str, var: str, func: str) -> str:
    """Strip ``y' =`` / ``dy/dx =`` / ``d?/d? =`` from an ODE string."""
    s = f_expr.strip()
    s = re.sub(rf"^d{func}\s*/\s*d{var}\s*=\s*", "", s)
    s = re.sub(rf"^d{var}\s*/\s*d{var}\s*=\s*", "", s)
    s = re.sub(rf"^{func}'\s*=\s*", "", s)
    return s.strip()

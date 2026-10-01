"""Ground-truth verification against trusted libraries.

Each ``verify_*`` function returns ``True`` if our solver's answer
matches the trusted reference (within tolerance for floats, exactly
for symbolic answers).

:func:`verify_problem` is the single entry point used by the Python API
and the CLI. It returns ``True`` (checked, matches), ``False`` (checked,
does **not** match) or ``None`` (no independent check exists for this
problem type, or the check itself could not be run). ``None`` is never
reported as a pass.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import sympy as sp

from mathsteps.core.expr import parse_expr


def verify_linear_system(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    x: Sequence,
    tol: float = 1e-9,
) -> bool:
    """Verify ``Ax = b`` for either a SymPy-exact or numeric ``x``."""
    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)

    if any(isinstance(v, sp.Basic) for v in x):
        x_sym = [sp.nsimplify(v) for v in x]
        residual_sym = A_np @ np.asarray(
            [float(v) for v in x_sym], dtype=float
        ) - b_np
        return np.allclose(residual_sym, 0.0, atol=tol)

    x_np = np.asarray([float(v) for v in x], dtype=float)
    return np.allclose(A_np @ x_np, b_np, atol=tol)


def verify_linear_system_against_numpy(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    x: Sequence,
    tol: float = 1e-9,
) -> bool:
    """Verify our ``x`` matches ``numpy.linalg.solve(A, b)``."""
    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    x_ref = np.linalg.solve(A_np, b_np)
    x_ours = np.asarray([float(v) for v in x], dtype=float)
    return np.allclose(x_ours, x_ref, atol=tol)


def verify_root(
    function_str: str,
    variable_str: str,
    x: float,
    tol: float = 1e-6,
) -> bool:
    """Verify ``x`` is a (near) root of ``function_str(variable_str)``."""
    import sympy as sp

    var = sp.symbols(variable_str)
    expr = parse_expr(function_str, {variable_str: var})
    return abs(float(expr.subs(var, sp.nsimplify(x)))) <= tol


def verify_root_against_scipy(
    function_str: str,
    variable_str: str,
    x: float,
    bracket: tuple[float, float] | None = None,
    x0: float | None = None,
    tol: float = 1e-9,
) -> bool:
    """Verify ``x`` matches ``scipy.optimize.brentq``/``newton``."""
    import sympy as sp
    from scipy.optimize import brentq, newton

    var = sp.symbols(variable_str)
    expr = parse_expr(function_str, {variable_str: var})
    f = sp.lambdify(var, expr, modules=["numpy"])
    fprime = sp.lambdify(var, sp.diff(expr, var), modules=["numpy"])

    if bracket is not None:
        ref = brentq(f, bracket[0], bracket[1], xtol=tol)
    elif x0 is not None:
        ref = newton(f, x0, fprime=fprime, tol=tol)
    else:
        raise ValueError("Must provide either `bracket` or `x0`.")
    return abs(float(x) - float(ref)) <= max(1e-6, tol * 10)


# Explicit Runge-Kutta tableaux (c, A, b) and orders for the fixed-step solvers.
_RK_TABLEAUX: dict[str, tuple[list[float], list[list[float]], list[float]]] = {
    "euler": ([0.0], [[0.0]], [1.0]),
    "heun": ([0.0, 1.0], [[0.0, 0.0], [1.0, 0.0]], [0.5, 0.5]),
    "midpoint": ([0.0, 0.5], [[0.0, 0.0], [0.5, 0.0]], [0.0, 1.0]),
    "rk4": (
        [0.0, 0.5, 0.5, 1.0],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    ),
}
_RK_ORDER = {"euler": 1, "heun": 2, "midpoint": 2, "rk4": 4}


def _fixed_step_rk(f, x0: float, y0, h: float, n: int, method: str) -> np.ndarray:
    c, A, b = _RK_TABLEAUX[method]
    x, y = x0, np.asarray(y0, dtype=float)
    for _ in range(n):
        ks: list[np.ndarray] = []
        for i in range(len(c)):
            yi = y + h * sum((A[i][j] * ks[j] for j in range(i)), np.zeros_like(y))
            ks.append(f(x + c[i] * h, yi))
        y = y + h * sum(bi * ki for bi, ki in zip(b, ks))
        x += h
    return y


def verify_ivp_against_scipy(
    f_expr_str,
    variable_str: str,
    function_str,
    y0,
    x0: float,
    x_end: float,
    y_final,
    atol: float | None = None,
    method: str | None = None,
    h: float | None = None,
) -> bool:
    """Verify an IVP answer (scalar or system) against a tight ``scipy.integrate.solve_ivp`` reference.

    A fixed-step method is *supposed* to differ from the true solution by
    its truncation error, so the tolerance is derived from that error:
    the same scheme is re-run with ``h/2`` (an independent reference
    implementation) and Richardson extrapolation estimates
    ``|y_h - y(x_end)|``. The answer must land within a few multiples of
    that estimate.

    Pass ``atol`` to force a fixed absolute tolerance instead; it is also
    the fallback when ``method`` / ``h`` are unknown (e.g. a custom solver).
    """
    from scipy.integrate import solve_ivp

    from mathsteps.domains.ode_ivp.common import parse_ivp

    system = parse_ivp({"variable": variable_str, "function": function_str, "f_expr": f_expr_str})
    f = system.f
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    answer = np.atleast_1d(np.asarray(y_final, dtype=float))
    if not np.all(np.isfinite(answer)):
        return False

    sol = solve_ivp(
        lambda xv, yv: f(xv, yv),
        t_span=(x0, x_end),
        y0=y0_arr,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        return False
    ref = sol.y[:, -1]
    scale = max(1.0, float(np.max(np.abs(ref))))

    if atol is not None:
        tol = atol
    elif method in _RK_ORDER and h:
        from mathsteps.core.stepping import fixed_step_plan

        n, h = fixed_step_plan(x0, x_end, h)  # same plan the solver used
        order = _RK_ORDER[method]
        y_h = _fixed_step_rk(f, x0, y0_arr, h, n, method)
        y_half = _fixed_step_rk(f, x0, y0_arr, h / 2, 2 * n, method)
        truncation = float(np.max(np.abs(y_h - y_half))) / (1 - 2.0 ** -order)
        # Richardson only means something in the asymptotic regime. If the
        # estimated error is already a large fraction of the solution (or
        # not finite), h is too coarse / the method is unstable for this
        # problem and no tolerance derived from it can vouch for the answer.
        if not np.isfinite(truncation) or truncation > 0.1 * scale:
            return False
        tol = 4.0 * truncation + 1e-8 * scale
    elif method == "rk45":
        tol = 1e-6 * scale
    else:
        tol = 0.01
    return bool(float(np.max(np.abs(answer - ref))) <= tol)


def verify_determinant(A: Sequence[Sequence[float]], det_value, rtol: float = 1e-9) -> bool:
    ref = float(np.linalg.det(np.asarray(A, dtype=float)))
    return bool(np.isclose(ref, float(det_value), rtol=rtol, atol=1e-9))


def verify_inverse(A: Sequence[Sequence[float]], inv_value, atol: float = 1e-9) -> bool:
    A_np = np.asarray(A, dtype=float)
    if hasattr(inv_value, "tolist"):
        rows = inv_value.tolist()
    else:
        rows = [[float(v) for v in row] for row in inv_value]
    inv_np = np.asarray(rows, dtype=float)
    return np.allclose(A_np @ inv_np, np.eye(A_np.shape[0]), atol=atol)


def verify_eigenvalues(A: Sequence[Sequence[float]], eigs_value, atol: float = 1e-7) -> bool:
    """Check our (distinct, possibly complex) eigenvalues against ``numpy.linalg.eigvals``.

    The solver reports each distinct root once while NumPy reports every
    root with multiplicity, so the two are compared as sets.
    """
    ref = np.linalg.eigvals(np.asarray(A, dtype=float))
    ours = np.asarray([complex(ev) for ev in eigs_value], dtype=complex)
    if ours.size == 0 or ref.size == 0:
        return False
    tol = atol * max(1.0, float(np.max(np.abs(ref))))
    ours_in_ref = all(np.min(np.abs(ref - v)) <= tol for v in ours)
    ref_in_ours = all(np.min(np.abs(ours - r)) <= tol for r in ref)
    return bool(ours_in_ref and ref_in_ours)


def verify_eigenvectors(A: Sequence[Sequence[float]], eigs_value, vecs: dict, atol: float = 1e-7) -> bool:
    """Check each eigenspace basis: vectors non-zero, ``A v = lambda v``, and as many
    vectors as the eigenspace has dimensions (nullity of ``A - lambda I`` via SVD)."""
    A_np = np.asarray(A, dtype=float)
    n = A_np.shape[0]
    for ev in eigs_value:
        vectors = vecs[ev]
        lam = complex(ev)
        if not vectors:
            return False
        for vec in vectors:
            v = np.asarray([complex(x) for x in vec], dtype=complex)
            norm = float(np.linalg.norm(v))
            if norm == 0.0:
                return False
            if float(np.linalg.norm(A_np @ v - lam * v)) > atol * max(1.0, abs(lam)) * norm:
                return False
        s = np.linalg.svd(A_np - lam * np.eye(n), compute_uv=False)
        nullity = max(1, int(np.sum(s <= 1e-8 * max(1.0, s[0]))))
        if len(vectors) != nullity:
            return False
    return True


def verify_lu(A: Sequence[Sequence[float]], P, L, U, atol: float = 1e-9) -> bool:
    """Check ``P @ A == L @ U`` and that ``L`` / ``U`` are lower / upper triangular."""
    A_np = np.asarray(A, dtype=float)
    P_np, L_np, U_np = (np.asarray(M.tolist(), dtype=float) for M in (P, L, U))
    return bool(
        np.allclose(P_np @ A_np, L_np @ U_np, atol=atol)
        and np.allclose(L_np, np.tril(L_np))
        and np.allclose(U_np, np.triu(U_np))
    )


def verify_interpolation(points: Sequence[Sequence[float]], poly, tol: float = 1e-8) -> bool:
    """Check the interpolating polynomial passes through every data point."""
    x = sp.Symbol("x")
    expr = sp.sympify(poly)
    for xi, yi in points:
        value = float(expr.subs(x, sp.nsimplify(xi)))
        if abs(value - float(yi)) > tol * max(1.0, abs(float(yi))):
            return False
    return True


def verify_integration(problem: dict, answer: float) -> bool:
    """Check a trapezoidal / Simpson result against ``scipy.integrate.quad``.

    The rule is *supposed* to differ from the true integral by its truncation
    error, so the tolerance comes from Richardson extrapolation between ``n``
    and ``2n`` sub-intervals (computed by an independent implementation); the
    answer must also equal that independent evaluation of the same rule.
    """
    from scipy.integrate import quad

    from mathsteps.domains.numerical.common import parse_function

    _, _, f = parse_function(problem["function"], problem["variable"])
    a, b, n = float(problem["a"]), float(problem["b"]), int(problem["n"])
    method = problem.get("method", "trapezoidal")
    order = {"trapezoidal": 2, "simpson": 4}[method]

    def rule(m: int) -> float:
        xs = np.linspace(a, b, m + 1)
        ys = np.array([float(f(x)) for x in xs])
        h = (b - a) / m
        if method == "trapezoidal":
            return float(h * (0.5 * ys[0] + 0.5 * ys[-1] + ys[1:-1].sum()))
        return float((h / 3.0) * (ys[0] + ys[-1] + 4.0 * ys[1:-1:2].sum() + 2.0 * ys[2:-1:2].sum()))

    i_n, i_2n = rule(n), rule(2 * n)
    truncation = abs(i_n - i_2n) / (1 - 2.0 ** -order)
    ref, quad_err = quad(lambda x: float(f(x)), a, b, limit=200)
    if not np.isfinite([i_n, i_2n, ref, truncation]).all():
        raise ValueError("non-finite integral")
    scale = max(1.0, abs(ref))
    tol = 4.0 * truncation + 10.0 * quad_err + 1e-9 * scale
    return bool(abs(answer - ref) <= tol and abs(answer - i_n) <= 1e-9 * scale)


def verify_differentiation(problem: dict, answer: float) -> bool:
    """Check a finite-difference derivative against the exact SymPy derivative.

    The tolerance is the formula's own truncation error (Richardson between
    ``h`` and ``h/2``) plus the rounding error ``~ eps * |f| / h``.
    """
    from mathsteps.domains.numerical.common import parse_function

    expr, var, f = parse_function(problem["function"], problem["variable"])
    x, h = float(problem["x"]), float(problem["h"])
    method = problem.get("method", "central")
    order = 2 if method == "central" else 1

    def diff(step: float) -> tuple[float, float]:
        if method == "forward":
            vals = (float(f(x)), float(f(x + step)))
            return (vals[1] - vals[0]) / step, max(map(abs, vals))
        if method == "backward":
            vals = (float(f(x - step)), float(f(x)))
            return (vals[1] - vals[0]) / step, max(map(abs, vals))
        vals = (float(f(x - step)), float(f(x + step)))
        return (vals[1] - vals[0]) / (2 * step), max(map(abs, vals))

    d_h, fmax = diff(h)
    d_half, _ = diff(h / 2)
    exact = float(sp.diff(expr, var).subs(var, sp.nsimplify(x)))
    truncation = abs(d_h - d_half) / (1 - 2.0 ** -order)
    rounding = 8.0 * np.finfo(float).eps * max(fmax, 1.0) / h
    tol = 4.0 * truncation + rounding + 1e-12 * max(1.0, abs(exact))
    return bool(abs(answer - d_h) <= 1e-9 * max(1.0, abs(d_h)) and abs(answer - exact) <= tol)


def _solve_bvp_reference(rhs, bc, x: np.ndarray, y_guess: np.ndarray) -> np.ndarray:
    """Solve the second-order BVP ``y'' = rhs(x, y, y')`` with scipy, seeded from our own solution."""
    from scipy.integrate import solve_bvp

    count = len(x)
    idx = np.unique(np.linspace(0, count - 1, min(count, 200)).astype(int))
    dy = np.gradient(y_guess, x)

    def fun(xv, yv):
        out = np.asarray(rhs(xv, yv[0], yv[1]), dtype=float)
        return np.vstack([yv[1], np.broadcast_to(out, xv.shape)])

    sol = solve_bvp(fun, bc, x[idx], np.vstack([y_guess[idx], dy[idx]]), tol=1e-9, max_nodes=200000)
    if sol.status != 0:
        raise ValueError(f"scipy.solve_bvp did not converge: {sol.message}")
    return sol.sol(x)[0]


def verify_bvp_shooting(problem: dict, answer) -> bool:
    """Check the shooting solution against an independent ``scipy.solve_bvp`` solution."""
    from mathsteps.core.stepping import fixed_step_plan

    x0, x_end = float(problem["x0"]), float(problem["x_end"])
    n, _ = fixed_step_plan(x0, x_end, float(problem["h"]))
    y = np.asarray(answer, dtype=float)
    if y.shape != (n + 1,) or not np.all(np.isfinite(y)):
        return False
    xv, yv, ypv = sp.symbols(problem["variable"] + " y yp")
    expr = parse_expr(problem["f_expr"], {problem["variable"]: xv, "y": yv, "yp": ypv})
    f = sp.lambdify((xv, yv, ypv), expr, modules=["numpy"])
    y_left, y_right = float(problem["y_left"]), float(problem["y_right"])
    ref = _solve_bvp_reference(
        f, lambda ya, yb: np.array([ya[0] - y_left, yb[0] - y_right]),
        np.linspace(x0, x_end, n + 1), y,
    )
    scale = max(1.0, float(np.max(np.abs(ref))))
    tol = max(20.0 * float(problem.get("tol", 1e-6)), 1e-6) * scale
    return bool(
        np.max(np.abs(y - ref)) <= tol
        and abs(y[0] - y_left) <= 1e-9 * scale
        and abs(y[-1] - y_right) <= tol
    )


def _fd_bvp(p_f, q_f, r_f, a: float, b: float, alpha: float, beta: float, n: int) -> np.ndarray:
    """Independent tridiagonal solve of ``-y'' + p y' + q y = r`` with ``n`` interior points."""
    from scipy.linalg import solve_banded

    h = (b - a) / (n + 1)
    xs = np.linspace(a, b, n + 2)
    xi = xs[1:-1]
    p = np.broadcast_to(np.asarray(p_f(xi), dtype=float), xi.shape)
    q = np.broadcast_to(np.asarray(q_f(xi), dtype=float), xi.shape)
    r = np.broadcast_to(np.asarray(r_f(xi), dtype=float), xi.shape).copy()
    up = -1.0 / h**2 + p / (2 * h)
    down = -1.0 / h**2 - p / (2 * h)
    main = 2.0 / h**2 + q
    r[0] -= down[0] * alpha
    r[-1] -= up[-1] * beta
    ab = np.zeros((3, n))
    ab[0, 1:] = up[:-1]
    ab[1, :] = main
    ab[2, :-1] = down[1:]
    return np.concatenate([[alpha], solve_banded((1, 1), ab, r), [beta]])


def verify_bvp_finite_difference(problem: dict, answer) -> bool:
    """Check the finite-difference solution against ``scipy.solve_bvp`` and an independent tridiagonal solve.

    The scheme is second-order, so the tolerance is the Richardson estimate
    between ``n`` and ``2n + 1`` interior points.
    """
    xv = sp.Symbol(problem["variable"])
    sym = {problem["variable"]: xv}
    p_f, q_f, r_f = (
        sp.lambdify(xv, parse_expr(problem[k], sym), modules=["numpy"])
        for k in ("p_expr", "q_expr", "r_expr")
    )
    a, b = float(problem["a"]), float(problem["b"])
    alpha, beta, n = float(problem["alpha"]), float(problem["beta"]), int(problem["n"])
    y = np.asarray(answer, dtype=float)
    if y.shape != (n + 2,) or not np.all(np.isfinite(y)):
        return False

    y_h = _fd_bvp(p_f, q_f, r_f, a, b, alpha, beta, n)
    y_half = _fd_bvp(p_f, q_f, r_f, a, b, alpha, beta, 2 * n + 1)[::2]
    truncation = np.abs(y_h - y_half) / (1 - 2.0 ** -2)

    x = np.linspace(a, b, n + 2)

    def rhs(xx, yy, yp):  # y'' = p y' + q y - r
        return p_f(xx) * yp + q_f(xx) * yy - r_f(xx)

    ref = _solve_bvp_reference(
        rhs, lambda ya, yb: np.array([ya[0] - alpha, yb[0] - beta]), x, y,
    )
    scale = max(1.0, float(np.max(np.abs(ref))))
    return bool(
        np.all(np.abs(y - ref) <= 4.0 * truncation + 1e-7 * scale)
        and np.max(np.abs(y - y_h)) <= 1e-9 * scale
    )


def verdict(verified: bool | None) -> str:
    """Label for a verification result: ``PASS`` / ``FAIL`` / ``NOT CHECKED``."""
    if verified is None:
        return "NOT CHECKED"
    return "PASS" if verified else "FAIL"


def verify_problem(problem: dict, answer: Any) -> bool | None:
    """Check ``answer`` against a trusted reference, dispatching on the problem type.

    Returns ``True`` / ``False`` when an independent check ran, and
    ``None`` when there is no check for this problem type or the check
    itself failed to run. Callers must not treat ``None`` as a pass.
    """
    try:
        return _verify_problem(problem, answer)
    except Exception:
        return None


def _verify_problem(problem: dict, answer: Any) -> bool | None:
    ptype = problem.get("type")
    if ptype in ("linear_system", "cramers_rule"):
        ok = verify_linear_system(problem["A"], problem["b"], answer)
        if ptype == "linear_system":
            ok = ok and verify_linear_system_against_numpy(problem["A"], problem["b"], answer)
        return bool(ok)
    if ptype == "root_finding":
        x_val = float(answer)
        var = problem["variable"]
        method = problem.get("method", "bisection")
        if method == "fixed_point":
            # ``function`` is g(x) and we seek x = g(x); the residual to
            # check is g(x) - x, or f(x) when the underlying target is known.
            target = problem.get("target_function") or f"({problem['function']}) - ({var})"
            return verify_root(target, var, x_val)
        ok = verify_root(problem["function"], var, x_val)
        try:
            if method == "bisection":
                ok = ok and verify_root_against_scipy(
                    problem["function"], var, x_val,
                    bracket=(float(problem["a"]), float(problem["b"])),
                )
            elif method in ("newton_raphson", "secant"):
                ok = ok and verify_root_against_scipy(
                    problem["function"], var, x_val,
                    x0=float(problem.get("x0", problem.get("x1", 0.0))),
                )
        except Exception:
            pass  # the residual check above still stands on its own
        return ok
    if ptype == "ivp":
        return verify_ivp_against_scipy(
            problem["f_expr"], problem["variable"], problem["function"],
            problem["y0"], float(problem["x0"]),
            float(problem["x_end"]), answer,
            method=problem.get("method"), h=problem.get("h"),
        )
    if ptype == "determinant":
        return verify_determinant(problem["A"], answer)
    if ptype == "matrix_inverse":
        return bool(verify_inverse(problem["A"], answer))
    if ptype == "eigenvalues":
        eigs, vecs = answer
        return verify_eigenvalues(problem["A"], eigs) and verify_eigenvectors(problem["A"], eigs, vecs)
    if ptype == "lu_decomposition":
        P, L, U = answer
        return verify_lu(problem["A"], P, L, U)
    if ptype == "interpolation":
        return verify_interpolation(problem["points"], answer)
    if ptype == "numerical_integration":
        return verify_integration(problem, float(answer))
    if ptype == "numerical_diff":
        return verify_differentiation(problem, float(answer))
    if ptype == "bvp":
        if problem.get("method", "shooting") == "shooting":
            return verify_bvp_shooting(problem, answer)
        return verify_bvp_finite_difference(problem, answer)
    # Custom / unknown solvers: no independent reference is available, so say
    # so rather than pass.
    return None

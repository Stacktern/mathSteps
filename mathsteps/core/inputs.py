"""Smart problem parsers: convert free-form user input into a problem dict.

The CLI exposes these via subcommands like::

    mathsteps linear-system --A "1 2; 3 4" --b "5 11"
    mathsteps root "x**3 - x - 2" --method newton --x0 1.5
    mathsteps ivp "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
    mathsteps ask

They also drive the ``ask`` interactive wizard.
"""
from __future__ import annotations

import re
from typing import Any

import sympy as sp


_METHOD_ALIASES = {
    "newton": "newton_raphson",
    "newton_dd": "newton_divided_differences",
    "divided_differences": "newton_divided_differences",
    "trapezoid": "trapezoidal",
    "simpsons": "simpson",
    "fd": "finite_difference",
}


def normalize_method(method: str | None) -> str | None:
    """Canonicalise a method name: case, ``-`` vs ``_``, and common aliases (``newton``)."""
    if method is None:
        return None
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    return _METHOD_ALIASES.get(key, key)


def parse_matrix(text: str) -> list[list[float]]:
    """Parse ``"1 2 3; 4 5 6; 7 8 9"`` → ``[[1,2,3],[4,5,6],[7,8,9]]``.

    Accepts ``;`` / ``,`` / newlines / whitespace as separators.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty matrix.")
    text = text.replace(",", " ")
    rows = re.split(r"[;\n]", text)
    matrix: list[list[float]] = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        cells = row.split()
        matrix.append([float(c) for c in cells])
    if not matrix:
        raise ValueError("Could not parse matrix.")
    widths = {len(r) for r in matrix}
    if len(widths) != 1:
        raise ValueError(f"Matrix rows have inconsistent widths: {widths}.")
    return matrix


def parse_vector(text: str) -> list[float]:
    """Parse ``"5 11"`` → ``[5.0, 11.0]``."""
    text = text.replace(",", " ").strip()
    if not text:
        raise ValueError("Empty vector.")
    return [float(t) for t in text.split()]


def parse_points(text: str) -> list[list[float]]:
    """Parse ``"(0,1); (1,2); (2,5)"`` or ``"0 1; 1 2; 2 5"`` → ``[[0,1],[1,2],[2,5]]``."""
    text = text.strip()
    if not text:
        raise ValueError("Empty points.")
    text = text.replace(",", " ").replace("(", "").replace(")", "")
    rows = re.split(r"[;\n]", text)
    pts: list[list[float]] = []
    for row in rows:
        cells = row.split()
        if len(cells) != 2:
            raise ValueError(f"Each point needs exactly 2 numbers; got {cells}.")
        pts.append([float(cells[0]), float(cells[1])])
    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")
    return pts


def parse_function(expr: str, var_str: str = "x") -> tuple[sp.Expr, sp.Symbol]:
    """Parse a SymPy-compatible expression."""
    var = sp.symbols(var_str)
    try:
        expr_sym = sp.sympify(expr, locals={var_str: var})
    except Exception as exc:
        raise ValueError(f"Could not parse {expr!r} as a SymPy expression: {exc}") from exc
    if not expr_sym.has(var) and var_str != "y":
        # Allow constant expressions but warn via metadata only.
        pass
    return expr_sym, var


def parse_ivp_equation(eq: str, var_str: str = "x", func_str: str = "y") -> str:
    """Parse ``"y' = -2*x*y"`` / ``"dy/dx = -2*x*y"`` / ``"-2*x*y"`` into the
    right-hand-side string ``"-2*x*y"``."""
    s = eq.strip()
    s = re.sub(r"^dy\s*/\s*d" + var_str + r"\s*=\s*", "", s)
    s = re.sub(r"^d" + func_str + r"\s*/\s*d" + var_str + r"\s*=\s*", "", s)
    s = re.sub(rf"^{func_str}'\s*=\s*", "", s)
    return s.strip()


def detect_and_build(text: str) -> dict[str, Any]:
    """Best-effort: guess the problem type from a free-form line and return a problem dict.

    Heuristics:
      * Contains ``=`` and looks like an algebraic equation (no derivatives) → root_finding.
      * Contains ``y'`` / ``dy/dx`` → ivp.
      * Otherwise: try root_finding (treat as f(x) = 0).
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty input.")

    if re.search(r"\b(y'|dy\s*/\s*dx|d" + r"\w*\s*/\s*d" + r"\w*)", s):
        rhs = parse_ivp_equation(s)
        return {
            "type": "ivp",
            "method": "rk4",
            "f_expr": rhs,
            "function": "y",
            "variable": "x",
            "y0": 1.0,
            "x0": 0.0,
            "x_end": 1.0,
            "h": 0.1,
        }

    if "=" in s and not s.startswith("=") and s.count("=") == 1:
        lhs, rhs = (t.strip() for t in s.split("=", 1))
        return {
            "type": "root_finding",
            "method": "newton_raphson",
            "function": f"({lhs}) - ({rhs})",
            "variable": "x",
            "x0": 0.5,
            "tol": 1e-10,
            "max_iter": 50,
        }

    return {
        "type": "root_finding",
        "method": "newton_raphson",
        "function": s,
        "variable": "x",
        "x0": 0.5,
        "tol": 1e-10,
        "max_iter": 50,
    }


def build_problem_from_inputs(
    domain: str,
    *,
    method: str | None = None,
    A: str | None = None,
    b: str | None = None,
    function: str | None = None,
    variable: str | None = None,
    a: str | None = None,
    c: str | None = None,
    x0: str | None = None,
    x1: str | None = None,
    x: str | None = None,
    y0: str | None = None,
    x_end: str | None = None,
    h: str | None = None,
    tol: str | None = None,
    max_iter: str | None = None,
    s0: str | None = None,
    s1: str | None = None,
    points: str | None = None,
    p_expr: str | None = None,
    q_expr: str | None = None,
    r_expr: str | None = None,
    alpha: str | None = None,
    beta: str | None = None,
    n: str | None = None,
    f_expr: str | None = None,
    target_function: str | None = None,
) -> dict[str, Any]:
    """Build a problem dict from CLI flag values, dispatching on ``domain``."""
    domain = domain.lower().replace("-", "_")
    method = normalize_method(method)
    if domain == "linear_system":
        if not A or not b:
            raise ValueError("linear-system requires --A and --b.")
        return {
            "type": "linear_system",
            "A": parse_matrix(A),
            "b": parse_vector(b),
        }
    if domain == "matrix_inverse":
        if not A:
            raise ValueError("matrix-inverse requires --A.")
        return {"type": "matrix_inverse", "A": parse_matrix(A)}
    if domain == "determinant":
        if not A:
            raise ValueError("determinant requires --A.")
        return {
            "type": "determinant",
            "method": method or "cofactor",
            "A": parse_matrix(A),
        }
    if domain == "lu":
        if not A:
            raise ValueError("lu requires --A.")
        return {"type": "lu_decomposition", "A": parse_matrix(A)}
    if domain == "eigen":
        if not A:
            raise ValueError("eigenvalues requires --A.")
        return {"type": "eigenvalues", "A": parse_matrix(A)}
    if domain == "cramer":
        if not A or not b:
            raise ValueError("cramers requires --A and --b.")
        return {
            "type": "cramers_rule",
            "A": parse_matrix(A),
            "b": parse_vector(b),
        }
    if domain == "root":
        if not function:
            raise ValueError("root requires --function (an expression in --variable).")
        var = variable or "x"
        problem: dict[str, Any] = {
            "type": "root_finding",
            "method": method or "newton_raphson",
            "function": function,
            "variable": var,
        }
        if target_function:
            problem["target_function"] = target_function
        if tol is not None:
            problem["tol"] = float(tol)
        if max_iter is not None:
            problem["max_iter"] = int(max_iter)
        if problem["method"] == "bisection":
            if a is None or c is None:
                raise ValueError("bisection requires --a and --c (bracket endpoints).")
            problem["a"] = float(a)
            problem["b"] = float(c)
        elif problem["method"] == "secant":
            if x0 is None or x1 is None:
                raise ValueError("secant requires --x0 and --x1.")
            problem["x0"] = float(x0)
            problem["x1"] = float(x1)
        else:
            if x0 is None:
                raise ValueError(f"{problem['method']} requires --x0.")
            problem["x0"] = float(x0)
        return problem
    if domain == "ivp":
        if not f_expr:
            raise ValueError("ivp requires --f-expr ('dy/dx = ...' or just RHS).")
        var = variable or "x"
        if ";" in f_expr:
            # A system: "v; -4*x" with --function "x v" and --y0 "1 0".
            rhs: Any = [e.strip() for e in f_expr.split(";") if e.strip()]
            names: Any = (
                [t for t in re.split(r"[,\s]+", function.strip()) if t]
                if function
                else [f"y{i + 1}" for i in range(len(rhs))]
            )
            if y0 is None:
                raise ValueError("A system needs --y0 with one value per equation, e.g. --y0 \"1 0\".")
            y0_value: Any = [float(t) for t in re.split(r"[,\s]+", y0.strip()) if t]
        else:
            names = (function or "y").strip()
            rhs = parse_ivp_equation(f_expr, var, names)
            y0_value = float(y0) if y0 is not None else 1.0
        problem = {
            "type": "ivp",
            "method": method or "rk4",
            "f_expr": rhs,
            "function": names,
            "variable": var,
            "y0": y0_value,
            "x0": float(x0) if x0 is not None else 0.0,
            "x_end": float(x_end) if x_end is not None else 1.0,
        }
        if method == "rk45":
            problem["atol"] = 1e-9
            problem["rtol"] = 1e-9
        else:
            if h is None:
                raise ValueError("Fixed-step IVP requires --h.")
            problem["h"] = float(h)
        return problem
    if domain == "bvp":
        problem = {
            "type": "bvp",
            "method": method or "shooting",
        }
        if method in (None, "shooting"):
            if not f_expr:
                raise ValueError("shooting requires --f_expr (the second derivative RHS).")
            var = variable or "x"
            problem.update({
                "f_expr": f_expr,
                "variable": var,
                "y_left": float(a) if a is not None else 0.0,
                "y_right": float(c) if c is not None else 0.0,
                "x0": float(x0) if x0 is not None else 0.0,
                "x_end": float(x_end) if x_end is not None else 1.0,
                "h": float(h) if h is not None else 0.01,
                "s0": float(s0) if s0 is not None else 0.0,
                "s1": float(s1) if s1 is not None else 1.0,
                "tol": float(tol) if tol is not None else 1e-6,
                "max_iter": int(max_iter) if max_iter is not None else 30,
            })
        elif method == "finite_difference":
            problem.update({
                "p_expr": p_expr or "0",
                "q_expr": q_expr or "1",
                "r_expr": r_expr or "0",
                "variable": variable or "x",
                # Interval is [x0, x_end] (same flags as shooting); boundary values alpha/beta.
                "a": float(x0) if x0 is not None else 0.0,
                "b": float(x_end) if x_end is not None else 1.0,
                "alpha": float(alpha) if alpha is not None else 0.0,
                "beta": float(beta) if beta is not None else 0.0,
                "n": int(n) if n is not None else 10,
            })
        return problem
    if domain == "interp":
        if not points:
            raise ValueError("interp requires --points.")
        return {
            "type": "interpolation",
            "method": method or "lagrange",
            "points": parse_points(points),
        }
    if domain == "integrate":
        if not function:
            raise ValueError("integrate requires --function.")
        var = variable or "x"
        return {
            "type": "numerical_integration",
            "function": function,
            "variable": var,
            "a": float(a) if a is not None else 0.0,
            "b": float(c) if c is not None else 1.0,
            "n": int(n) if n is not None else 100,
            "method": method or "trapezoidal",
        }
    if domain == "diff":
        if not function:
            raise ValueError("diff requires --function, --x, --h.")
        var = variable or "x"
        return {
            "type": "numerical_diff",
            "function": function,
            "variable": var,
            "x": float(x) if x is not None else 1.0,
            "h": float(h) if h is not None else 0.001,
            "method": method or "central",
        }
    raise ValueError(f"Unknown domain {domain!r}.")

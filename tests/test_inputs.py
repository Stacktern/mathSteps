"""Tests for the input parser and problem-builder."""
import pytest

from mathsteps.core.inputs import (
    build_problem_from_inputs,
    detect_and_build,
    parse_function,
    parse_ivp_equation,
    parse_matrix,
    parse_points,
    parse_vector,
)


def test_parse_matrix():
    assert parse_matrix("1 2; 3 4") == [[1.0, 2.0], [3.0, 4.0]]
    assert parse_matrix("1 2 3\n4 5 6\n7 8 9") == [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],
    ]
    assert parse_matrix("1,2;3,4") == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_matrix_inconsistent_widths():
    with pytest.raises(ValueError, match="inconsistent"):
        parse_matrix("1 2 3; 4 5")


def test_parse_vector():
    assert parse_vector("5 11") == [5.0, 11.0]
    assert parse_vector("5, 11") == [5.0, 11.0]


def test_parse_points():
    assert parse_points("(0,1); (1,2); (2,5)") == [[0.0, 1.0], [1.0, 2.0], [2.0, 5.0]]
    assert parse_points("0 1; 1 2; 2 5") == [[0.0, 1.0], [1.0, 2.0], [2.0, 5.0]]


def test_parse_ivp_equation():
    assert parse_ivp_equation("y' = -2*x*y") == "-2*x*y"
    assert parse_ivp_equation("dy/dx = sin(x) - y") == "sin(x) - y"
    assert parse_ivp_equation("-2*x*y") == "-2*x*y"


def test_parse_function_returns_sympy():
    import sympy as sp

    expr, var = parse_function("x**3 - x - 2")
    assert isinstance(expr, sp.Expr)
    assert var == sp.symbols("x")


def test_detect_root_finding_default():
    problem = detect_and_build("x**3 - x - 2")
    assert problem["type"] == "root_finding"
    assert problem["function"] == "x**3 - x - 2"
    assert problem["variable"] == "x"


def test_detect_root_finding_with_equals():
    problem = detect_and_build("x**2 - 4 = 0")
    assert problem["type"] == "root_finding"
    assert problem["function"] == "(x**2 - 4) - (0)"


def test_detect_ivp():
    problem = detect_and_build("y' = -2*x*y")
    assert problem["type"] == "ivp"
    assert problem["f_expr"] == "-2*x*y"
    assert problem["method"] == "rk4"


def test_build_linear_system():
    problem = build_problem_from_inputs(
        "linear_system", A="1 2; 3 4", b="5 11",
    )
    assert problem["type"] == "linear_system"
    assert problem["A"] == [[1.0, 2.0], [3.0, 4.0]]
    assert problem["b"] == [5.0, 11.0]


def test_build_root_newton():
    problem = build_problem_from_inputs(
        "root", function="x**3 - x - 2", method="newton_raphson", x0="1.5",
    )
    assert problem["type"] == "root_finding"
    assert problem["x0"] == 1.5


def test_build_root_bisection_requires_bracket():
    with pytest.raises(ValueError, match="bisection requires"):
        build_problem_from_inputs(
            "root", function="x**3 - x - 2", method="bisection",
        )


def test_build_ivp_rk4():
    problem = build_problem_from_inputs(
        "ivp", f_expr="y' = -2*x*y", y0="1", x_end="2", h="0.1", method="rk4",
    )
    assert problem["type"] == "ivp"
    assert problem["f_expr"] == "-2*x*y"
    assert problem["y0"] == 1.0


def test_build_bvp_shooting():
    problem = build_problem_from_inputs(
        "bvp", f_expr="sin(x) - y", a="0", c="0", x_end="1", s0="0", s1="1",
    )
    assert problem["type"] == "bvp"
    assert problem["method"] == "shooting"
    assert problem["y_left"] == 0.0


def test_build_interp_lagrange():
    problem = build_problem_from_inputs(
        "interp", points="(0,1); (1,2); (2,5)", method="lagrange",
    )
    assert problem["type"] == "interpolation"
    assert problem["points"] == [[0.0, 1.0], [1.0, 2.0], [2.0, 5.0]]


def test_build_unknown_domain():
    with pytest.raises(ValueError, match="Unknown domain"):
        build_problem_from_inputs("nonexistent")


def test_build_linear_system_missing_args():
    with pytest.raises(ValueError, match="--A"):
        build_problem_from_inputs("linear_system", A="1 2")

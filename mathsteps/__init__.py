"""MathSteps: step-by-step math solver.

Quick start
-----------

As a library::

    import mathsteps

    r = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
    print(r.answer)            # [1. 2.]   (a numpy array; r.exact has exact rationals)

    r = mathsteps.root("cos(x) - x", method="newton", x0=0)
    print(r.answer, r.verified)

    r = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")
    print(r.answer)

Every function returns a :class:`mathsteps.Result` with:

    r.solver    # solver name (e.g. "newton_raphson")
    r.steps     # list[Step], each with .description / .before / .after / .data
    r.answer    # the final answer as plain Python / NumPy values (float, ndarray, ...)
    r.exact     # exact SymPy form where one exists (rationals, matrices), else None
    r.details   # extra outputs, e.g. the x / y grid of an ODE or BVP solution
    r.verified  # True/False vs NumPy/SciPy/SymPy ground truth; None if not checked
    r.converged # iterative solvers: False if max_iter ran out (also warns)

As a CLI::

    mathsteps linear-system --A "1 2; 3 4" --b "5 11"
    mathsteps root --function "x**3 - x - 2" --method bisection --a 1 --c 2
    mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1
    mathsteps ask
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import (
    Result,
    available_solvers,
    bvp_finite_difference,
    bvp_shooting,
    cramers_rule,
    determinant,
    differentiate,
    eigenvalues,
    integrate,
    inverse,
    ivp,
    ivp_second_order,
    lagrange,
    linear_system,
    lu,
    newton_divided_differences,
    root,
    solve_problem,
)
from .core.answers import InterpolatingPolynomial
from .core.convergence import ConvergenceWarning
from .core.step import Step

# Importing the solver modules registers them with the global registry.
# Listing them here ensures ``import mathsteps`` is enough — no manual
# ``import mathsteps.domains.linalg.gaussian_elimination`` needed.
import mathsteps.domains.linalg.cofactor_determinant  # noqa: F401
import mathsteps.domains.linalg.cramers_rule  # noqa: F401
import mathsteps.domains.linalg.eigenvalues  # noqa: F401
import mathsteps.domains.linalg.gauss_jordan_inverse  # noqa: F401
import mathsteps.domains.linalg.gaussian_elimination  # noqa: F401
import mathsteps.domains.linalg.lu_decomposition  # noqa: F401
import mathsteps.domains.numerical.bisection  # noqa: F401
import mathsteps.domains.numerical.fixed_point  # noqa: F401
import mathsteps.domains.numerical.lagrange  # noqa: F401
import mathsteps.domains.numerical.newton_divided_differences  # noqa: F401
import mathsteps.domains.numerical.newton_raphson  # noqa: F401
import mathsteps.domains.numerical.numerical_diff  # noqa: F401
import mathsteps.domains.numerical.numerical_integration  # noqa: F401
import mathsteps.domains.numerical.secant  # noqa: F401
import mathsteps.domains.ode_bvp.finite_difference  # noqa: F401
import mathsteps.domains.ode_bvp.shooting  # noqa: F401
import mathsteps.domains.ode_ivp.euler  # noqa: F401
import mathsteps.domains.ode_ivp.heun  # noqa: F401
import mathsteps.domains.ode_ivp.midpoint  # noqa: F401
import mathsteps.domains.ode_ivp.rk4  # noqa: F401
import mathsteps.domains.ode_ivp.rk45  # noqa: F401

__all__ = [
    "ConvergenceWarning",
    "InterpolatingPolynomial",
    "Result",
    "Step",
    "__version__",
    "available_solvers",
    "bvp_finite_difference",
    "bvp_shooting",
    "cramers_rule",
    "determinant",
    "differentiate",
    "eigenvalues",
    "integrate",
    "inverse",
    "ivp",
    "ivp_second_order",
    "lagrange",
    "linear_system",
    "lu",
    "newton_divided_differences",
    "root",
    "solve_problem",
]

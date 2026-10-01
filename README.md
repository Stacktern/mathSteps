# mathsteps

A **dual-interface** step-by-step symbolic/numerical math solver — use as a **CLI** or
**Python library**.

`mathsteps` takes a math problem and prints every intermediate step
along the way to the final answer — like Symbolab or Mathway, but
local, scripted, and open-source.

> **Libraries compute; you explain.** NumPy / SciPy / SymPy give you
> correct, fast final answers. They don't naturally expose *how* they
> got there. `mathsteps` is the step-tracking / explanation layer on
> top.

## Installation

```bash
pip install mathsteps
```

This installs:
- ✅ `mathsteps` CLI command (ready to use immediately)
- ✅ `mathsteps` Python package (importable in scripts/notebooks)
- ✅ All dependencies pre-configured

For development (editable install from source):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Quick Start

### As a Python Library (Recommended for Scripts & Apps)

```python
import mathsteps

# Solve Ax = b (system of 2 equations, 2 unknowns)
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
print(result.answer)    # [1. 2.]        a numpy array
print(result.exact)     # [1, 2]         the exact SymPy rationals
print(result.verified)  # True           checked against numpy.linalg.solve
print(result.solver)    # gaussian_elimination

# Print step-by-step solution
for i, step in enumerate(result.steps, 1):
    print(f"Step {i}: {step.description}")
    if step.before:
        print(f"  Before: {step.before}")
    if step.after:
        print(f"  After:  {step.after}")

# Root finding: find x where cos(x) = x
result = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
print(result.answer)    # 0.7390851332151607 (a float)

# Solve ODE: dy/dx = -2xy with initial condition
result = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")
print(result.answer)    # y(2) = 0.01832...
t, y = result.details["x"], result.details["y"]   # the whole trajectory

# Interpolation returns a callable polynomial
p = mathsteps.lagrange([[0, 1], [1, 2], [2, 5]]).answer
print(p(3))             # 10.0
print(p.expr)           # x**2 + 1   (SymPy form)

# Matrices
result = mathsteps.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
print(result.answer)    # -306.0   (result.exact == -306)

values, vectors = mathsteps.eigenvalues([[4, 2], [2, 3]]).answer   # numpy.linalg.eig layout
```

### Systems, second-order equations and BVP profiles

```python
# A system of ODEs (predator-prey), forwards ...
r = mathsteps.ivp(["1.1*u - 0.4*u*w", "0.1*u*w - 0.4*w"],
                  variable="t", function=["u", "w"], y0=[10, 5], x_end=20, h=0.005)
r.answer              # [u(20), w(20)]
r.details["y"]        # shape (points, 2): the whole solution

# ... a second-order equation (damped spring  x'' = -4x - 0.5 x')
r = mathsteps.ivp_second_order("-4*x - 0.5*xp", variable="t", function="x",
                               y0=1, dy0=0, x_end=10, h=0.005)

# ... and backwards in x (x_end < x0)
mathsteps.ivp("y", y0=2.718281828, x0=1, x_end=0, h=0.01)    # y(0) = 1

# Boundary value problems return the whole solution, not just y(x_end)
r = mathsteps.bvp_shooting("0.01*(y-20)", y_left=100, y_right=50, x_end=10)
r.answer              # T(x) on the grid r.details["x"]
r.details["slope"]    # the converged T'(0)
```

### Result Structure

Every function returns a **`mathsteps.Result`** object:

```python
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])

result.solver      # str: solver name ("gaussian_elimination")
result.answer      # the answer as plain Python / NumPy: float, ndarray, tuple of arrays, ...
result.exact       # the exact SymPy form where one exists (rationals, matrices), else None
result.details     # extra outputs: x / y grids of ODE and BVP solutions, shooting slope, ...
result.steps       # list[Step]: step-by-step breakdown
result.verified    # True / False vs NumPy/SciPy, or None if not checked
result.converged   # iterative solvers: False if max_iter ran out; else None/True
result.problem     # dict: the original problem dict

# Inspect each step
for step in result.steps:
    step.description  # str: what happened (e.g. "Swap rows")
    step.before       # str: state before (e.g. matrix as string)
    step.after        # str: state after
    step.data         # dict: structured numerical data
```

### Full library API

| Function | `result.answer` | Problem |
|---|---|---|
| `linear_system(A, b)` | `ndarray` `x` (`exact`: rationals) | `Ax = b` |
| `determinant(A)` | `float` (`exact`: exact value) | determinant |
| `inverse(A)` | `ndarray` (`exact`: `sympy.Matrix`) | matrix inverse |
| `lu(A)` | tuple of arrays `(P, L, U)`, `P @ A = L @ U` | LU decomposition |
| `eigenvalues(A)` | `(values, vectors)` like `numpy.linalg.eig`, one column per independent eigenvector | eigenvalues / eigenvectors |
| `cramers_rule(A, b)` | `ndarray` | Cramer's rule |
| `root(f, …)` | `float` | find `f(x)=0` |
| `ivp(f_expr, …)` | `y(x_end)`: `float`, or `ndarray` for a system; `details`: `x`, `y` | `dy/dx = f(x,y)`, systems, backwards |
| `ivp_second_order(f, …)` | `[y(x_end), y'(x_end)]`; `details`: `x`, `y` | `y'' = f(x, y, y')` |
| `bvp_shooting(f, …)` | `y` on the grid `details["x"]`; `details["slope"]` | two-point BVP |
| `bvp_finite_difference(…)` | `y` on the grid `details["x"]` | linear BVP via finite differences |
| `lagrange(points)` | callable polynomial (`p(x)`, `p.expr`, `p.coefficients`) | polynomial interpolation |
| `newton_divided_differences(points)` | callable polynomial | Newton-form interpolation |
| `integrate(f, a, b, …)` | `float` | numerical integration |
| `differentiate(f, x, …)` | `float` | numerical derivative |
| `solve_problem(dict)` | `Result` | generic (raw problem dict) |
| `available_solvers()` | `list[str]` | every registered solver |

### As a Command-Line Tool (Recommended for Quick Exploration)

### 1. Direct subcommands (easiest — flags prompt for everything)

```powershell
mathsteps linear-system --A "1 2; 3 4" --b "5 11"
mathsteps root          --function "x**3 - x - 2" --method newton --x0 1.5
mathsteps root          --function "x**3 - x - 2" --method bisection --a 1 --c 2
mathsteps root          --function "x**3 - x - 2" --method secant   --x0 1 --x1 2
mathsteps ivp           --f-expr "-2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
mathsteps bvp           --f-expr "sin(x) - y" --a 0 --c 0 --x-end 1 --s0 0 --s1 1
mathsteps determinant   --A "6 1 1; 4 -2 5; 2 8 7"
mathsteps inverse       --A "1 2 3; 0 1 4; 5 6 0"
mathsteps lu            --A "2 1 1; 4 -6 0; -2 7 2"
mathsteps eigen         --A "2 0 0; 0 3 4; 0 4 9"
mathsteps cramer        --A "2 1 -1; -3 -1 2; -2 1 2" --b "8 -11 -3"
mathsteps interp        --points "(0,1); (1,2); (2,5)"
mathsteps integrate     --function "sin(x)" --a 0 --c 3.14159 --n 100 --method simpson
mathsteps diff          --function "sin(x)" --x 1 --h 0.001
```

Add `--pretty` to any command for Rich tables + LaTeX-style math.

### 2. Free-form equation (auto-detects type)

```powershell
mathsteps solve "x**3 - x - 2"
mathsteps solve "x**2 - 4 = 0"
mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
```

The detector:
- ``y'`` / ``dy/dx`` / ``d?/d?`` ⇒ IVP
- exactly one ``=`` with no derivatives ⇒ root-finding (treats ``lhs - rhs`` as f(x))
- otherwise ⇒ root-finding on the whole expression

### 3. JSON file (power-user / scripted)

```powershell
mathsteps solve examples\linear_system_3x3.json
mathsteps verify examples\linear_system_3x3.json   # exit 0 on PASS, 1 on FAIL
mathsteps list-solvers                              # show every solver
```

JSON files live under `examples/` and have a strict per-solver schema
(see `mathsteps/domains/<domain>/<solver>.py` for the exact keys).

### 4. Interactive wizard

```powershell
mathsteps ask
```

Walks you through a numbered menu, prompting for each input.

## Supported problem types

### Linear algebra

| Type | Solver | Notes |
|---|---|---|
| `linear_system` | Gaussian elimination | partial pivoting + back-substitution |
| `matrix_inverse` | Gauss-Jordan | reduced row echelon form |
| `determinant` | Cofactor expansion | |
| `lu_decomposition` | Doolittle | partial pivoting |
| `eigenvalues` | Characteristic polynomial | small matrices |
| `cramers_rule` | Cramer's rule | |

### Numerical analysis

| Type | Solver |
|---|---|
| `root_finding` (method `bisection`) | Bisection method |
| `root_finding` (method `newton_raphson`) | Newton-Raphson |
| `root_finding` (method `secant`) | Secant method |
| `root_finding` (method `fixed_point`) | Fixed-point iteration x = g(x) |
| `numerical_diff` | Finite differences (forward / backward / central) |
| `numerical_integration` | Trapezoidal or Simpson's rule |
| `interpolation` (method `lagrange`) | Lagrange interpolation |
| `interpolation` (method `newton_divided_differences`) | Newton form |

### IVP — initial value problems

| Type | Solver |
|---|---|
| `ivp` (method `euler`) | Forward Euler |
| `ivp` (method `heun`) | Improved Euler / Heun |
| `ivp` (method `midpoint`) | Midpoint method (RK2) |
| `ivp` (method `rk4`) | Classical 4th-order Runge-Kutta |
| `ivp` (method `rk45`) | Adaptive Dormand-Prince RK45 |

### BVP — boundary value problems

| Type | Solver |
|---|---|
| `bvp` (method `shooting`) | Shooting method (uses RK4 + secant internally) |
| `bvp` (method `finite_difference`) | Finite-difference discretisation + linear algebra |

See `examples/` for the exact JSON schema of each solver.

## Documentation

* [docs/python-guide.md](docs/python-guide.md) — using `mathsteps` as a library
* [docs/features.md](docs/features.md) — every problem type and its parameters
* [docs/setup.md](docs/setup.md) — installation and troubleshooting
* [examples_python/](examples_python/) — runnable scripts
* [CONTRIBUTING.md](CONTRIBUTING.md) — adding a solver

## Verification and convergence

Answers are checked against a trusted reference (NumPy / SciPy / SymPy
ground truth) by `mathsteps.verify.verify_problem`, the single source of
truth shared by the Python API and the CLI. The result is three-valued:

| `result.verified` | Meaning |
|---|---|
| `True`  | Checked against an independent reference and it matches. |
| `False` | Checked and it does **not** match. |
| `None`  | Not checked: you passed `verify=False`, or the check could not run. Never a pass. |

Every problem type has an independent check: linear algebra against NumPy,
roots against SciPy, ODEs (scalar and systems) against `solve_ivp`, integrals
against `scipy.integrate.quad`, derivatives against the exact SymPy
derivative, BVPs against `scipy.integrate.solve_bvp`, interpolants through
their data points. Numerical methods are *supposed* to differ from the true
answer by their truncation error, so each tolerance is derived from that
error (Richardson extrapolation): a correct Euler run passes and a wrong RK4
answer does not.

The CLI prints `Verification: PASS`, `FAIL` or `NOT CHECKED`, and
`mathsteps verify file.json` exits `0` / `1` / `2` respectively.

Iterative solvers (Newton, secant, bisection, fixed-point, shooting)
set `result.converged`. If `max_iter` runs out before `tol` is met it is
`False`, a `mathsteps.ConvergenceWarning` is emitted, and the last step
says so; the CLI prints a warning on stderr. Raise `max_iter` (slowly
converging problems like `cos(x)` fixed-point need more than the default
50) or loosen `tol`.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -v
```

## Project layout

```
mathsteps/
├── core/
│   ├── step.py            # Step dataclass
│   ├── solver_base.py     # abstract Solver interface
│   ├── registry.py        # maps problem type -> solver
│   └── render.py          # pretty LaTeX / Rich output
├── domains/
│   ├── linalg/            # 6 solvers
│   ├── numerical/         # 8 solvers
│   ├── ode_ivp/           # 5 solvers
│   └── ode_bvp/           # 2 solvers
├── cli.py                 # mathsteps console script
└── verify.py              # ground-truth verification
examples/                  # one JSON file per supported problem type
tests/                     # pytest suite
```

## License

MIT — see `LICENSE`.

## Large problems: the `detail` setting

A 40x40 elimination or an ODE with 30 000 steps would produce a flood of
steps nobody reads. Solvers that can flood take `detail=`:

| `detail` | Records |
|---|---|
| `"full"` | every step (the default for small problems) |
| `"summary"` | the first and last few steps plus a note, or one step per pivot column; the default for large problems (linear systems / inverses / LU above 10x10, ODEs above 200 steps) |
| `"none"` | only the introduction and the final step |

The answer, `details` and `verified` are identical in every mode:

```python
mathsteps.linear_system(A, b)                  # 40x40: ~40 steps, not ~800
mathsteps.linear_system(A, b, detail="full")   # every row operation
mathsteps.ivp("-0.5*y", y0=1, x_end=20, h=0.001, detail="none")
```

The CLI has the same flag (`--detail full|summary|none`) on `linear-system`,
`inverse`, `lu` and `ivp`.

## What it does not do

* ODE systems and second-order IVPs are solved as first-order systems; boundary
  value problems are scalar and second-order (shooting, or linear
  finite differences).
* Eigenvalues are exact when the characteristic polynomial factors into pieces
  of degree <= 2 over the rationals (integers, fractions, `a ± sqrt(b)`) and
  15-digit numerical roots otherwise.
* `integrate` samples the interval's endpoints, so an integrand that is
  infinite or undefined there (`sin(x)/x` at 0) raises an error instead of
  returning garbage; start at a small `a` instead.

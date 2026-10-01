# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release.

### Library

- One function per problem type, each returning a `Result` with `answer`,
  `exact`, `details`, `steps`, `verified` and `converged`.
- Linear algebra: Gaussian elimination, Gauss-Jordan inverse, cofactor
  determinant, LU with partial pivoting, Cramer's rule, eigenvalues and
  eigenvectors (exact when the characteristic polynomial factors into pieces of
  degree <= 2, 15-digit numerical roots otherwise; full eigenspaces).
- Root finding: bisection, Newton-Raphson, secant, fixed-point.
- Interpolation: Lagrange and Newton divided differences, returned as callable
  polynomials.
- Numerical integration (trapezoidal, Simpson) and differentiation (forward,
  backward, central).
- Initial value problems: Euler, Heun, midpoint, RK4 and adaptive RK45, for one
  equation or a system, forwards or backwards, plus second-order equations.
- Boundary value problems: shooting and linear finite differences, returning the
  whole solution.
- Answers are plain `float` / NumPy values; the exact SymPy form is in
  `Result.exact`.
- `detail="full" | "summary" | "none"` controls how many steps are recorded;
  large problems are summarised by default.

### Verification

- `Result.verified` is `True` / `False` / `None` (not checked). Every problem
  type is checked against an independent reference (NumPy, SciPy, SymPy), with
  tolerances derived from each method's truncation error.
- Iterative solvers report `Result.converged` and emit a `ConvergenceWarning`
  when `max_iter` is exhausted.

### Command line

- `mathsteps` with one subcommand per problem type, free-form `solve`, JSON
  problem files, `verify` (exit code 0 / 1 / 2), and an interactive `ask`
  wizard. `--version`, `--detail` and `--pretty` flags.

# MathSteps — Algorithms to Include

A prioritized list of algorithms for the step-by-step solver, organized by domain and build phase. Each entry notes why it's included, its rough difficulty to implement step-by-step from scratch, and what it composes with.

Priority key: **P0** = build first (core, proves the architecture) · **P1** = build next (rounds out the domain) · **P2** = stretch goal / later

---

## 1. Linear Algebra

| Algorithm | Priority | Why include it | Step-by-step difficulty |
|---|---|---|---|
| Gaussian elimination (with partial pivoting) | P0 | Reference implementation for the whole project; cleanest possible "step = rewrite" mapping | Low |
| Back-substitution | P0 | Required companion to Gaussian elimination to actually produce `x` | Low |
| Gauss-Jordan elimination (reduced row echelon form) | P1 | Natural extension once Gaussian elimination exists; also gives you matrix inverse for free | Low |
| Matrix inverse via row reduction | P1 | Common student use case; reuses Gauss-Jordan steps | Low |
| Determinant via cofactor expansion | P1 | More "showable" step-by-step than LU-based determinant for small matrices | Low-Medium |
| LU decomposition | P1 | Natural byproduct of Gaussian elimination; useful for solving multiple systems with the same `A` | Medium |
| Eigenvalues/eigenvectors via characteristic polynomial (small matrices, 2x2/3x3) | P2 | Pedagogically the "showable" route; doesn't generalize to large matrices (that needs iterative methods, which are harder to narrate) | Medium |
| Cramer's Rule | P2 | Classic textbook method, nice contrast case to Gaussian elimination for small systems | Low |

---

## 2. Numerical Analysis (Root Finding, Interpolation, Numerical Calculus)

| Algorithm | Priority | Why include it | Step-by-step difficulty |
|---|---|---|---|
| Bisection method | P0 | Simplest possible root-finder; each iteration is trivially one step | Low |
| Newton-Raphson method | P0 | Second domain to prove the `Solver` interface generalizes (per build order); classic, widely taught | Low |
| Secant method | P1 | Natural variant of Newton-Raphson (no derivative needed); cheap to add once Newton-Raphson exists | Low |
| Fixed-point iteration | P1 | Conceptually simple, good contrast to bisection/Newton | Low |
| Lagrange interpolation | P1 | Classic, each term of the polynomial is a natural step | Medium |
| Newton's divided differences interpolation | P2 | Alternative to Lagrange, more efficient for adding points; more complex to narrate step-by-step | Medium |
| Trapezoidal rule (numerical integration) | P1 | Simple, each sub-interval is a step | Low |
| Simpson's rule (numerical integration) | P1 | Natural pair with trapezoidal rule | Low-Medium |
| Numerical differentiation (finite differences: forward/backward/central) | P1 | Foundational, also used internally by other methods | Low |

---

## 3. IVP — Initial Value Problems (ODEs)

| Algorithm | Priority | Why include it | Step-by-step difficulty |
|---|---|---|---|
| Euler's method | P0 | Simplest ODE solver; each time-step is one step, easiest to get end-to-end | Low |
| Improved Euler / Heun's method | P1 | Natural next step in accuracy, good contrast case | Low-Medium |
| RK4 (4th-order Runge-Kutta) | P1 | Industry-standard method; each stage (k1-k4) can be shown as a sub-step | Medium |
| Midpoint method | P2 | Another RK variant, cheap to add once RK4 exists | Low-Medium |
| Adaptive step-size RK (e.g. RK45 / Dormand-Prince) | P2 | More advanced; step-size adjustment itself becomes an extra "step" to explain — good stretch goal | High |

---

## 4. BVP — Boundary Value Problems

| Algorithm | Priority | Why include it | Step-by-step difficulty |
|---|---|---|---|
| Shooting method (using Euler or RK4 internally + secant/Newton for the boundary condition) | P0 | Best proof of composability — literally reuses your IVP solver and root-finding solver | Medium-High |
| Finite difference method (turns BVP into a linear system) | P1 | Reuses your linear-algebra (Gaussian elimination) solver directly | Medium |
| Collocation method | P2 | More advanced, only worth adding once the above two are solid | High |

---

## 5. Explicitly Out of Scope (for now)

- **General symbolic calculus** (arbitrary indefinite integrals, arbitrary derivatives via chain/product/quotient rule combinations) — this is `sympy.integrals.manualintegrate` territory; a multi-month effort on its own. Revisit only after the four domains above are solid, and treat it as a fifth, separate domain rather than folding it in early.
- **Large-matrix iterative eigenvalue methods** (QR algorithm, power iteration) — valuable eventually, but harder to narrate as clean discrete "steps" than the characteristic-polynomial route; a P3/future addition.
- **PDE solvers** — a different problem class entirely (different discretization, different step semantics); not part of this project's initial scope.

---

## 6. Suggested Implementation Order (combining domains)

1. Gaussian elimination + back-substitution (P0, linear algebra)
2. Bisection + Newton-Raphson (P0, numerical analysis) — proves the `Solver` interface generalizes
3. Euler's method (P0, IVP)
4. Shooting method using Euler + Newton-Raphson (P0, BVP) — proves composability
5. Round out each domain with its P1 algorithms
6. RK4 replaces/supplements Euler for better accuracy in IVP and shooting method
7. P2 items as stretch goals once the core four domains are stable and tested

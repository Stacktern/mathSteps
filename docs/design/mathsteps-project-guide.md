# MathSteps — Step-by-Step Symbolic/Numerical Solver: Project Guide

A CLI-first Python tool that takes a math problem (linear algebra, numerical analysis, IVP, BVP) and returns **step-by-step reasoning plus the final answer** — not just the answer.

---

## 1. Guiding Principles

1. **Libraries compute; you explain.** NumPy/SciPy/SymPy give you correct, fast final answers. They don't naturally expose *how* they got there. The actual project is the step-tracking/explanation layer on top.
2. **One domain fully working beats four domains half-working.** Ship linear algebra end-to-end first. Prove the architecture generalizes with a second domain before touching the rest.
3. **Always verify.** Every step-by-step answer must be checked against a trusted library (NumPy/SciPy/SymPy ground truth) before being shown as correct.
4. **Design for composability from day one.** BVP solvers should be able to reuse your IVP solver and your linear-algebra solver internally — this is also the best proof your architecture works.

---

## 2. Tech Stack

**Core (all you need for v1):**
| Tool | Purpose |
|---|---|
| Python 3.10+ | language |
| NumPy | matrix ops, linear algebra, numerical arrays |
| SciPy | `optimize`, `integrate` (solve_ivp, solve_bvp), `linalg` — used for *verification*, not step generation |
| SymPy | symbolic manipulation where exactness matters (fractions, symbolic derivatives if extended later) |
| Typer (or Click) | CLI framework — `mathsteps solve problem.json` with free `--help` |
| Rich | optional — colored/tabular terminal output for readable steps |
| pytest | testing, especially the verification layer |

**Packaging:**
- `pyproject.toml` with a console-script entry point → one codebase gives both a `mathsteps` CLI command and an importable `import mathsteps; mathsteps.solve(...)` function.

**No backend needed for v1.** No FastAPI/Flask, no database, no async. Everything runs locally: input → Python functions → step objects → printed to terminal.

A backend only becomes relevant later if you want:
- A web UI (Symbolab-style, steps rendered as HTML/LaTeX)
- A hosted API so other apps can call the solver over HTTP
- User accounts / saved problem history / rate limiting

If/when that happens, you'd wrap the *same* `Solver` classes in a thin FastAPI layer — no rearchitecting, since the core logic is already decoupled from the CLI.

---

## 3. What Problems You'll Be Able to Solve

Depends on which `Solver` modules exist, but here's the realistic map:

### Linear algebra (easiest to do well)
- Solving linear systems `Ax = b` (Gaussian elimination, LU decomposition)
- Matrix inverse via row reduction
- Determinants (cofactor expansion or elimination-based)
- Eigenvalues/eigenvectors for small matrices (characteristic-polynomial route is more "showable" than iterative methods)

### Numerical analysis (medium)
- Root finding: bisection, Newton-Raphson, secant method — each iteration is naturally a step
- Interpolation: Lagrange, Newton's divided differences
- Numerical differentiation/integration: finite differences, trapezoidal/Simpson's rule
- Nonlinear equation solving when symbolic solving fails

### IVP — initial value problems (medium-hard)
- Euler's method, improved Euler (Heun's), RK4 — each step = one time-step iteration
- Harder to make pedagogically rich than algebra: the "reasoning" per step is mechanical (plug into formula) rather than a rewrite

### BVP — boundary value problems (hardest)
- Shooting method: turns BVP into repeated IVP solves + root-finding on the boundary condition (composes your IVP + root-finding solvers)
- Finite difference method: turns BVP into a linear system (composes your linear-algebra solver)

### Out of scope for now
General symbolic calculus with full step explanations (arbitrary integrals/derivatives) — this is SymPy's `manualintegrate` territory and is its own multi-month effort. Worth studying `sympy/integrals/manualintegrate.py` later as a reference implementation, but don't fold it into v1 scope.

---

## 4. Project Architecture

```
mathsteps/
├── core/
│   ├── step.py          # Step dataclass, shared by every domain
│   ├── solver_base.py   # abstract Solver interface
│   └── registry.py      # maps problem type -> solver class
├── domains/
│   ├── linalg/
│   │   └── gaussian_elimination.py
│   ├── numerical/
│   │   └── newton_raphson.py
│   ├── ode_ivp/
│   │   └── rk4.py
│   └── ode_bvp/
│       └── shooting_method.py
├── cli.py               # entry point (typer)
└── verify.py            # ground-truth checks (numpy/scipy)
```

Every domain implements the same `Solver` interface and returns `list[Step]` + a final answer. That uniformity is what makes it easy to open-source later — a contributor adds one file under `domains/`, registers it, done.

### Core building blocks

```python
# core/step.py
from dataclasses import dataclass, field

@dataclass
class Step:
    description: str          # human-readable explanation
    before: str                # expression/state before, as string or LaTeX
    after: str                 # expression/state after
    data: dict = field(default_factory=dict)  # e.g. matrix snapshot, residual, etc.
```

```python
# core/solver_base.py
from abc import ABC, abstractmethod

class Solver(ABC):
    name: str

    @abstractmethod
    def can_solve(self, problem) -> bool: ...

    @abstractmethod
    def solve(self, problem) -> tuple[list[Step], object]:
        """Returns (steps, final_answer)."""
```

```python
# core/registry.py
_SOLVERS: list[Solver] = []

def register(solver: Solver):
    _SOLVERS.append(solver)

def find_solver(problem):
    for s in _SOLVERS:
        if s.can_solve(problem):
            return s
    raise ValueError("No solver found for this problem type")
```

### First concrete solver: Gaussian elimination

```python
# domains/linalg/gaussian_elimination.py
import numpy as np
from core.step import Step
from core.solver_base import Solver

class GaussianEliminationSolver(Solver):
    name = "gaussian_elimination"

    def can_solve(self, problem):
        return problem.get("type") == "linear_system"

    def solve(self, problem):
        A = np.array(problem["A"], dtype=float)
        b = np.array(problem["b"], dtype=float)
        n = len(b)
        aug = np.hstack([A, b.reshape(-1, 1)])
        steps = []

        for col in range(n):
            pivot_row = np.argmax(abs(aug[col:, col])) + col
            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
                steps.append(Step(
                    description=f"Swap row {col} and row {pivot_row} for numerical stability (partial pivoting)",
                    before=str(aug), after=str(aug)
                ))
            for r in range(col + 1, n):
                factor = aug[r, col] / aug[col, col]
                before = aug.copy()
                aug[r] -= factor * aug[col]
                steps.append(Step(
                    description=f"R{r} = R{r} - ({factor:.4g}) * R{col}",
                    before=str(before), after=str(aug)
                ))

        x = np.zeros(n)
        for i in reversed(range(n)):
            x[i] = (aug[i, -1] - aug[i, i+1:n] @ x[i+1:n]) / aug[i, i]
            steps.append(Step(
                description=f"Back-substitute to solve for x{i}",
                before="", after=f"x{i} = {x[i]:.6g}"
            ))

        return steps, x
```

### Verification layer

```python
# verify.py
import numpy as np

def verify_linear_system(A, b, x, tol=1e-8):
    return np.allclose(A @ x, b, atol=tol)
```

For IVP/BVP: compare against `scipy.integrate.solve_ivp` / `solve_bvp`.
For numerical root-finding: compare against `scipy.optimize`.

### CLI entry point

```python
# cli.py
import typer, json
from core.registry import find_solver
import domains.linalg.gaussian_elimination  # triggers registration

app = typer.Typer()

@app.command()
def solve(problem_file: str):
    problem = json.load(open(problem_file))
    solver = find_solver(problem)
    steps, answer = solver.solve(problem)
    for i, s in enumerate(steps, 1):
        typer.echo(f"Step {i}: {s.description}")
        typer.echo(f"  {s.before} -> {s.after}\n")
    typer.echo(f"Final answer: {answer}")

if __name__ == "__main__":
    app()
```

Package with a `pyproject.toml` entry point (`mathsteps = "mathsteps.cli:app"`), so `pip install -e .` gives you a `mathsteps` command immediately.

---

## 5. Suggested Build Order

1. **Linear algebra first.** Gaussian elimination end-to-end: input → steps → answer → verified against `numpy.linalg.solve`.
2. **Second domain to test the architecture.** Add Newton-Raphson root finding — this is the real test of whether the `Solver` interface generalizes, not the first domain.
3. **IVP (RK4).** Each step is an iteration snapshot rather than an algebraic rewrite, so `Step.data` (for storing intermediate arrays) matters more here than `before`/`after` strings.
4. **BVP (shooting method or finite differences).** Best proof the composable design works, since it literally reuses the IVP and linear-algebra solvers.
5. **Polish.** LaTeX/pretty output (Rich tables, `sympy.latex()` for symbolic pieces), then packaging and open-source docs (README, CONTRIBUTING.md, a "how to add a new domain" guide).

---

## 6. Open-Sourcing Checklist (for later)

- [ ] `README.md` — install instructions, quick example, supported problem types
- [ ] `CONTRIBUTING.md` — explain the `Solver` interface and how to add a new domain
- [ ] `pyproject.toml` — proper packaging metadata, console-script entry point
- [ ] Tests for every solver + the verification layer (pytest)
- [ ] Example problems in `examples/` (one JSON file per supported problem type)
- [ ] License (MIT/Apache-2.0 are the common defaults for this kind of tool)

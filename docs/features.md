# MathSteps - Complete Feature Overview

## Quick Facts

- **Dual Interface**: CLI + Python Library
- **Installation**: `pip install mathsteps`
- **Python Version**: 3.10+
- **Status**: Beta (0.1.0)
- **License**: MIT
- **Verification**: Every answer is verified against NumPy/SciPy/SymPy

## Supported Problem Types

### 📊 Linear Algebra (LU, QR, Eigenvalues, etc.)

| Problem | Solver | Input | Output |
|---------|--------|-------|--------|
| Solve `Ax = b` | Gaussian Elimination | Matrix A, Vector b | Solution vector x |
| Solve `Ax = b` | Cramer's Rule | Matrix A, Vector b | Solution vector x |
| Determinant | Cofactor Expansion | Square matrix A | Scalar det(A) |
| Matrix Inverse | Gauss-Jordan | Square matrix A | Matrix A⁻¹ |
| LU Decomposition | LU with Pivoting | Square matrix A | (P, L, U) tuple |
| Eigenvalues | Characteristic Poly | Square matrix A | (eigenvalues, eigenvectors) |

**Python API**:
```python
mathsteps.linear_system(A, b)
mathsteps.cramers_rule(A, b)
mathsteps.determinant(A)
mathsteps.inverse(A)
mathsteps.lu(A)
mathsteps.eigenvalues(A)
```

**CLI**:
```bash
mathsteps linear-system --A "1 2; 3 4" --b "5 11"
mathsteps cramer --A "..." --b "..."
mathsteps determinant --A "..."
mathsteps inverse --A "..."
mathsteps lu --A "..."
mathsteps eigen --A "..."
```

### 🔍 Root Finding (Equations & Transcendental)

| Problem | Solver | Requirements |
|---------|--------|--------------|
| Find x where f(x)=0 | **Newton-Raphson** | Initial guess x₀ |
| Find x where f(x)=0 | **Bisection** | Bracket [a, b] with f(a)·f(b)<0 |
| Find x where f(x)=0 | **Secant** | Two guesses x₀, x₁ |
| Find x where x=g(x) | **Fixed-Point** | Initial guess x₀, function g(x) |

**Python API**:
```python
mathsteps.root(f, method="newton_raphson", x0=...)
mathsteps.root(f, method="bisection", a=..., b=...)
mathsteps.root(f, method="secant", x0=..., x1=...)
mathsteps.root(f, method="fixed_point", x0=...)
```

**CLI**:
```bash
mathsteps root --function "cos(x) - x" --method newton --x0 0.5
mathsteps root --function "x**3 - 2" --method bisection --a 1 --c 2
mathsteps solve "x**3 - x - 2"  # auto-detects
```

### 📈 Differential Equations (IVPs & BVPs)

| Problem | Solver | Input | Output |
|---------|--------|-------|--------|
| dy/dx = f(x,y), y(x₀)=y₀ | **Euler** | y₀, x₀, x_end, h | y(x_end) |
| dy/dx = f(x,y), y(x₀)=y₀ | **Heun** | y₀, x₀, x_end, h | y(x_end) |
| dy/dx = f(x,y), y(x₀)=y₀ | **RK4** (4th order) | y₀, x₀, x_end, h | y(x_end) |
| dy/dx = f(x,y), y(x₀)=y₀ | **RK45** (adaptive) | y₀, x₀, x_end, h | y(x_end) |
| dy/dx = f(x,y), y(x₀)=y₀ | **Midpoint** | y₀, x₀, x_end, h | y(x_end) |
| y'' + p(x)y' + q(x)y = r(x), BCs | **Shooting** | p, q, r, a, b, α, β | Approximate y(x) |
| y'' + p(x)y' + q(x)y = r(x), BCs | **Finite Difference** | p, q, r, a, b, α, β, n | [y₀, y₁, ...] |

**Python API**:
```python
mathsteps.ivp(f_expr, y0=..., x_end=..., h=..., method="rk4")
mathsteps.bvp_shooting(f, a, b, alpha, beta, s0, s1)
mathsteps.bvp_finite_difference(p, q, r, a, b, alpha, beta, n)
```

**CLI**:
```bash
mathsteps ivp --f-expr "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
mathsteps bvp --f-expr "..." --a 0 --c 1 --alpha 0 --beta 1 --s0 1 --s1 2
mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2  # auto-detects ODE
```

### 🧮 Numerical Methods (Integration, Differentiation)

| Problem | Solver | Formula |
|---------|--------|---------|
| ∫ₐᵇ f(x)dx | **Simpson's Rule** | (h/3)(f₀ + 4f₁ + 2f₂ + ... + fₙ) |
| ∫ₐᵇ f(x)dx | **Trapezoidal** | (h/2)(f₀ + 2f₁ + ... + fₙ) |
| f'(x) at point | **Central Difference** | [f(x+h) - f(x-h)] / 2h |
| f'(x) at point | **Forward Difference** | [f(x+h) - f(x)] / h |
| f'(x) at point | **Backward Difference** | [f(x) - f(x-h)] / h |

**Python API**:
```python
mathsteps.integrate(f, a, b, n=100, method="simpson")
mathsteps.differentiate(f, x, h=0.001, method="central")
```

**CLI**:
```bash
mathsteps integrate --function "sin(x)" --a 0 --c 3.14 --n 100 --method simpson
mathsteps diff --function "x**2" --x 2 --h 0.001 --method central
```

### 🔀 Interpolation & Polynomial Fitting

| Problem | Solver | Input | Output |
|---------|--------|-------|--------|
| Poly through points | **Lagrange** | List of (x, y) points | SymPy polynomial |
| Poly through points | **Newton Divided Differences** | List of (x, y) points | SymPy polynomial |

**Python API**:
```python
mathsteps.lagrange([(0,1), (1,2), (2,5)])
mathsteps.newton_divided_differences([(0,1), (1,2), (2,5)])
```

**CLI**:
```bash
mathsteps interp --points "(0,1); (1,2); (2,5)"
```

## Key Features

### ✅ Step-by-Step Explanation
Every operation is broken down into detailed steps:
```python
result = mathsteps.linear_system(A, b)
for step in result.steps:
    print(f"{step.description}")
    # Output: "Row swap: R1 ↔ R2"
    #         "Eliminate below pivot: R2 = R2 - 3*R1"
    #         "Back substitution: x2 = 7/2"
    #         etc.
```

### ✅ Automatic Verification
Results are checked against trusted libraries:
```python
result.verified  # True if verified against NumPy/SciPy/SymPy
```

### ✅ Structured Step Data
Each step has:
- `description`: Human-readable text
- `before`: State before the operation (as string)
- `after`: State after the operation (as string)
- `data`: Structured dict with numerical details

### ✅ Multiple Solution Methods
Compare approaches:
```python
for method in ["euler", "rk4", "rk45"]:
    result = mathsteps.ivp(f, y0=1, x_end=2, h=0.1, method=method)
    print(f"{method}: y({2}) = {result.answer}")
```

### ✅ Flexible Input (Python API)
- Pass matrices as lists: `[[1,2],[3,4]]`
- Pass expressions as strings: `"cos(x) - x"` (SymPy compatible)
- Pass functions as strings: `"-2*x*y"` for ODEs
- No numpy/sympy conversion needed — handled internally

### ✅ Dual Interface

**As a CLI** (for quick checks, scripting):
```bash
mathsteps linear-system --A "1 2; 3 4" --b "5 11"
```

**As a Library** (for applications, notebooks, scripts):
```python
import mathsteps
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
```

### ✅ JSON Problem Files
Store and solve from JSON:
```bash
mathsteps solve examples/linear_system_3x3.json
```

## Result Object Structure

```python
result = mathsteps.linear_system(A, b)

# Attributes available:
result.solver        # str: solver name ("gaussian_elimination")
result.steps         # list[Step]: step-by-step breakdown
result.answer        # varies: the final answer
result.verified      # bool: True if verified against trusted library
result.problem       # dict: original problem definition
result.__str__()     # formatted output with all info
result.__repr__()    # machine-readable representation
```

## Step Object Structure

```python
for step in result.steps:
    step.description  # str: "Swap rows R1 and R2"
    step.before       # str: state representation before
    step.after        # str: state representation after
    step.data         # dict: structured data (row nums, coefficients, etc)
```

## Return Types by Problem

| Function | Returns |
|----------|---------|
| `linear_system` | List of floats: `[x₁, x₂, ..., xₙ]` |
| `determinant` | Scalar: `float` |
| `inverse` | NumPy array: `(n, n)` matrix |
| `lu` | Tuple: `(P, L, U)` each NumPy array |
| `eigenvalues` | Tuple: `(eigs, eigvecs)` |
| `cramers_rule` | List of floats: `[x₁, x₂, ..., xₙ]` |
| `root` | Scalar: `float` |
| `ivp` | Scalar: `float` (value at x_end) |
| `bvp_*` | `ndarray` of `y` on the grid `details["x"]` |
| `integrate` | Scalar: `float` (integral value) |
| `differentiate` | Scalar: `float` (derivative value) |
| `lagrange` | Callable polynomial (`p(x)`, `p.expr`) |
| `newton_divided_differences` | Callable polynomial (`p(x)`, `p.expr`) |

## Parameter Types

### Linear Algebra
```python
A: Sequence[Sequence[float]]    # 2D matrix
b: Sequence[float]               # Vector
```

### Root Finding
```python
function: str                    # SymPy expression ("cos(x) - x")
variable: str                    # Variable name (default "x")
method: str                      # "newton_raphson", "bisection", "secant", "fixed_point"
a, b: float                      # Bisection brackets
x0, x1: float                    # Initial guesses
tol: float                       # Tolerance (default 1e-10)
max_iter: int                    # Max iterations (default 50)
```

### ODE Solving (IVP)
```python
f_expr: str                      # Right-hand side ("-2*x*y")
y0: float                        # Initial y value
x0: float                        # Initial x (default 0)
x_end: float                     # End of integration
h: float                         # Step size
method: str                      # "euler", "heun", "midpoint", "rk4", "rk45"
```

### Numerical Integration
```python
f: str                           # Function ("sin(x)")
a: float                         # Lower bound
b: float                         # Upper bound
n: int                           # Number of intervals
method: str                      # "simpson" or "trapezoidal"
```

### Numerical Differentiation
```python
f: str                           # Function ("sin(x)")
x: float                         # Point to evaluate
h: float                         # Step size (default 0.001)
method: str                      # "central", "forward", "backward"
```

## CLI Command Structure

### Three ways to use the CLI:

1. **Direct subcommands** (most common):
   ```bash
   mathsteps linear-system --A "..." --b "..."
   mathsteps root --function "..." --method newton --x0 1.5
   ```

2. **Free-form problem** (auto-detected):
   ```bash
   mathsteps solve "x**3 - x - 2"
   mathsteps solve "y' = -2*x*y" --y0 1 --x-end 2
   ```

3. **JSON file**:
   ```bash
   mathsteps solve examples/problem.json
   mathsteps verify examples/problem.json
   ```

4. **Interactive wizard**:
   ```bash
   mathsteps ask
   ```

## Available Solvers (Query at Runtime)

```python
import mathsteps

solvers = mathsteps.available_solvers()
# Returns: ['gaussian_elimination', 'bisection', 'newton_raphson', 
#           'secant', 'fixed_point', 'rk4', 'euler', 'heun', 
#           'midpoint', 'rk45', 'lagrange', 'lagrange_ndd', ...]
```

## Error Handling

Common exceptions:
- `SingularMatrixError`: Matrix has no inverse
- `ConvergenceError`: Method didn't converge
- `BracketError`: Bisection bracket invalid
- `ValueError`: Invalid input
- `SymPyParseError`: Expression syntax error

## Performance Notes

- **First import**: ~500ms (SymPy initialization)
- **Gaussian elimination**: O(n³)
- **Eigenvalues**: O(n³) to O(n⁴)
- **Root finding**: Depends on method and convergence
- **ODE solving**: O(n·steps) where n = problem size

## What's NOT Implemented

- **Sparse matrix methods** (full matrices only)
- **Partial differential equations** (ODEs only)
- **Complex numbers** (real numbers only, mostly)
- **Symbolic algebra** (SymPy used internally but results are numerical)
- **Constraint optimization** (root finding only)

## References

- **Linear Algebra**: Uses NumPy/SciPy backends
- **Root Finding**: Custom implementations + SciPy references
- **ODEs**: Custom step-tracking + SciPy/SymPy backends
- **Verification**: NumPy, SciPy, SymPy ground truth


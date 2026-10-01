# Using MathSteps as a Python Library

This guide covers everything you need to use `mathsteps` in Python scripts, notebooks, or applications.

## Installation

```bash
pip install mathsteps
```

Then import:
```python
import mathsteps
```

## Basic Workflow

Every mathsteps function follows this pattern:

1. **Call the function** with your problem
2. **Get a `Result` object** back
3. **Access the answer** via `result.answer`
4. **Inspect the steps** via `result.steps`
5. **Check verification** via `result.verified`

```python
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
#        ↓
#        Returns Result with answer, steps, verified
```

## All Available Functions

| Function | Problem Type | Returns |
|----------|--------------|---------|
| `linear_system(A, b)` | Solve Ax = b | `ndarray` x |
| `determinant(A)` | Find det(A) | `float` |
| `inverse(A)` | Find A⁻¹ | `ndarray` |
| `lu(A)` | LU decomposition | `(P, L, U)` arrays, `P @ A = L @ U` |
| `eigenvalues(A)` | Find eigenvalues/vectors | `(values, vectors)` like `numpy.linalg.eig` |
| `cramers_rule(A, b)` | Solve Ax = b (Cramer's) | `ndarray` x |
| `root(f, method, ...)` | Find root of f(x)=0 | `float` |
| `ivp(f_expr, y0, x_end, h, method)` | Solve dy/dx = f(x,y); systems, backwards | `y(x_end)`; `details["x"]`, `details["y"]` |
| `ivp_second_order(f_expr, y0, dy0, ...)` | Solve y'' = f(x, y, y') | `[y, y']` at `x_end` |
| `bvp_shooting(f, y_left, y_right, ...)` | Solve BVP (shooting) | `y` on grid `details["x"]` |
| `bvp_finite_difference(p, q, r, a, b, alpha, beta, n)` | Solve BVP (FD) | `y` on grid `details["x"]` |
| `lagrange(points)` | Polynomial interpolation | callable polynomial |
| `newton_divided_differences(points)` | Newton form interpolation | callable polynomial |
| `integrate(f, a, b, n, method)` | Numerical integration | `float` |
| `differentiate(f, x, h, method)` | Numerical derivative | `float` |
| `solve_problem(problem_dict)` | Generic solver | varies |
| `available_solvers()` | List all solvers | list[str] |

## Examples by Problem Type

### 1. Linear Systems

```python
import mathsteps

# Solve 2x₁ + x₂ = 5
#        3x₁ + 4x₂ = 11
A = [[2, 1], [3, 4]]
b = [5, 11]

result = mathsteps.linear_system(A, b)

print(f"Answer: {result.answer}")  # [1. 2.]  (numpy array; result.exact has the rationals)
print(f"Solver: {result.solver}")  # gaussian_elimination
print(f"Steps: {len(result.steps)}")  # e.g., 5

# See the breakdown
for i, step in enumerate(result.steps, 1):
    print(f"Step {i}: {step.description}")
```

### 2. Root Finding

```python
# Find x where cos(x) = x
result = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
print(result.answer)      # 0.7390851332151607
print(result.verified)    # True (checked against SciPy)

# Bisection method (bracket required)
result = mathsteps.root("x**3 - x - 2", method="bisection", a=1, b=2)
print(result.answer)      # 1.5213797...

# Secant method
result = mathsteps.root("x**3 - x - 2", method="secant", x0=1.0, x1=2.0)
print(result.answer)

# Try fixed-point: rearrange as x = g(x) and find fixed point
result = mathsteps.root("cos(x)", method="fixed_point", x0=0.7)
```

### 3. Differential Equations (IVPs)

```python
# Solve dy/dx = -2xy with y(0) = 1, integrate to x = 2, step size h = 0.1
result = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")
print(f"y(2) ≈ {result.answer}")  # ≈ exp(-4) ≈ 0.0183

# Try different methods
for method in ["euler", "rk4", "heun", "midpoint"]:
    r = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method=method)
    print(f"{method:8} → y(2) = {r.answer:.6f}")
```

### 4. Matrix Operations

```python
# Determinant
A = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
result = mathsteps.determinant(A)
print(result.answer)  # -306.0  (result.exact == -306)

# Inverse
A = [[1, 2], [3, 4]]
result = mathsteps.inverse(A)
print(result.answer)
# A numpy array: [[-2. , 1. ], [ 1.5, -0.5]]  (result.exact is a sympy.Matrix)

# LU Decomposition
P, L, U = mathsteps.lu(A).answer
# numpy arrays with P @ A = L @ U (partial pivoting)

# Eigenvalues
A = [[2, 0, 0], [0, 3, 4], [0, 4, 9]]
values, vectors = mathsteps.eigenvalues(A).answer
# numpy.linalg.eig layout: A @ vectors == vectors * values (unit-norm columns, one per
# independent eigenvector; a repeated eigenvalue appears once per eigenvector)
```

### 5. Interpolation

```python
# Lagrange polynomial through (0,1), (1,2), (2,5)
points = [[0, 1], [1, 2], [2, 5]]
result = mathsteps.lagrange(points)
p = result.answer          # a callable polynomial
print(p(3), p([0, 1, 2]))  # evaluate at a point or an array (exact internally)
print(p.expr)              # x**2 + 1   (SymPy expression)

# Newton divided differences
result = mathsteps.newton_divided_differences(points)
print(result.answer.expr)  # same polynomial
```

### 6. Numerical Integration

```python
# ∫₀^π sin(x) dx using Simpson's rule with 100 intervals
result = mathsteps.integrate("sin(x)", a=0, b=3.14159, n=100, method="simpson")
print(result.answer)  # ≈ 2.0

# Try different methods
for method in ["simpson", "trapezoidal"]:
    r = mathsteps.integrate("sin(x)", a=0, b=3.14159, n=100, method=method)
    print(f"{method}: {r.answer}")
```

### 7. Numerical Differentiation

```python
# f'(x) for f(x) = sin(x) at x = 1
result = mathsteps.differentiate("sin(x)", x=1.0, h=0.001)
print(result.answer)  # ≈ cos(1) ≈ 0.5403

# Different methods  ("forward", "backward", "central")
result = mathsteps.differentiate("sin(x)", x=1.0, h=0.001, method="central")
print(result.answer)
```

## Understanding Steps

Each `Step` object contains detailed information about what happened:

```python
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])

for step in result.steps:
    print(f"Description: {step.description}")  # Human-readable
    print(f"Before:      {step.before}")       # State before
    print(f"After:       {step.after}")        # State after
    print(f"Data:        {step.data}")         # Structured dict
    print()
```

Example output:
```
Description: Row swap: R1 ↔ R2
Before:      [[1, 2, 5], [3, 4, 11]]
After:       [[3, 4, 11], [1, 2, 5]]
Data:        {'row1': 0, 'row2': 1}
```

## Error Handling

```python
import mathsteps
import sys

try:
    # Invalid matrix (singular)
    result = mathsteps.linear_system([[1, 2], [2, 4]], [1, 2])
    
    if not result.verified:
        print("Warning: Answer may not be reliable")
    
    print(result.answer)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
```

## Disabling Verification

By default, every function verifies its answer against NumPy/SciPy. To skip this (faster):

```python
# Don't verify against numpy/scipy
result = mathsteps.linear_system(A, b, verify=False)
print(result.verified)  # False
```

## Working with Results Programmatically

```python
result = mathsteps.root("x**2 - 4", method="newton", x0=1.0)

# Extract answer
x_root = float(result.answer)

# Count steps
num_steps = len(result.steps)

# Get only the descriptions
descriptions = [step.description for step in result.steps]

# Get structured data
data_only = [step.data for step in result.steps if step.data]

# Re-run with same problem
result2 = mathsteps.solve_problem(result.problem)
```

## Using in Jupyter Notebooks

```python
import mathsteps
from IPython.display import display, HTML

result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])

# Display answer
print(f"✓ Answer: {result.answer}")

# Display steps
print("\nStep-by-step solution:")
for i, step in enumerate(result.steps, 1):
    print(f"  {i}. {step.description}")

# Display as HTML table
import pandas as pd
df = pd.DataFrame([
    {
        "Step": i,
        "Description": s.description,
        "Before": s.before,
        "After": s.after
    }
    for i, s in enumerate(result.steps, 1)
])
display(df)
```

## Available Solvers

```python
# List all registered solvers
solvers = mathsteps.available_solvers()
print(solvers)
# ['gaussian_elimination', 'bisection', 'newton_raphson', 'rk4', ...]

# The solver name is always in result.solver
result = mathsteps.root("cos(x) - x", method="newton", x0=0)
print(result.solver)  # "newton_raphson"
```

## Parameters Reference

### `root(function, *, variable="x", method="newton_raphson", ...)`
- `function`: SymPy expression (string)
- `variable`: Variable name (default "x")
- `method`: "newton_raphson" | "bisection" | "secant" | "fixed_point"
- **Newton/Fixed-point**: need `x0` (initial guess)
- **Bisection**: need `a, b` (bracket where f(a) and f(b) have opposite signs)
- **Secant**: need `x0, x1` (two starting guesses)
- `tol`: convergence tolerance (default 1e-10)
- `max_iter`: max iterations (default 50)

### `ivp(f_expr, *, variable="y", y0, x0=0, x_end, h, method="rk4")`
- `f_expr`: Right-hand side as string (e.g., "-2*x*y")
- `y0`: Initial value at x0
- `x0`: Initial x (default 0)
- `x_end`: Integration end point
- `h`: Step size
- `method`: "rk4" | "euler" | "heun" | "midpoint" | "rk45"

### `integrate(f, a, b, n=100, *, method="simpson")`
- `f`: Function expression (string)
- `a`: Lower bound
- `b`: Upper bound
- `n`: Number of sub-intervals
- `method`: "simpson" | "trapezoidal"

### `differentiate(f, x, h=0.001, *, method="central")`
- `f`: Function expression (string)
- `x`: Point to evaluate at
- `h`: Step size for finite differences
- `method`: "central" | "forward" | "backward"

## Tips & Best Practices

1. **Always check `result.verified`** for verification status
2. **Handle verification failures gracefully** — it might still be correct
3. **Use the right method** — different solvers have different accuracy/speed
4. **Small step sizes (`h`) give better accuracy** in numerical methods
5. **Use iteration limits** (`max_iter`) to prevent hanging
6. **Catch exceptions** — malformed expressions will raise errors
7. **Print steps for debugging** — step descriptions help verify correctness
8. **Use `solve_problem(dict)`** for maximum flexibility

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MatrixError/SingularMatrix | Matrix is singular; no unique solution |
| ConvergenceError | Method didn't converge; try different x0 or method |
| No root in bracket | Ensure f(a) and f(b) have opposite signs |
| Inaccurate result | Use smaller step size `h` or more intervals `n` |
| SymPy parse error | Check expression syntax; use `x`, `y` as variables |

## See Also

- **CLI Guide**: Run `mathsteps --help` for command-line usage
- **Examples**: See `examples/` directory for JSON problem files

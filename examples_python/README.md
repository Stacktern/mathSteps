# Python Examples for MathSteps

These examples demonstrate how to use **mathsteps** as a Python library.

## Quick Start

### 1. Verify Installation

```bash
python 00_test_installation.py
```

This script tests all core functionality to ensure mathsteps is properly installed.

### 2. Explore Examples

Each example file focuses on a different problem domain:

| File | Topic | Description |
|------|-------|-------------|
| `00_test_installation.py` | **Testing** | Verify mathsteps installation |
| `01_linear_algebra.py` | **Linear Algebra** | Solving systems, determinants, matrix operations |
| `02_root_finding.py` | **Root Finding** | Newton-Raphson, bisection, secant methods |
| `03_differential_equations.py` | **ODEs** | Solving initial value problems with RK4, Euler, etc. |
| `04_numpy_integration.py` | **NumPy Integration** | Using mathsteps with NumPy, batch operations |

### 3. Run Individual Examples

```bash
# Linear algebra examples
python 01_linear_algebra.py

# Root finding with different methods
python 02_root_finding.py

# Solving differential equations
python 03_differential_equations.py

# Integration with NumPy
python 04_numpy_integration.py
```

## What You'll Learn

### Basic Usage Pattern

Every mathsteps function follows this pattern:

```python
import mathsteps

# 1. Call the function
result = mathsteps.linear_system(A, b)

# 2. Get the result (Result object)
# result.answer      ← the solution
# result.steps       ← list of Step objects
# result.verified    ← True if verified against NumPy/SciPy
# result.solver      ← name of solver used
```

### Understanding Steps

Each `Step` contains:
- `description`: What happened (e.g., "Row reduction: R2 = R2 - 3*R1")
- `before`: State before (e.g., initial matrix)
- `after`: State after (e.g., matrix after operation)
- `data`: Structured info (e.g., row indices, multipliers)

### Verification

By default, every result is verified:
```python
result = mathsteps.linear_system(A, b)
if result.verified:
    print("✓ Answer is correct (verified against NumPy)")
else:
    print("⚠ Warning: could not verify answer")
```

## Problem Types

### 📊 Linear Algebra
- Solve `Ax = b` (Gaussian elimination, Cramer's rule)
- Determinant (cofactor expansion)
- Matrix inverse (Gauss-Jordan)
- LU decomposition
- Eigenvalues/eigenvectors

### 🔍 Root Finding
- Newton-Raphson method
- Bisection method
- Secant method
- Fixed-point iteration

### 📈 Differential Equations
- Initial value problems (IVPs)
- Methods: Euler, Heun, Midpoint, RK4, RK45
- Boundary value problems (shooting, finite difference)

### 🧮 Numerical Methods
- Integration (Simpson's, trapezoidal)
- Differentiation (forward, backward, central differences)
- Polynomial interpolation (Lagrange, Newton)

## Tips

1. **Start with `00_test_installation.py`** to verify everything works
2. **Read each example's docstrings** for detailed explanations
3. **Print `result` directly** for a formatted output:
   ```python
   result = mathsteps.root("cos(x) - x", method="newton", x0=0)
   print(result)  # Shows steps and verification
   ```
4. **Use step data for automation**:
   ```python
   for step in result.steps:
       print(f"{step.description}: {step.data}")
   ```
5. **Compare methods side-by-side** for accuracy/speed tradeoffs

## Common Patterns

### Get just the answer
```python
result = mathsteps.linear_system(A, b)
answer = result.answer
```

### Print step-by-step solution
```python
for i, step in enumerate(result.steps, 1):
    print(f"Step {i}: {step.description}")
```

### Verify result
```python
if result.verified:
    print("✓ Verified against NumPy/SciPy/SymPy")
else:
    print("⚠ Could not verify")
```

### Skip verification (faster)
```python
result = mathsteps.linear_system(A, b, verify=False)
```

### Get solver info
```python
print(result.solver)    # e.g., "gaussian_elimination"
print(result.problem)   # Original problem dict
```

## Next Steps

- Read **[docs/python-guide.md](../docs/python-guide.md)** for comprehensive documentation
- Check **[README.md](../README.md)** for CLI usage & overview
- See `examples/` directory for JSON problem files

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'mathsteps'` | Run `pip install mathsteps` first |
| `SingularMatrixError` | Matrix is singular; no unique solution |
| `ConvergenceError` | Root finder didn't converge; try different x0 |
| `SymPyParseError` | Check expression syntax (use valid SymPy expressions) |
| Test script fails | Check Python version (need 3.10+): `python --version` |

## Examples Structure

Each example file has:
- **Clear function names** describing what's solved
- **Docstrings** explaining the problem
- **Step-by-step output** showing the solution process
- **Verification** where applicable
- **Comments** explaining the code

Run them in order for progressive complexity!

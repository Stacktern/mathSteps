#!/usr/bin/env python3
"""
Example: Integration with Other Libraries

Shows how to use mathsteps results with NumPy, SciPy, and Pandas
for further analysis and visualization.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps
import numpy as np

def example_integrate_with_numpy():
    """Use mathsteps with NumPy arrays."""
    print("=" * 60)
    print("EXAMPLE: Integration with NumPy")
    print("=" * 60)
    
    # Solve a linear system
    A_list = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b_list = [8, -11, -3]
    
    result = mathsteps.linear_system(A_list, b_list)
    
    print(f"Problem: Solve Ax = b")
    print(f"Solver: {result.solver}")
    print(f"Steps: {len(result.steps)}")
    
    # Convert to NumPy arrays for further computation
    x_numpy = np.array(result.answer, dtype=float)
    A_numpy = np.array(A_list, dtype=float)
    b_numpy = np.array(b_list, dtype=float)
    
    print(f"\n[OK] Solution: x = {x_numpy}")
    
    # Verify with NumPy
    residual = A_numpy @ x_numpy - b_numpy
    print(f"Verification (Ax - b): {residual}")
    print(f"All components near zero: {np.allclose(residual, 0)}")
    print()


def example_steps_to_dataframe():
    """Convert step information to a table/DataFrame format."""
    print("=" * 60)
    print("EXAMPLE: Analyzing Steps as Structured Data")
    print("=" * 60)
    
    result = mathsteps.root("x**2 - 4", method="newton", x0=1.0)
    
    print(f"Problem: Find x where x² = 4 (using Newton's method)")
    print(f"\n{len(result.steps)} steps total")
    
    # Extract step information as structured data
    step_data = []
    for i, step in enumerate(result.steps, 1):
        step_data.append({
            "step_num": i,
            "description": step.description,
            "has_before": step.before is not None,
            "has_after": step.after is not None,
            "data_keys": list(step.data.keys()) if step.data else []
        })
    
    print("\nStructured step data:")
    for item in step_data:
        print(f"  Step {item['step_num']}: {item['description'][:50]}")
        if item['data_keys']:
            print(f"    Data keys: {item['data_keys']}")
    
    print(f"\n[OK] Root found: x = {float(result.answer):.10f}")
    print()


def example_batch_computation():
    """Solve multiple related problems."""
    print("=" * 60)
    print("EXAMPLE: Batch Computation (Multiple Problems)")
    print("=" * 60)
    
    print("Solving: x² - c = 0 for c = 1, 2, 3, 4, 5")
    print("\n{:10} {:<20} {:<15}".format("c", "Root (x = √c)", "Steps"))
    print("-" * 45)
    
    for c in range(1, 6):
        result = mathsteps.root(f"x**2 - {c}", method="newton", x0=1.0)
        sqrt_c = np.sqrt(c)
        error = abs(float(result.answer) - sqrt_c)
        print("{:<10} {:<20.10f} {:<15}".format(
            c,
            float(result.answer),
            len(result.steps)
        ))
    print()


def example_numerical_integration_analysis():
    """Analyze a numerical integration with different settings."""
    print("=" * 60)
    print("EXAMPLE: Analyzing Numerical Accuracy")
    print("=" * 60)
    
    print("Computing ∫₀^π sin(x) dx")
    print("Exact answer: 2.0")
    
    pi = 3.14159265359
    methods = ["simpson", "trapezoidal"]
    interval_sizes = [10, 50, 100, 500]
    
    for method in methods:
        print(f"\n{method.upper()} METHOD:")
        print("{:8} {:<15} {:<15}".format("Intervals", "Result", "Error"))
        print("-" * 38)
        
        for n in interval_sizes:
            result = mathsteps.integrate(
                "sin(x)", a=0, b=pi, n=n, method=method
            )
            error = abs(result.answer - 2.0)
            print("{:<8} {:<15.10f} {:<15.6e}".format(
                n,
                result.answer,
                error
            ))
    
    print("\nObservations:")
    print("  • More intervals → more accurate")
    print("  • Simpson's rule converges faster than trapezoidal")
    print()


def example_eigenvalues_with_numpy():
    """Solve eigenvalue problem and use NumPy for additional analysis."""
    print("=" * 60)
    print("EXAMPLE: Eigenvalue Analysis with NumPy")
    print("=" * 60)
    
    A = [[2, 0, 0], [0, 3, 4], [0, 4, 9]]
    
    result = mathsteps.eigenvalues(A)
    eigs, eigvecs = result.answer
    
    print(f"Matrix A:")
    for row in A:
        print(f"  {row}")
    
    print(f"\nEigenvalues found: {len(eigs)}")
    for i, eig in enumerate(eigs, 1):
        print(f"  λ_{i} = {eig}")
    
    # Convert to NumPy for verification
    A_numpy = np.array(A, dtype=float)
    eigenvalues_numpy = np.linalg.eigvalsh(A_numpy)
    
    print(f"\nNumPy eigenvalues (for comparison):")
    for i, eig in enumerate(eigenvalues_numpy, 1):
        print(f"  λ_{i} = {eig:.10f}")
    
    print(f"\nVerification: {result.verified}")
    print()


if __name__ == "__main__":
    example_integrate_with_numpy()
    example_steps_to_dataframe()
    example_batch_computation()
    example_numerical_integration_analysis()
    example_eigenvalues_with_numpy()

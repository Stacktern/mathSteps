#!/usr/bin/env python3
"""
Example: Root Finding with MathSteps

Demonstrates finding roots using different numerical methods:
Newton-Raphson, Bisection, Secant, Fixed-Point.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps

def example_newton_raphson():
    """Find the Dottie number using Newton-Raphson method."""
    print("=" * 60)
    print("EXAMPLE: Finding Dottie Number (Newton-Raphson)")
    print("=" * 60)
    print("\nProblem: Find x where cos(x) = x")
    print("(This is the Dottie number, a unique solution)")
    
    result = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
    
    print(f"\nMethod: {result.solver}")
    print(f"Initial guess: x₀ = 0.0")
    print(f"Number of steps: {len(result.steps)}")
    print(f"Verified against SciPy: {result.verified}")
    
    print("\nNewton iterations:")
    for i, step in enumerate(result.steps, 1):
        if "Iteration" in step.description or "Initial" in step.description:
            print(f"  {i}. {step.description}")
            if step.after:
                print(f"     Result: {step.after}")
    
    print(f"\n[OK] Root found: x ~= {float(result.answer):.15f}")
    print(f"  Verification: cos({float(result.answer):.6f}) ~= {float(result.answer):.6f}")
    print()


def example_bisection():
    """Find root using bisection method."""
    print("=" * 60)
    print("EXAMPLE: Finding Root (Bisection Method)")
    print("=" * 60)
    print("\nProblem: Find x where x³ - x - 2 = 0 in [1, 2]")
    
    result = mathsteps.root("x**3 - x - 2", method="bisection", a=1, b=2)
    
    print(f"\nMethod: {result.solver}")
    print(f"Bracket: [{1}, {2}]")
    print(f"Number of iterations: {len(result.steps)}")
    
    # Show first few and last few bisections
    print("\nBisection iterations (first 3):")
    for i, step in enumerate(result.steps[:3], 1):
        print(f"  {i}. {step.description}")
    
    if len(result.steps) > 6:
        print(f"  ... ({len(result.steps) - 6} more iterations)")
        print("\nLast 2 iterations:")
        for i, step in enumerate(result.steps[-2:], len(result.steps) - 1):
            print(f"  {i}. {step.description}")
    
    print(f"\n[OK] Root found: x ~= {float(result.answer):.10f}")
    print()


def example_secant():
    """Find root using secant method."""
    print("=" * 60)
    print("EXAMPLE: Finding Root (Secant Method)")
    print("=" * 60)
    print("\nProblem: Find x where x³ - x - 2 = 0")
    print("Using two starting guesses: x₀=1.0, x₁=2.0")
    
    result = mathsteps.root("x**3 - x - 2", method="secant", x0=1.0, x1=2.0)
    
    print(f"\nMethod: {result.solver}")
    print(f"Number of iterations: {len(result.steps)}")
    
    print("\nSecant iterations:")
    for i, step in enumerate(result.steps, 1):
        if i <= 3 or i > len(result.steps) - 2:
            print(f"  {i}. {step.description}")
        elif i == 4:
            print(f"  ... ({len(result.steps) - 4} more iterations)")
    
    print(f"\n[OK] Root: {float(result.answer):.10f}")
    print()


def example_fixed_point():
    """Find fixed point using fixed-point iteration."""
    print("=" * 60)
    print("EXAMPLE: Finding Fixed Point (Fixed-Point Iteration)")
    print("=" * 60)
    print("\nProblem: Find fixed point of g(x) = cos(x)")
    print("(equivalent to solving x = cos(x))")
    
    result = mathsteps.root("cos(x)", method="fixed_point", x0=0.7, max_iter=200)
    
    print(f"\nMethod: {result.solver}")
    print(f"Initial guess: x₀ = 0.7")
    print(f"Number of iterations: {len(result.steps)}")
    
    print("\nIteration sequence:")
    for i, step in enumerate(result.steps, 1):
        if i <= 5 or i > len(result.steps) - 2:
            print(f"  {i}. {step.description}")
        elif i == 6:
            print(f"  ... ({len(result.steps) - 6} more)")
    
    print(f"\n[OK] Fixed point: x ~= {float(result.answer):.10f}")
    print()


def example_compare_methods():
    """Compare different root-finding methods."""
    print("=" * 60)
    print("EXAMPLE: Comparing Root-Finding Methods")
    print("=" * 60)
    print("\nFinding root of f(x) = x² - 4 = 0 (roots: ±2)")
    print("\n{:20} {:<15} {:<10}".format("Method", "Root Found", "Steps"))
    print("-" * 45)
    
    function = "x**2 - 4"
    methods = [
        {"method": "newton", "x0": 1.5},
        {"method": "bisection", "a": 1, "b": 3},
        {"method": "secant", "x0": 1.0, "x1": 3.0},
    ]
    
    for params in methods:
        method = params.pop("method")
        result = mathsteps.root(function, method=method, **params)
        print("{:20} {:<15.10f} {:<10}".format(
            result.solver,
            float(result.answer),
            len(result.steps)
        ))
    print()


if __name__ == "__main__":
    example_newton_raphson()
    example_bisection()
    example_secant()
    example_fixed_point()
    example_compare_methods()

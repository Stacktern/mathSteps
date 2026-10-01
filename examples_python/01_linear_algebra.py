#!/usr/bin/env python3
"""
Example: Linear Algebra with MathSteps

Demonstrates solving systems of equations, determinants, and matrix operations.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps

def example_linear_system():
    """Solve a 2x2 system of equations."""
    print("=" * 60)
    print("EXAMPLE: Solving Linear System (Gaussian Elimination)")
    print("=" * 60)
    print("\nProblem: Solve Ax = b")
    print("  2x₁ + x₂ = 5")
    print("  3x₁ + 4x₂ = 11")
    
    A = [[2, 1], [3, 4]]
    b = [5, 11]
    
    result = mathsteps.linear_system(A, b)
    
    print(f"\nSolver used: {result.solver}")
    print(f"Verified against numpy: {result.verified}")
    
    print("\nStep-by-step solution:")
    for i, step in enumerate(result.steps, 1):
        print(f"\n  Step {i}: {step.description}")
        if step.before:
            print(f"    Before: {step.before}")
        if step.after:
            print(f"    After:  {step.after}")
    
    print(f"\n[OK] Final Answer: x = {result.answer}")
    print()


def example_determinant():
    """Calculate the determinant of a 3x3 matrix."""
    print("=" * 60)
    print("EXAMPLE: Matrix Determinant (Cofactor Expansion)")
    print("=" * 60)
    
    A = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    print(f"\nMatrix:\n{A}")
    
    result = mathsteps.determinant(A)
    
    print(f"\nSolver: {result.solver}")
    print(f"Steps performed: {len(result.steps)}")
    
    # Show just descriptions (full step details are verbose for this)
    print("\nCalculation steps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  {i}. {step.description}")
    
    print(f"\n[OK] det(A) = {result.answer}")
    print()


def example_cramer_rule():
    """Solve using Cramer's rule."""
    print("=" * 60)
    print("EXAMPLE: Solve with Cramer's Rule")
    print("=" * 60)
    
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    
    print(f"\nSolving 3×3 system using Cramer's rule...")
    result = mathsteps.cramers_rule(A, b)
    
    print(f"Solver: {result.solver}")
    print(f"Verified: {result.verified}")
    print(f"\n[OK] Solution: x = {result.answer}")
    print()


if __name__ == "__main__":
    example_linear_system()
    example_determinant()
    example_cramer_rule()

#!/usr/bin/env python3
"""
Example: Solving Differential Equations with MathSteps

Demonstrates solving IVPs (Initial Value Problems) using different numerical methods:
Euler, Heun, Midpoint, RK4, RK45.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps
import math

def example_basic_ivp():
    """Solve a simple exponential decay ODE."""
    print("=" * 60)
    print("EXAMPLE: Exponential Decay ODE")
    print("=" * 60)
    print("\nProblem: dy/dx = -2xy")
    print("Initial condition: y(0) = 1")
    print("Find: y(2)")
    print("\nExact solution: y(x) = e^(-x^2)")
    print(f"Exact answer: y(2) = e^(-4) ~= {math.exp(-4):.6e}")

    result = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")

    print(f"\nNumerical solution (RK4, h=0.1):")
    print(f"Solver: {result.solver}")
    print(f"Number of steps: {len(result.steps)}")
    print(f"[OK] y(2) ~= {float(result.answer):.6e}")
    print(f"  Error: {abs(float(result.answer) - math.exp(-4)):.6e}")
    print()


def example_compare_methods():
    """Compare different ODE solving methods."""
    print("=" * 60)
    print("EXAMPLE: Comparing ODE Methods")
    print("=" * 60)
    print("\nSolving: dy/dx = -2xy, y(0)=1, find y(2)")
    print("Exact solution: y(2) = e^(-4) ~= 1.8316e-02")
    
    exact = math.exp(-4)
    methods = ["euler", "heun", "midpoint", "rk4"]
    h = 0.1

    print("\n" + "-" * 60)
    print("Step size h = 0.1")
    print("-" * 60)
    print("{:15} {:<18} {:<15}".format("Method", "Result", "Error"))
    print("-" * 48)

    for method in methods:
        result = mathsteps.ivp(
            "-2*x*y",
            y0=1, x_end=2, h=h, method=method
        )
        error = abs(float(result.answer) - exact)
        print("{:15} {:<18.10f} {:<15.6e}".format(
            result.solver,
            float(result.answer),
            error
        ))

    # Smaller step size
    print("\n" + "-" * 60)
    print("Step size h = 0.01 (more accurate)")
    print("-" * 60)
    print("{:15} {:<18} {:<15}".format("Method", "Result", "Error"))
    print("-" * 48)

    h = 0.01
    for method in methods:
        result = mathsteps.ivp(
            "-2*x*y",
            y0=1, x_end=2, h=h, method=method
        )
        error = abs(float(result.answer) - exact)
        print("{:15} {:<18.10f} {:<15.6e}".format(
            result.solver,
            float(result.answer),
            error
        ))

    print("\nObservations:")
    print("  - Smaller step size -> more accurate")
    print("  - RK4/RK45 converge faster than Euler")
    print("  - Higher-order methods are more efficient")
    print()


def example_population_growth():
    """Solve a population growth model."""
    print("=" * 60)
    print("EXAMPLE: Population Growth Model")
    print("=" * 60)
    print("\nProblem: dP/dt = 0.05 * P * (1 - P/1000)")
    print("Initial population: P(0) = 100")
    print("Find population after 50 time units")
    print("\nThis is a logistic growth model (carrying capacity = 1000)")

    result = mathsteps.ivp(
        "0.05*y*(1 - y/1000)",
        y0=100, x_end=50, h=0.5, method="rk4"
    )

    print(f"\nNumerical solution (RK4, h=0.5):")
    print(f"Steps: {len(result.steps)}")
    print(f"[OK] P(50) ≈ {float(result.answer):.2f}")
    print(f"  (Population approaches carrying capacity of 1000)")
    print()


def example_ivp_details():
    """Show detailed steps of an IVP solution."""
    print("=" * 60)
    print("EXAMPLE: Detailed ODE Solution Breakdown")
    print("=" * 60)
    print("\nProblem: dy/dx = x - y, y(0) = 1")
    print("Solve from x=0 to x=0.5 with h=0.1 (Euler method)")
    
    result = mathsteps.ivp(
        "x - y",
        y0=1, x_end=0.5, h=0.1, method="euler"
    )
    
    print(f"\nSolver: {result.solver}")
    print(f"Total steps: {len(result.steps)}")
    
    # Show first few and last few steps
    print("\nFirst 3 steps of the solution:")
    for i, step in enumerate(result.steps[:3], 1):
        print(f"\n  Step {i}: {step.description}")
        if step.before:
            print(f"    Before: {step.before}")
        if step.after:
            print(f"    After:  {step.after}")
    
    if len(result.steps) > 6:
        print(f"\n  ... ({len(result.steps) - 6} more internal steps)")
        print("\nLast step:")
        step = result.steps[-1]
        print(f"  {step.description}")
        if step.after:
            print(f"  Result: {step.after}")
    
    print(f"\n[OK] Final answer: y(0.5) ~= {float(result.answer):.6f}")
    print()


if __name__ == "__main__":
    example_basic_ivp()
    example_compare_methods()
    example_population_growth()
    example_ivp_details()

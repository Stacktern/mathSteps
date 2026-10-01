#!/usr/bin/env python3
"""
Interactive Example: Differential Equation Solver

Prompts user for ODE and solving parameters.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps
import sys
import math


def validate_expression(expr: str) -> str:
    """Validate that user provided an expression."""
    expr = expr.strip()
    if not expr:
        raise ValueError("Expression cannot be empty")
    return expr


def get_float(prompt: str) -> float:
    """Get a float from user input."""
    try:
        return float(input(prompt).strip())
    except ValueError:
        print("[X] Invalid number")
        sys.exit(1)


def interactive_ivp_solver():
    """Interactively solve an initial value problem."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Solve Initial Value Problem (IVP)")
    print("=" * 60)
    
    print("\nProblem: dy/dx = f(x, y)")
    print("Initial condition: y(x0) = y0")
    print("Find: y(x_end)")
    
    # Get ODE
    print("\nEnter right-hand side f(x, y):")
    print("Examples: '-2*x*y', 'y - x', 'sin(x) - y'")
    f_expr = validate_expression(input("> "))
    
    # Get initial conditions
    x0 = get_float("Initial x (x0): ")
    y0 = get_float("Initial y (y0): ")
    x_end = get_float("End point (x_end): ")
    h = get_float("Step size (h): ")
    
    # Validate
    if (x_end - x0) * h <= 0:
        print("[X] Step size sign must match direction (x_end - x0)")
        sys.exit(1)
    
    if h == 0:
        print("[X] Step size cannot be zero")
        sys.exit(1)
    
    # Choose method
    print("\nSelect ODE solving method:")
    print("  1. Euler (1st order, fast but less accurate)")
    print("  2. Heun (2nd order, medium)")
    print("  3. Midpoint (2nd order, medium)")
    print("  4. RK4 (4th order, accurate but slower)")
    print("  5. RK45 (adaptive, best accuracy)")
    
    method_choice = input("Choice (1-5): ").strip()
    
    methods = {
        "1": "euler",
        "2": "heun",
        "3": "midpoint",
        "4": "rk4",
        "5": "rk45"
    }
    
    if method_choice not in methods:
        print("[X] Invalid method")
        sys.exit(1)
    
    method = methods[method_choice]
    
    print(f"\nSolving: dy/dx = {f_expr}")
    print(f"Initial: y({x0}) = {y0}")
    print(f"Method: {method}")
    print(f"Step size: {h}")
    
    try:
        result = mathsteps.ivp(f_expr, y0=y0, x0=x0, x_end=x_end, h=h, method=method)
        
        print(f"\n{'=' * 60}")
        print("RESULT")
        print('=' * 60)
        print(f"[OK] Solution: y({x_end}) = {float(result.answer)}")
        print(f"  Solver: {result.solver}")
        print(f"  Steps computed: {len(result.steps)}")
        print(f"  Verified against SciPy: {result.verified}")
        
        # Show steps info
        show_steps = input("\nShow step breakdown? (y/n): ").strip().lower()
        if show_steps == 'y':
            print(f"\nStep-by-step computation ({len(result.steps)} steps):")
            for i, step in enumerate(result.steps, 1):
                if i <= 5 or i > len(result.steps) - 3:
                    print(f"  {i}. {step.description}")
                elif i == 6:
                    print(f"  ... ({len(result.steps) - 7} more steps)")
                    break
    
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_compare_methods():
    """Compare different ODE methods."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Compare ODE Methods")
    print("=" * 60)
    
    # Get ODE
    print("\nEnter ODE (dy/dx = f(x,y)):")
    print("Examples: '-2*x*y', 'y - x', 'sin(x)'")
    f_expr = validate_expression(input("> "))
    
    # Get parameters
    x0 = get_float("Initial x (x0): ")
    y0 = get_float("Initial y (y0): ")
    x_end = get_float("End point (x_end): ")
    h = get_float("Step size (h): ")
    
    if (x_end - x0) * h <= 0:
        print("[X] Step size sign must match direction")
        sys.exit(1)
    
    print(f"\nComparing methods for: dy/dx = {f_expr}")
    print(f"Initial: y({x0}) = {y0}")
    print(f"Compute: y({x_end})")
    print(f"Step size: {h}")
    
    try:
        methods = ["euler", "heun", "midpoint", "rk4"]
        results_data = []
        
        for method in methods:
            print(f"\n  Computing with {method}...", end=" ", flush=True)
            result = mathsteps.ivp(f_expr, y0=y0, x0=x0, x_end=x_end, h=h, method=method)
            results_data.append((method, result))
            print("✓")
        
        # Comparison table
        print(f"\n{'=' * 60}")
        print("METHOD COMPARISON")
        print('=' * 60)
        print(f"{'Method':<15} {'Result':<18} {'Steps':<10}")
        print("-" * 43)
        
        for name, result in results_data:
            print(f"{name:<15} {float(result.answer):<18.10f} {len(result.steps):<10}")
        
        # Find best (smallest step count = most efficient)
        best = min(results_data, key=lambda x: len(x[1].steps))
        print(f"\nMost efficient: {best[0]} ({len(best[1].steps)} computational steps)")

        # Accuracy comparison (difference from best)
        print(f"\nAccuracy (difference from {best[0]}):")
        best_answer = float(best[1].answer)
        for name, result in results_data:
            diff = abs(float(result.answer) - best_answer)
            print(f"  {name:<15}: {diff:.2e}")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def interactive_convergence_analysis():
    """Analyze convergence by varying step size."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Convergence Analysis")
    print("=" * 60)
    
    print("\nAnalyze how solution changes with step size")
    
    # Get ODE
    print("\nEnter ODE (dy/dx = f(x,y)):")
    f_expr = validate_expression(input("> "))
    
    # Get parameters
    x0 = get_float("Initial x (x0): ")
    y0 = get_float("Initial y (y0): ")
    x_end = get_float("End point (x_end): ")
    
    # Choose method
    print("\nSelect method:")
    print("  1. Euler")
    print("  2. RK4")
    
    method_choice = input("Choice (1-2): ").strip()
    method = "euler" if method_choice == "1" else "rk4"
    
    print(f"\nAnalyzing {method} for: dy/dx = {f_expr}")
    print(f"Initial: y({x0}) = {y0}, find y({x_end})")
    
    # Various step sizes
    h_values = [0.5, 0.1, 0.05, 0.01, 0.001]
    num_steps_list = []
    results_data = []
    
    print(f"\n{'h (step size)':<15} {'y(x_end)':<18} {'Steps':<10}")
    print("-" * 43)
    
    try:
        for h in h_values:
            if (x_end - x0) * h <= 0:
                continue
            
            result = mathsteps.ivp(f_expr, y0=y0, x0=x0, x_end=x_end, h=h, method=method)
            results_data.append((h, result))
            num_steps_list.append(len(result.steps))

            print(f"{h:<15.4f} {float(result.answer):<18.10f} {len(result.steps):<10}")
        
        # Convergence analysis
        print(f"\nConvergence Analysis:")
        if len(results_data) > 1:
            print(f"  Solution varies as step size decreases")
            min_h_result = results_data[-1]
            print(f"  Most accurate guess (h={min_h_result[0]}): y = {float(min_h_result[1].answer):.10f}")

            # Show trend
            print(f"\n  Error reduction with smaller h:")
            for i in range(1, len(results_data)):
                h_curr, res_curr = results_data[i]
                h_prev, res_prev = results_data[i-1]
                error = abs(float(res_curr.answer) - float(res_prev.answer))
                print(f"    h={h_prev} -> h={h_curr}: change = {error:.2e}")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def main():
    """Main interactive menu."""
    print("\n" + "=" * 60)
    print("INTERACTIVE ODE SOLVER")
    print("=" * 60)
    
    print("\nChoose task:")
    print("  1. Solve an ODE (single method)")
    print("  2. Compare methods (side-by-side)")
    print("  3. Convergence analysis (varying step size)")
    print("  4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        interactive_ivp_solver()
    elif choice == "2":
        interactive_compare_methods()
    elif choice == "3":
        interactive_convergence_analysis()
    elif choice == "4":
        print("Goodbye!")
        return
    else:
        print("[X] Invalid choice")
        return
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError) as exc:
        print(f"\n[X] {exc or 'No input received'}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

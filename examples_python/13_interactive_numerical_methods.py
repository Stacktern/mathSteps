#!/usr/bin/env python3
"""
Interactive Example: Numerical Methods

Prompts user for integration and differentiation problems.
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


def get_int(prompt: str) -> int:
    """Get an integer from user input."""
    try:
        return int(input(prompt).strip())
    except ValueError:
        print("[X] Invalid number")
        sys.exit(1)


def interactive_integration():
    """Interactively compute numerical integration."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Numerical Integration")
    print("=" * 60)
    
    print("\nCompute: ∫ₐᵇ f(x) dx")
    
    # Get function
    print("\nEnter function f(x):")
    print("Examples: 'sin(x)', 'x**2', 'exp(x)', '1/(1+x**2)'")
    f_expr = validate_expression(input("> "))
    
    # Get bounds
    a = get_float("Lower bound (a): ")
    b = get_float("Upper bound (b): ")
    
    if a >= b:
        print("[X] Must have a < b")
        sys.exit(1)
    
    # Get number of intervals
    n = get_int("Number of sub-intervals (n): ")
    if n < 1:
        print("[X] Must have n ≥ 1")
        sys.exit(1)
    
    # Choose method
    print("\nSelect integration method:")
    print("  1. Simpson's rule (more accurate)")
    print("  2. Trapezoidal rule (simpler)")
    
    method_choice = input("Choice (1-2): ").strip()
    method = "simpson" if method_choice == "1" else "trapezoidal"
    
    print(f"\nComputing ∫_{a}^{b} {f_expr} dx using {method} rule...")
    print(f"Intervals: {n}")
    
    try:
        result = mathsteps.integrate(f_expr, a=a, b=b, n=n, method=method)
        
        print(f"\n{'=' * 60}")
        print("RESULT")
        print('=' * 60)
        print(f"[OK] Integral ~= {float(result.answer)}")
        print(f"  Method: {result.solver}")
        print(f"  Intervals: {n}")
        print(f"  Verified against SciPy: {result.verified}")

        # Try to compute exact value symbolically
        try:
            import sympy as sp
            x = sp.Symbol('x')
            f = sp.sympify(f_expr)
            exact = sp.integrate(f, (x, a, b))
            exact_float = float(exact)
            error = abs(float(result.answer) - exact_float)
            print(f"\nExact (symbolic): {exact_float:.10f}")
            print(f"Error: {error:.2e}")
        except:
            pass
        
        show_details = input("\nShow computation details? (y/n): ").strip().lower()
        if show_details == 'y' and result.steps:
            print(f"\nComputation steps ({len(result.steps)} total):")
            for i, step in enumerate(result.steps[:5], 1):
                print(f"  {i}. {step.description}")
            if len(result.steps) > 5:
                print(f"  ... and {len(result.steps) - 5} more")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_differentiation():
    """Interactively compute numerical derivatives."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Numerical Differentiation")
    print("=" * 60)
    
    print("\nCompute: f'(x) at a point")
    
    # Get function
    print("\nEnter function f(x):")
    print("Examples: 'sin(x)', 'x**2', 'exp(x)', '1/(1+x**2)'")
    f_expr = validate_expression(input("> "))
    
    # Get point
    x = get_float("Point to evaluate (x): ")
    
    # Get step size
    h = get_float("Step size (h, e.g., 0.001): ")
    if h == 0:
        print("[X] Step size cannot be zero")
        sys.exit(1)
    
    # Choose method
    print("\nSelect differentiation method:")
    print("  1. Central difference (most accurate)")
    print("  2. Forward difference (one-sided)")
    print("  3. Backward difference (one-sided)")
    
    method_choice = input("Choice (1-3): ").strip()
    methods = {"1": "central", "2": "forward", "3": "backward"}
    
    if method_choice not in methods:
        print("[X] Invalid choice")
        sys.exit(1)
    
    method = methods[method_choice]
    
    print(f"\nComputing f'({x}) for f(x) = {f_expr}")
    print(f"Method: {method} difference")
    print(f"Step size: {h}")
    
    try:
        result = mathsteps.differentiate(f_expr, x=x, h=h, method=method)
        
        print(f"\n{'=' * 60}")
        print("RESULT")
        print('=' * 60)
        print(f"[OK] f'({x}) ~= {float(result.answer)}")
        print(f"  Method: {method} difference with h={h}")
        print(f"  Verified against SciPy: {result.verified}")

        # Try to compute exact value symbolically
        try:
            import sympy as sp
            x_sym = sp.Symbol('x')
            f = sp.sympify(f_expr)
            f_prime = sp.diff(f, x_sym)
            exact = float(f_prime.subs(x_sym, x))
            error = abs(float(result.answer) - exact)
            print(f"\nExact (symbolic): {exact:.10f}")
            print(f"Error: {error:.2e}")
        except:
            pass
        
        # Accuracy with different step sizes
        show_convergence = input("\nShow accuracy with different h values? (y/n): ").strip().lower()
        if show_convergence == 'y':
            print(f"\nAccuracy vs step size:")
            h_values = [h * 10, h, h / 10, h / 100]
            print(f"{'h':<15} {'f\'({x:.3f})':<18} {'Error':<15}")
            print("-" * 48)
            
            try:
                import sympy as sp
                x_sym = sp.Symbol('x')
                f = sp.sympify(f_expr)
                f_prime = sp.diff(f, x_sym)
                exact = float(f_prime.subs(x_sym, x))
                
                for h_test in h_values:
                    r_test = mathsteps.differentiate(f_expr, x=x, h=h_test, method=method)
                    error = abs(float(r_test.answer) - exact)
                    print(f"{h_test:<15.6f} {float(r_test.answer):<18.10f} {error:<15.2e}")
            except:
                print("  (Could not compute - symbolic differentiation failed)")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_compare_integration_methods():
    """Compare integration methods."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Compare Integration Methods")
    print("=" * 60)
    
    # Get function
    print("\nEnter function f(x):")
    f_expr = validate_expression(input("> "))
    
    # Get bounds
    a = get_float("Lower bound (a): ")
    b = get_float("Upper bound (b): ")
    
    if a >= b:
        print("[X] Must have a < b")
        sys.exit(1)
    
    # Get number of intervals
    n = get_int("Number of sub-intervals (n): ")
    if n < 1:
        print("[X] Must have n ≥ 1")
        sys.exit(1)
    
    print(f"\nComparing methods for ∫_{a}^{b} {f_expr} dx")
    
    try:
        simpson_result = mathsteps.integrate(f_expr, a=a, b=b, n=n, method="simpson")
        trap_result = mathsteps.integrate(f_expr, a=a, b=b, n=n, method="trapezoidal")
        
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print('=' * 60)
        print(f"{'Method':<20} {'Result':<20} {'Difference':<15}")
        print("-" * 55)
        
        print(f"{'Simpson':<20} {float(simpson_result.answer):<20.10f}")
        print(f"{'Trapezoidal':<20} {float(trap_result.answer):<20.10f}")

        diff = abs(float(simpson_result.answer) - float(trap_result.answer))
        print(f"{'Difference':<20} {diff:<20.2e}")
        
        print(f"\nObservation:")
        if diff < 1e-6:
            print(f"  Methods agree to very high precision")
        else:
            print(f"  Simpson's is typically more accurate than trapezoidal")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def main():
    """Main interactive menu."""
    print("\n" + "=" * 60)
    print("INTERACTIVE NUMERICAL METHODS")
    print("=" * 60)
    
    print("\nChoose problem type:")
    print("  1. Numerical integration (∫ f dx)")
    print("  2. Numerical differentiation (f'(x))")
    print("  3. Compare integration methods")
    print("  4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        interactive_integration()
    elif choice == "2":
        interactive_differentiation()
    elif choice == "3":
        interactive_compare_integration_methods()
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

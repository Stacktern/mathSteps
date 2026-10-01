#!/usr/bin/env python3
"""
Interactive Example: Root Finder

Prompts user for equation, method, and parameters.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps
import sys


def validate_expression(expr: str) -> str:
    """Validate that user provided an expression."""
    expr = expr.strip()
    if not expr:
        raise ValueError("Expression cannot be empty")
    return expr


def get_float(prompt: str, allow_none: bool = False) -> float | None:
    """Get a float from user input."""
    try:
        value = input(prompt).strip()
        if allow_none and value.lower() in ('none', 'skip', ''):
            return None
        return float(value)
    except ValueError:
        print("[X] Invalid number")
        sys.exit(1)


def interactive_root_finder():
    """Interactively find a root of an equation."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Root Finder")
    print("=" * 60)
    
    # Get equation
    print("\nEnter equation (e.g., 'cos(x) - x', 'x**2 - 4', 'sin(x)'):")
    print("Note: Use 'x' as the variable")
    equation = validate_expression(input("> "))
    
    # Choose method
    print("\nSelect root-finding method:")
    print("  1. Newton-Raphson (fast, needs good initial guess)")
    print("  2. Bisection (slow, but robust)")
    print("  3. Secant method (medium, needs two guesses)")
    print("  4. Fixed-point (finds x where x = f(x))")
    
    method_choice = input("Choice (1-4): ").strip()
    
    methods = {
        "1": "newton_raphson",
        "2": "bisection",
        "3": "secant",
        "4": "fixed_point"
    }
    
    if method_choice not in methods:
        print("[X] Invalid method")
        sys.exit(1)
    
    method = methods[method_choice]
    
    print(f"\nSolving f(x) = {equation} using {method}...")
    
    try:
        # Get method-specific parameters
        if method == "newton_raphson" or method == "fixed_point":
            x0 = get_float("Enter initial guess x0: ")
            result = mathsteps.root(equation, method=method, x0=x0)
        
        elif method == "bisection":
            a = get_float("Enter left bracket a: ")
            b = get_float("Enter right bracket b: ")
            result = mathsteps.root(equation, method=method, a=a, b=b)
        
        elif method == "secant":
            x0 = get_float("Enter first guess x0: ")
            x1 = get_float("Enter second guess x1: ")
            result = mathsteps.root(equation, method=method, x0=x0, x1=x1)
        
        # Optional parameters
        tol = get_float("Tolerance (default 1e-10, press Enter to skip): ", allow_none=True)
        max_iter = None
        try:
            max_iter_str = input("Max iterations (default 50, press Enter to skip): ").strip()
            if max_iter_str:
                max_iter = int(max_iter_str)
        except ValueError:
            print("[X] Invalid max_iter")
            sys.exit(1)
        
        # Re-solve with custom parameters if provided
        if tol is not None or max_iter is not None:
            kwargs = {"method": method}
            if method in ["newton_raphson", "fixed_point"]:
                kwargs["x0"] = x0
            elif method == "bisection":
                kwargs["a"] = a
                kwargs["b"] = b
            elif method == "secant":
                kwargs["x0"] = x0
                kwargs["x1"] = x1
            
            if tol is not None:
                kwargs["tol"] = tol
            if max_iter is not None:
                kwargs["max_iter"] = max_iter
            
            result = mathsteps.root(equation, **kwargs)
        
        # Display results
        print(f"\n{'=' * 60}")
        print("RESULT")
        print('=' * 60)
        print(f"[OK] Root found: x = {float(result.answer)}")
        print(f"  Solver used: {result.solver}")
        print(f"  Number of steps: {len(result.steps)}")
        print(f"  Verified against SciPy: {result.verified}")

        # Verify the root
        import sympy as sp
        x = sp.Symbol('x')
        f = sp.sympify(equation)
        f_at_root = float(f.subs(x, result.answer))
        print(f"\nVerification:")
        print(f"  f({float(result.answer):.10f}) = {f_at_root:.2e}")
        print(f"  -> Very close to 0? {abs(f_at_root) < 1e-6}")

        # Show iterations
        show_iterations = input("\nShow iteration details? (y/n): ").strip().lower()
        if show_iterations == 'y':
            print(f"\nIteration breakdown ({len(result.steps)} steps):")
            for i, step in enumerate(result.steps, 1):
                if i <= 5 or i > len(result.steps) - 3:
                    print(f"  {i}. {step.description}")
                elif i == 6:
                    print(f"  ... ({len(result.steps) - 8} more iterations)")
                    break
    
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_compare_methods():
    """Compare different root-finding methods."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Compare Root-Finding Methods")
    print("=" * 60)
    
    equation = validate_expression(input("Enter equation: "))
    
    print("\nWhich methods to compare?")
    print("  1. Newton-Raphson & Bisection")
    print("  2. Newton-Raphson & Secant")
    print("  3. All methods (Newton, Bisection, Secant)")
    
    choice = input("Choice (1-3): ").strip()
    
    print(f"\nComparing methods for f(x) = {equation}...")
    
    try:
        results_data = []
        
        if choice in ["1", "3"]:
            # Newton-Raphson
            print("\nNewton-Raphson: Enter initial guess")
            x0 = get_float("x0: ")
            r = mathsteps.root(equation, method="newton_raphson", x0=x0)
            results_data.append(("Newton-Raphson", r, f"x0={x0}"))
            
            # Bisection
            print("\nBisection: Enter bracket [a, b]")
            a = get_float("a: ")
            b = get_float("b: ")
            r = mathsteps.root(equation, method="bisection", a=a, b=b)
            results_data.append(("Bisection", r, f"[{a}, {b}]"))
        
        if choice in ["2", "3"]:
            if choice != "1":  # Newton already done in option 1
                print("\nNewton-Raphson: Enter initial guess")
                x0 = get_float("x0: ")
                r = mathsteps.root(equation, method="newton_raphson", x0=x0)
                results_data.append(("Newton-Raphson", r, f"x0={x0}"))
            
            # Secant
            print("\nSecant: Enter two guesses")
            x0 = get_float("x0: ")
            x1 = get_float("x1: ")
            r = mathsteps.root(equation, method="secant", x0=x0, x1=x1)
            results_data.append(("Secant", r, f"x0={x0}, x1={x1}"))
        
        # Display comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print('=' * 60)
        print(f"{'Method':<20} {'Root':<20} {'Steps':<10} {'Verified'}")
        print("-" * 60)
        
        for name, result, params in results_data:
            print(f"{name:<20} {float(result.answer):<20.10f} {len(result.steps):<10} {result.verified}")
        
        print(f"\nParameters: {', '.join(p for _, _, p in results_data)}")
        
        # Best method
        best = min(results_data, key=lambda x: len(x[1].steps))
        print(f"\nFastest: {best[0]} ({len(best[1].steps)} steps)")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def main():
    """Main interactive menu."""
    print("\n" + "=" * 60)
    print("INTERACTIVE ROOT FINDER")
    print("=" * 60)
    
    print("\nChoose task:")
    print("  1. Find a root (single method)")
    print("  2. Compare methods (side-by-side)")
    print("  3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        interactive_root_finder()
    elif choice == "2":
        interactive_compare_methods()
    elif choice == "3":
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

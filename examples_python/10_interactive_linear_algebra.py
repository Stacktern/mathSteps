#!/usr/bin/env python3
"""
Interactive Example: Linear Algebra Solver

Prompts user for input and solves systems of equations interactively.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import mathsteps
import sys


def parse_matrix_input(prompt: str) -> list[list[float]]:
    """Parse matrix from user input format: '1 2; 3 4'."""
    print(prompt)
    print("  Format: rows separated by semicolon, columns by space")
    print("  Example: 1 2; 3 4")
    
    try:
        user_input = input("> ").strip()
        rows = user_input.split(";")
        matrix = []
        for row in rows:
            matrix.append([float(x.strip()) for x in row.split()])
        
        # Validate matrix is rectangular
        if not matrix:
            raise ValueError("Empty matrix")
        row_len = len(matrix[0])
        if not all(len(row) == row_len for row in matrix):
            raise ValueError("All rows must have same number of columns")
        
        return matrix
    except ValueError as e:
        print(f"[X] Invalid input: {e}")
        sys.exit(1)


def parse_vector_input(prompt: str, expected_len: int = None) -> list[float]:
    """Parse vector from user input format: '5 11'."""
    print(prompt)
    print("  Format: space-separated numbers")
    print("  Example: 5 11")
    
    try:
        user_input = input("> ").strip()
        vector = [float(x.strip()) for x in user_input.split()]
        
        if expected_len and len(vector) != expected_len:
            raise ValueError(f"Expected {expected_len} elements, got {len(vector)}")
        
        return vector
    except ValueError as e:
        print(f"[X] Invalid input: {e}")
        sys.exit(1)


def interactive_linear_system():
    """Interactively solve a linear system."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Solve Linear System (Ax = b)")
    print("=" * 60)
    
    print("\nEnter coefficient matrix A:")
    A = parse_matrix_input("A = ")
    
    n = len(A)  # Number of equations
    print(f"\nEnter right-hand side vector b ({n} elements):")
    b = parse_vector_input("b = ", expected_len=n)
    
    print(f"\nSolving system with {n} equations...")
    try:
        result = mathsteps.linear_system(A, b)
        
        print(f"\n✓ Solution found using: {result.solver}")
        print(f"  Verified against NumPy: {result.verified}")
        print(f"\nAnswer: x = {result.answer}")
        
        # Show verification
        import numpy as np
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)
        x_np = np.array(result.answer, dtype=float)
        residual = np.linalg.norm(A_np @ x_np - b_np)
        print(f"Residual ||Ax - b||: {residual:.2e}")
        
        # Show steps
        show_steps = input("\nShow step-by-step solution? (y/n) ").strip().lower()
        if show_steps == 'y':
            print("\nStep-by-step solution:")
            for i, step in enumerate(result.steps, 1):
                print(f"\n  Step {i}: {step.description}")
                if step.before and step.before != step.after:
                    print(f"    Before: {step.before}")
                    print(f"    After:  {step.after}")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def interactive_matrix_operation():
    """Interactively compute matrix operations."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Matrix Operations")
    print("=" * 60)
    
    print("\nChoose operation:")
    print("  1. Determinant")
    print("  2. Matrix Inverse")
    print("  3. LU Decomposition")
    print("  4. Eigenvalues")
    
    choice = input("Choice (1-4): ").strip()
    
    print("\nEnter square matrix:")
    A = parse_matrix_input("A = ")
    
    # Check if square
    if len(A) != len(A[0]):
        print("[X] Matrix must be square for this operation")
        sys.exit(1)
    
    try:
        if choice == "1":
            result = mathsteps.determinant(A)
            print(f"\n[OK] det(A) = {result.answer}")
            print(f"  Verified: {result.verified}")
            
        elif choice == "2":
            result = mathsteps.inverse(A)
            print(f"\n✓ A^(-1) =")
            inv_array = result.answer.tolist()
            for row in inv_array:
                print(f"  {row}")
            print(f"  Verified: {result.verified}")
            
        elif choice == "3":
            result = mathsteps.lu(A)
            P, L, U = result.answer
            print(f"\n✓ LU Decomposition: P @ A = L @ U")
            print(f"  Found by: {result.solver}")
            print(f"  Verified: {result.verified}")
            
        elif choice == "4":
            result = mathsteps.eigenvalues(A)
            eigs, eigvecs = result.answer
            print(f"\n✓ Eigenvalues: {eigs}")
            print(f"  Verified: {result.verified}")
        
        else:
            print("[X] Invalid choice")
            sys.exit(1)
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def interactive_cramer_rule():
    """Interactively solve using Cramer's rule."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Solve Using Cramer's Rule")
    print("=" * 60)
    
    print("\nEnter coefficient matrix A:")
    A = parse_matrix_input("A = ")
    
    n = len(A)
    print(f"\nEnter right-hand side vector b ({n} elements):")
    b = parse_vector_input("b = ", expected_len=n)
    
    print(f"\nSolving using Cramer's rule...")
    try:
        result = mathsteps.cramers_rule(A, b)
        
        print(f"\n[OK] Solution: x = {result.answer}")
        print(f"  Verified: {result.verified}")
        
        # Show determinants used
        print(f"\nSteps in Cramer's rule ({len(result.steps)} total):")
        for i, step in enumerate(result.steps, 1):
            if i <= 5 or i > len(result.steps) - 2:
                print(f"  {i}. {step.description}")
            elif i == 6:
                print(f"  ... ({len(result.steps) - 6} more steps)")
    
    except Exception as e:
        print(f"[X] Error: {e}")
        sys.exit(1)


def main():
    """Main interactive menu."""
    print("\n" + "=" * 60)
    print("INTERACTIVE LINEAR ALGEBRA SOLVER")
    print("=" * 60)
    
    print("\nChoose problem type:")
    print("  1. Solve Ax = b (Gaussian elimination)")
    print("  2. Matrix operations (det, inverse, LU, eigenvalues)")
    print("  3. Solve Ax = b (Cramer's rule)")
    print("  4. Exit")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        interactive_linear_system()
    elif choice == "2":
        interactive_matrix_operation()
    elif choice == "3":
        interactive_cramer_rule()
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

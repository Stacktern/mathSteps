#!/usr/bin/env python3
"""
Quick Test Script: Validate mathsteps Installation

Run this after `pip install mathsteps` to verify everything works.
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def test_import():
    """Test that mathsteps can be imported."""
    try:
        import mathsteps
        print("[OK] mathsteps imported successfully")
        print(f"  Version: {mathsteps.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import mathsteps: {e}")
        return False


def test_linear_system():
    """Test linear system solving."""
    try:
        import mathsteps
        result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
        answer = [float(v) for v in result.answer]
        assert answer == [1.0, 2.0], f"Wrong answer: {result.answer}"
        assert result.verified, "Answer not verified"
        print("[OK] Linear system solving works")
        return True
    except Exception as e:
        print(f"[FAIL] Linear system test failed: {e}")
        return False


def test_root_finding():
    """Test root finding."""
    try:
        import mathsteps
        result = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
        assert abs(float(result.answer) - 0.7390851332151607) < 1e-6
        assert result.verified, "Root not verified"
        print("[OK] Root finding works")
        return True
    except Exception as e:
        print(f"[FAIL] Root finding test failed: {e}")
        return False


def test_ivp():
    """Test ODE solving."""
    try:
        import mathsteps
        result = mathsteps.ivp("-2*x*y", y0=1, x_end=1, h=0.1, method="rk4")
        import math
        expected = math.exp(-1)  # e^(-1)
        assert abs(float(result.answer) - expected) < 0.01, f"Wrong answer: {result.answer}"
        print("[OK] ODE solving works")
        return True
    except Exception as e:
        print(f"[FAIL] IVP test failed: {e}")
        return False


def test_determinant():
    """Test determinant calculation."""
    try:
        import mathsteps
        result = mathsteps.determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        assert int(result.answer) == -306, f"Wrong det: {result.answer}"
        assert result.verified, "Determinant not verified"
        print("[OK] Determinant calculation works")
        return True
    except Exception as e:
        print(f"[FAIL] Determinant test failed: {e}")
        return False


def test_result_structure():
    """Test Result object structure."""
    try:
        import mathsteps
        result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])

        # Check all attributes exist
        assert hasattr(result, 'solver'), "Missing 'solver' attribute"
        assert hasattr(result, 'steps'), "Missing 'steps' attribute"
        assert hasattr(result, 'answer'), "Missing 'answer' attribute"
        assert hasattr(result, 'verified'), "Missing 'verified' attribute"
        assert hasattr(result, 'problem'), "Missing 'problem' attribute"

        # Check types
        assert isinstance(result.solver, str), "'solver' should be string"
        assert isinstance(result.steps, list), "'steps' should be list"
        assert isinstance(result.verified, bool), "'verified' should be bool"
        assert isinstance(result.problem, dict), "'problem' should be dict"

        # Check steps have required attributes
        for step in result.steps:
            assert hasattr(step, 'description'), "Step missing 'description'"
            assert hasattr(step, 'before'), "Step missing 'before'"
            assert hasattr(step, 'after'), "Step missing 'after'"
            assert hasattr(step, 'data'), "Step missing 'data'"

        print("[OK] Result structure correct")
        return True
    except AssertionError as e:
        print(f"[FAIL] Result structure test failed: {e}")
        return False


def test_available_solvers():
    """Test that we can list available solvers."""
    try:
        import mathsteps
        solvers = mathsteps.available_solvers()
        assert isinstance(solvers, list), f"available_solvers should return list, got {type(solvers)}"
        assert len(solvers) > 0, "No solvers registered"
        print(f"[OK] Available solvers: {len(solvers)} registered")
        return True
    except Exception as e:
        print(f"[FAIL] Available solvers test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MathSteps Installation Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Import", test_import),
        ("Linear System", test_linear_system),
        ("Root Finding", test_root_finding),
        ("ODE Solving", test_ivp),
        ("Determinant", test_determinant),
        ("Result Structure", test_result_structure),
        ("Available Solvers", test_available_solvers),
    ]

    results = []
    for name, test_func in tests:
        results.append(test_func())
        print()

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("[OK] All tests passed! mathsteps is ready to use.")
        print("\nNext steps:")
        print("  1. Run example scripts: python examples_python/01_linear_algebra.py")
        print("  2. Read PYTHON_GUIDE.md for comprehensive usage guide")
        print("  3. Try the CLI: mathsteps --help")
        return 0
    else:
        print("[FAIL] Some tests failed. Check installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

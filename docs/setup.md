# Installation & Setup Guide

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **pip** package manager (comes with Python)

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install mathsteps
```

That's it! You now have:
- ✅ CLI command: `mathsteps`
- ✅ Python library: `import mathsteps`
- ✅ All dependencies installed

### Option 2: Install from Source (Development)

```bash
# Clone the repository
git clone <repository-url>
cd mathsteps

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Verification

### Quick Check

```bash
# Check it's installed
python -c "import mathsteps; print(mathsteps.__version__)"
```

Expected output:
```
0.1.0
```

### Run Test Suite

```bash
# Full test suite
pytest tests/ -v

# Quick smoke tests
python examples_python/00_test_installation.py
```

### Try the CLI

```bash
# Linear system
mathsteps linear-system --A "1 2; 3 4" --b "5 11"

# Root finding
mathsteps root --function "x**3 - x - 2" --method newton --x0 1.5

# ODE
mathsteps ivp --f-expr "y' = -2*x*y" --y0 1 --x-end 2 --h 0.1 --method rk4
```

### Try Python Library

```python
import mathsteps

# Linear system
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
print(result.answer)  # [1. 2.]

# Root finding
result = mathsteps.root("cos(x) - x", method="newton", x0=0.0)
print(result.answer)  # 0.7390851332151607

# ODE
result = mathsteps.ivp("-2*x*y", y0=1, x_end=2, h=0.1, method="rk4")
print(result.answer)  # ~0.0183 (e^-4)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'mathsteps'"

**Problem**: mathsteps not found after installation.

**Solution**:
```bash
# Reinstall
pip uninstall mathsteps
pip install --upgrade mathsteps

# Verify Python version
python --version  # Should be 3.10+

# Check pip is using correct Python
pip --version
```

### "mathsteps: command not found" (CLI)

**Problem**: CLI command not available.

**Solution**:
```bash
# Reinstall with pip
pip install --force-reinstall mathsteps

# Check installation location
pip show -f mathsteps | grep Location

# On Windows: add to PATH if needed
# Run: pip show mathsteps (note the Location path)
```

### Import Error for Dependencies (numpy, scipy, sympy)

**Problem**: Missing dependencies like numpy, scipy, sympy.

**Solution**:
```bash
# Reinstall with all dependencies
pip install --force-reinstall mathsteps

# Or install dependencies separately
pip install numpy scipy sympy typer rich
```

### "AttributeError: module has no attribute..." when using API

**Problem**: Function doesn't exist in mathsteps.

**Solution**:
```python
# Check what's available
import mathsteps
print(mathsteps.available_solvers())
print(dir(mathsteps))

# Ensure you imported correctly
from mathsteps import linear_system, root, ivp  # specific imports
# or
import mathsteps  # general import, use as mathsteps.linear_system()
```

### Tests Fail

**Problem**: Some pytest tests fail.

**Solution**:
```bash
# Check Python version (need 3.10+)
python --version

# Install dev dependencies
pip install -e ".[dev]"

# Run tests with verbose output
pytest tests/ -v --tb=short
```

## Environment Setup (Virtual Environment)

### Windows

```bash
# Create virtual environment
python -m venv mathsteps_env

# Activate
mathsteps_env\Scripts\activate

# Install
pip install mathsteps

# When done, deactivate
deactivate
```

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv mathsteps_env

# Activate
source mathsteps_env/bin/activate

# Install
pip install mathsteps

# When done
deactivate
```

## Upgrading

```bash
# Check current version
pip show mathsteps | grep Version

# Upgrade to latest
pip install --upgrade mathsteps

# Check new version
python -c "import mathsteps; print(mathsteps.__version__)"
```

## Next Steps

After successful installation:

1. **Read the README**: `README.md` - Overview and quick start
2. **Try examples**: `examples_python/` - Working code examples
3. **Read API guide**: [python-guide.md](python-guide.md) - Complete API reference
4. **Try CLI**: `mathsteps --help` - Command-line interface

## Getting Help

- **API Documentation**: See [python-guide.md](python-guide.md)
- **CLI Help**: `mathsteps --help` or `mathsteps <command> --help`
- **Issues**: open an issue on the project's GitHub repository
- **Examples**: `examples/` (JSON) and `examples_python/` (Python scripts)

## System Requirements

| System | Python | Status |
|--------|--------|--------|
| Windows 10/11 | 3.10, 3.11, 3.12, 3.13 | ✅ Supported |
| macOS (Intel) | 3.10, 3.11, 3.12, 3.13 | ✅ Supported |
| macOS (Apple Silicon) | 3.10, 3.11, 3.12, 3.13 | ✅ Supported |
| Linux (x86_64) | 3.10, 3.11, 3.12, 3.13 | ✅ Supported |

## Common Workflows

### Scientific Notebook Environment (Jupyter)

```bash
# Install with Jupyter support
pip install mathsteps jupyter

# Start Jupyter
jupyter notebook

# In a cell:
import mathsteps
result = mathsteps.linear_system([[1, 2], [3, 4]], [5, 11])
print(result)
```

### Scripting Environment

```bash
# Create a script
cat > solve_system.py << 'EOF'
import mathsteps

A = [[2, 1], [3, 4]]
b = [5, 11]

result = mathsteps.linear_system(A, b)
print(f"Solution: {result.answer}")
print(f"Verified: {result.verified}")
EOF

# Run it
python solve_system.py
```

### Production Application

```bash
# Install with specific version pinning
pip install mathsteps==0.1.0

# Use in your application
from mathsteps import linear_system, root, ivp
# ... your code ...
```

## Performance Notes

- **First import** ~500ms (due to SymPy initialization)
- **Solving** speed depends on problem size and method
- **Verification** adds ~10-50% overhead but catches bugs

## Uninstall

```bash
pip uninstall mathsteps
```

This removes mathsteps but keeps dependencies (numpy, scipy, etc).

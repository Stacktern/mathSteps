# Contributing to mathsteps

Thanks for your interest! `mathsteps` is built around one simple
invariant — **every solver returns `(list[Step], final_answer)` and
auto-registers on import.** Adding a new solver is a matter of
implementing that interface and dropping a file in
`mathsteps/domains/<your_domain>/`.

## Quickstart

```bash
git clone https://github.com/OWNER/mathsteps.git   # replace with the repository URL
cd mathsteps
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS / Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Adding a new solver

1. **Pick the right domain.** If your problem is solving a linear
   system, root-finding, IVP, BVP, etc., put it under the matching
   subdirectory of `mathsteps/domains/`. If none fits, create a new
   subdirectory and an `__init__.py`.

2. **Implement the `Solver` interface.** Every solver is a class
   inheriting from `mathsteps.core.solver_base.Solver`:

   ```python
   from mathsteps.core.registry import register
   from mathsteps.core.solver_base import Solver
   from mathsteps.core.step import Step

   @register
   class MyAwesomeSolver(Solver):
       name = "my_awesome_solver"

       def can_solve(self, problem: dict) -> bool:
           return problem.get("type") == "my_problem_type"

       def solve(self, problem: dict) -> tuple[list[Step], Any]:
           steps = []
           # ... do the math, append one Step per transformation ...
           return steps, final_answer
   ```

   The `@register` decorator appends the solver to the global
   registry; `find_solver(problem)` walks the registry and picks the
   first solver whose `can_solve` returns `True`.

3. **Register the import.** Add the module to the import block in
   `mathsteps/__init__.py` (and `mathsteps/cli.py`) so the `@register`
   side effect runs. **Order matters only when two solvers can solve
   the same problem type** — first match wins.

4. **Return native answers, then convert.** Solvers return their
   natural (often exact SymPy) answer. Add a branch for your problem
   type to `finalize()` in `mathsteps/core/answers.py` so the public
   `Result.answer` is a plain `float` / `numpy` array and the exact
   form goes to `Result.exact`. Extra outputs (a grid, a trajectory)
   go in the final step's `data["details"]` and surface as
   `Result.details`. An iterative solver records
   `data["converged"] = True/False` on its final step.

5. **Verify your work.** Add a `verify_my_problem(...)` helper in
   `mathsteps/verify.py` against an *independent* NumPy / SciPy /
   SymPy reference and dispatch to it from `_verify_problem`. Return
   `True` / `False`; if no independent reference exists, leave the
   problem type out so it reports `None` ("not checked") — never
   report a pass you did not earn. For numerical methods derive the
   tolerance from the method's truncation error (see
   `verify_integration`).

   If your solver can emit many steps, honour `problem["detail"]` with
   `mathsteps/core/detail.py` (`resolve_detail`, `keep_step`).

6. **Add an example.** Drop a `examples/<your_solver>.json` file
   with a representative problem.

7. **Write tests.** Add a `tests/test_<your_solver>.py` covering
   happy path, edge cases, and verification.

8. **Run the test suite.**

   ```bash
   python -m pytest
   ```

## Conventions

- **One Step per transformation.** Don't batch. Each `Step` should
  correspond to a single algorithmic step the user could write down
  by hand.
- **Exact arithmetic where it is cheap, floats where it is not.**
  Linear algebra and interpolation run on `fractions.Fraction`
  (`mathsteps/core/exact.py`); iterative methods run on floats. Avoid
  SymPy's `nsimplify` inside loops — it is slow and can rewrite an
  approximate number as an unrelated closed form.
- **Always set `data`.** The `Step.data` dict carries structured
  payload (iteration index, intermediate values, swap pairs). It's
  what makes the steps machine-checkable, not just human-readable.
- **Validate inputs.** Raise `ValueError` with a descriptive
  message; the CLI will surface it.
- **Unicode caution.** Windows consoles default to cp1252. Avoid
  Δ, λ, ≈ in step descriptions unless you're going to render with
  `--pretty` (which uses `sympy.latex` + Rich). The current solvers
  use ASCII substitutes (`delta x`, `lambda`, `~=`).

## Style

- Python 3.10+, type hints everywhere.
- Parse user expressions with `mathsteps.core.expr.parse_expr` (it rejects
  unknown symbols with a clear message) and guard evaluated values with
  `finite_float`.
- `from __future__ import annotations` at the top of every file.
- No comments unless they're clarifying *why*, not *what*.

## Reporting bugs

Open an issue with: a minimal problem JSON, the actual output, the
expected output, and the version of `numpy` / `scipy` / `sympy`
installed (`python -m pip freeze`).

## License

By contributing you agree your contributions are licensed under the
project's MIT license.

# Building a Step-by-Step Symbolic Solution Tool

A guide for building a tool that takes a mathematical expression (equation, derivative, integral, simplification, etc.) and shows the **intermediate reasoning steps** — not just the final answer — similar to Symbolab or Mathway.

---

## 1. Why This Is Hard

Most CAS libraries (SymPy included) are built to compute a *final result* efficiently. They don't naturally expose *how* they got there — internally they apply dozens of rewrite rules, often in a different order than a human would, and don't log "steps" by default.

So this project has two real components:
1. **The math engine** (you don't build this — you use SymPy)
2. **The step-tracking/explanation layer** (this is the actual project — you build this)

Understanding this split early will save you a lot of wasted effort trying to make SymPy itself "explain" things it isn't designed to explain.

---

## 2. Core Approaches (pick one to start)

### Approach A: Rule-based manual stepper (recommended starting point)
You implement a small set of **your own transformation rules** for a narrow domain (e.g., solving linear equations, or basic differentiation), and manually track each rule application as a "step." You use SymPy only to verify correctness of intermediate expressions, not to generate the steps themselves.

- **Pros:** Full control over step granularity and wording; realistic scope for a solo/GSoC-sized project
- **Cons:** Only covers the domain you implement (e.g., won't generalize to arbitrary integrals)

### Approach B: Leverage SymPy's `refine`, `manualintegrate`, and stepwise internals
SymPy has some existing manual/steppable machinery worth knowing about:
- `sympy.integrals.manualintegrate` — implements integration the way a *human* would (u-substitution, integration by parts, etc.) rather than the fast internal Risch/heuristic algorithms. This is the closest existing thing to what you want, and studying its source is valuable even if you don't reuse it directly.
- `sympy.solvers` has some verbose/logging options in places, but nothing as clean as `manualintegrate`.

- **Pros:** Piggybacks on real, tested logic; a legitimate area for a SymPy contribution (this is close to real GSoC project territory)
- **Cons:** Steeper learning curve — you're reading and extending SymPy internals, not writing your own from scratch

### Approach C: Wrap an LLM to narrate steps
Use SymPy to compute the correct final answer (ground truth), then prompt an LLM to generate a step-by-step explanation *constrained* to match that answer.

- **Pros:** Fast to prototype, works across many domains without hand-coding each one
- **Cons:** LLM steps can be subtly wrong or skip steps even if the final answer matches; needs strong verification against SymPy's ground truth at each step

**Recommendation:** Start with Approach A on a narrow domain (e.g., solving linear/quadratic equations, or basic derivative rules) to build something working end-to-end quickly. Study Approach B's `manualintegrate` source afterward — it's the best real-world example of "steppable" symbolic math and could turn into an actual contribution.

---

## 3. Suggested Build Order

### Phase 1: Pick one narrow domain
Don't try to handle "all of math" at once. Good first domains, easiest to hardest:
1. Simplifying polynomial expressions (expand, combine like terms)
2. Solving linear equations (`3x + 5 = 20`)
3. Basic differentiation (power rule, product rule, chain rule — one at a time)
4. Solving quadratics (factoring, then completing the square, then the quadratic formula as fallback)

### Phase 2: Design your step data structure
Each step should capture:
```python
{
    "expression_before": "3*x + 5",
    "expression_after": "3*x",
    "rule_applied": "Subtract 5 from both sides",
    "explanation": "Isolate the term with x by moving the constant to the other side."
}
```
Use SymPy expressions internally (`sympify`, `Eq`, etc.) so each step is a *real*, verifiable mathematical object — not just a string. This lets you sanity-check that `expression_before` actually transforms into `expression_after` under valid algebra, catching bugs in your own rule logic.

### Phase 3: Implement rule functions
Each rule is a small function: takes an expression, returns `(new_expression, explanation)` or `None` if the rule doesn't apply. Example for a linear equation solver:

```python
def isolate_variable_addition(eq, var):
    """Move constant term from the variable's side to the other side."""
    lhs, rhs = eq.lhs, eq.rhs
    const_term = [t for t in lhs.as_ordered_terms() if not t.has(var)]
    if const_term:
        c = const_term[0]
        new_eq = Eq(lhs - c, rhs - c)
        return new_eq, f"Subtract {c} from both sides"
    return None
```

### Phase 4: Chain rules into a solver loop
Apply rules in sequence until the expression reaches a terminal/solved form, logging each transformation as a step.

### Phase 5: Verification layer (important — don't skip)
After generating all steps, **always double check** the final step's result against SymPy's own `solve()`/`diff()`/`integrate()` output. If they don't match, your rule logic has a bug. This catches silent errors before a user ever sees a wrong step-by-step explanation.

### Phase 6: Output/display layer
- Command-line: print each step in order with LaTeX via SymPy's `latex()` function
- Web: render each step's LaTeX with MathJax/KaTeX, one step revealed at a time (or all at once with a "show steps" toggle)
- Notebook: use `display()` + `Math()` from `IPython.display` for step-by-step rendering directly in Jupyter

---

## 4. Key SymPy Functions You'll Rely On

| Function | Purpose |
|---|---|
| `sympify()` | Convert a string into a real SymPy expression |
| `Eq()` | Represent an equation as a proper object, not just `lhs - rhs` |
| `as_ordered_terms()` | Break an expression into its additive terms |
| `simplify()`, `expand()`, `factor()` | Use as verification, not as "the step" itself |
| `latex()` | Convert any expression to LaTeX for nice rendering |
| `sympy.integrals.manualintegrate` | Reference implementation for step-based integration |
| `srepr()` | See the raw internal tree structure of an expression — useful for debugging your rule-matching logic |

---

## 5. Realistic Scope Warning

Full generalized step-by-step solving (matching something like Symbolab across all of algebra, calculus, and beyond) is a multi-year effort by a funded team. For a solo project or a GSoC-sized proposal:

- Pick **one** domain (e.g., "step-by-step linear and quadratic equation solving") and do it well
- A working, well-tested narrow tool is a far stronger portfolio piece and GSoC proposal than a half-working broad one
- If you go the SymPy-contribution route (Approach B), consider proposing to **extend `manualintegrate`** with new techniques, or building an analogous `manualsolve`/`manualdiff` module — this is a realistic, scoped GSoC-style proposal

---

## 6. Suggested Next Steps

1. Install SymPy (already done) and read through `sympy/integrals/manualintegrate.py` on GitHub to see a real "steppable" implementation
2. Pick your starting domain (recommend: linear equations — smallest scope, fastest to get working end-to-end)
3. Build the step data structure and 3–4 rule functions
4. Get a basic CLI version working before touching any UI/rendering
5. Add the verification layer against `sympy.solve()`
6. Only then consider LaTeX/notebook/web rendering

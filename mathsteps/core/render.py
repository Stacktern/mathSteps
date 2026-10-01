"""Pretty rendering of Step objects.

Converts step descriptions and ``before``/``after`` payloads into
LaTeX (when they look like SymPy expressions) and renders the result
with the ``rich`` library for colored terminal output.

This module is a *display* concern; it does not affect verification
or solver logic.
"""
from __future__ import annotations

import re
from typing import Iterable

import sympy as sp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mathsteps.core.step import Step


_SYM_HINTS = re.compile(
    r"\b(sin|cos|tan|log|exp|sqrt|pi|E)\b|\*\*|[a-zA-Z]\w*\s*\(|[+\-*/^]"
)


def maybe_latex(s: str) -> str:
    """Best-effort: if ``s`` parses as a SymPy expression, return LaTeX."""
    if not s or not _SYM_HINTS.search(s):
        return s
    try:
        expr = sp.sympify(s)
        if expr.free_symbols:
            return "$" + sp.latex(expr) + "$"
        return sp.latex(expr)
    except Exception:
        return s


def render_steps(steps: Iterable[Step], console: Console | None = None) -> None:
    """Print ``steps`` to ``console`` (default: a fresh ``Console``)."""
    if console is None:
        console = Console()
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Description", style="cyan", ratio=3)
    table.add_column("Before → After", ratio=4)

    for i, step in enumerate(steps, 1):
        before = maybe_latex(step.before)
        after = maybe_latex(step.after)
        transition = f"{before} → {after}" if before or after else ""
        table.add_row(str(i), step.description, transition)

    console.print(table)


def render_panel(title: str, body: str, console: Console | None = None) -> None:
    if console is None:
        console = Console()
    console.print(Panel(body, title=title, border_style="green"))

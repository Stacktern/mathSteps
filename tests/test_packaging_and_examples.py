"""Things a release must not break: example scripts, entry points, version string."""
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mathsteps

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SCRIPTS = sorted((ROOT / "examples_python").glob("*.py"))


def _run(*args, **kwargs):
    # Works whether or not the package is installed: put the source tree on PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, encoding="utf-8",
        check=False, cwd=ROOT, env=env, **kwargs,
    )


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_every_example_script_compiles(script, tmp_path):
    py_compile.compile(str(script), cfile=str(tmp_path / "x.pyc"), doraise=True)


@pytest.mark.parametrize(
    "script",
    [s for s in EXAMPLE_SCRIPTS if not s.name[:2].isdigit() or int(s.name[:2]) < 10],
    ids=lambda p: p.name,
)
def test_non_interactive_examples_run(script):
    r = _run("-W", "ignore", str(script))
    assert r.returncode == 0, r.stderr[-500:]


@pytest.mark.parametrize(
    "script",
    [s for s in EXAMPLE_SCRIPTS if s.name[:2].isdigit() and int(s.name[:2]) >= 10],
    ids=lambda p: p.name,
)
def test_interactive_examples_fail_cleanly_on_empty_input(script):
    r = _run("-W", "ignore", str(script), input="1\n" + "\n" * 40)
    assert "Traceback" not in r.stderr and "SyntaxError" not in r.stderr
    assert r.returncode in (0, 1)


def test_version_is_consistent_everywhere():
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc.]\w*)?", mathsteps.__version__)
    for entry in (["-m", "mathsteps", "--version"], ["-m", "mathsteps.cli", "--version"]):
        r = _run(*entry)
        assert r.returncode == 0 and r.stdout.strip() == f"mathsteps {mathsteps.__version__}"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"[{mathsteps.__version__}]" in changelog


def test_python_dash_m_mathsteps_runs_a_problem():
    r = _run("-m", "mathsteps", "determinant", "--A", "1 2; 3 4")
    assert r.returncode == 0 and "-2" in r.stdout and "Verification: PASS" in r.stdout


def test_py_typed_marker_ships():
    assert (Path(mathsteps.__file__).parent / "py.typed").is_file()


def test_no_placeholder_urls_in_project_metadata():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "github.com/mathsteps" not in text and "readthedocs" not in text

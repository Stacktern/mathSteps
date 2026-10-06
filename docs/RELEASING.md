# Branching and releasing

## Branches

| Branch | Purpose | What runs |
|---|---|---|
| `dev` | Integration. Feature branches are merged here. | CI (tests on Linux for Python 3.10-3.13, Windows and macOS for 3.12, plus the oldest supported dependencies) |
| `staging` | Release candidate. Merge `dev` here when it is ready to ship. | CI, then a dry-run publish to **TestPyPI** |
| `main` | Released code only. Merge `staging` here once the TestPyPI build checks out. | CI; a GitHub Release on `main` publishes to **PyPI** |

Flow: `feature/*` -> `dev` -> `staging` -> `main`. Fix a bug found on
`staging` on a branch from `dev`, merge it to `dev`, then re-merge to `staging`.

Protect `main` and `staging` on GitHub (Settings -> Branches): require pull
requests and passing CI checks before merging.

## One-time setup

1. Create the GitHub repository and push the three branches
   (`git push -u origin main dev staging`).
2. Fill in `[project.urls]` and the author in `pyproject.toml`.
3. On <https://test.pypi.org> and <https://pypi.org> add a **trusted publisher**
   (Account -> Publishing) for this repository:
   * workflow: `publish.yml`
   * environment: `testpypi` (TestPyPI) / `pypi` (PyPI)
4. In the GitHub repository create the two environments, `testpypi` and `pypi`
   (Settings -> Environments). Add required reviewers to `pypi` if you want a
   manual approval before real publishing.

## Cutting a release

1. On `dev`: bump `__version__` in `mathsteps/__init__.py` and move the entries in
   `CHANGELOG.md` from *Unreleased* under the new version heading.
2. Merge `dev` into `staging`. The workflow publishes to TestPyPI. Check it:

   ```bash
   python -m venv /tmp/t && /tmp/t/bin/pip install \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ mathsteps==X.Y.Z
   /tmp/t/bin/python -c "import mathsteps; print(mathsteps.__version__)"
   ```

3. Merge `staging` into `main`, then create a GitHub Release tagged `vX.Y.Z` on
   `main`. The workflow builds and publishes to PyPI.

A version can be uploaded to PyPI only once; if a release is bad, publish a new
patch version (and "yank" the bad one on PyPI) rather than trying to replace it.

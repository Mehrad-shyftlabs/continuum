# Versioning

How Continuum's version number is defined, exposed at runtime, and released.

## TL;DR

- **One source of truth:** the `version` field in [`pyproject.toml`](../pyproject.toml).
- **Runtime access:** `continuum.__version__` is read back from the installed
  package metadata — it is **not** a second hardcoded copy.
- **Scheme:** [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — `MAJOR.MINOR.PATCH`.
- Installs use the distribution name `shyftlabs-continuum`; imports use `continuum`.

## Single source of truth

The version lives in exactly one place:

```toml
# pyproject.toml
[project]
name = "shyftlabs-continuum"
version = "0.2.2"
```

When the package is built (`python -m build`), setuptools copies that value into
the distribution metadata (`METADATA` / `PKG-INFO`). That metadata is what `pip`
reads — so `pip show shyftlabs-continuum` and the PyPI release page always reflect
`pyproject.toml`.

## How `continuum.__version__` works

Rather than duplicating the number in code, `src/continuum/__init__.py` reads it
back from the installed metadata at import time:

```python
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("shyftlabs-continuum")
except Exception:  # source checkout that was never installed
    __version__ = "0.0.0+unknown"
```

- `importlib.metadata.version("shyftlabs-continuum")` looks up the **distribution**
  name (note: the hyphenated install name, not the `continuum` import name) and
  returns whatever version was installed.
- Because the value comes from the same metadata `pip` uses, `continuum.__version__`
  is **guaranteed** to equal the installed distribution version. They cannot drift.
- The `except` branch only triggers when the package is imported from a source tree
  that was never installed (no metadata present). In normal use — `pip install`
  or `pip install -e .` — the lookup succeeds.

### Why not just hardcode `__version__ = "x.y.z"`?

That is exactly what caused the bug fixed in `0.2.2`: `0.2.1` shipped with a
hardcoded `__version__ = "0.2.0"` because the literal was forgotten during the
bump. `pip` reported `0.2.1` (from metadata) while `continuum.__version__`
reported `0.2.0` (the stale literal). Deriving from metadata removes the second
copy entirely, so a forgotten edit is impossible.

## Releasing a new version

`dev` is branch-protected, so every change lands via pull request. PyPI versions
are **immutable** — a number can never be re-uploaded — so each release needs a
fresh version.

### Automated release (recommended)

Publishing is automated with **GitHub Actions + PyPI Trusted Publishing**
([`.github/workflows/release.yml`](../.github/workflows/release.yml)). No API
token is stored anywhere — GitHub mints a short-lived OIDC token at publish time.

**One-time setup** (PyPI → project → *Settings* → *Publishing* → *Add a trusted
publisher*):

| Field | Value |
|-------|-------|
| Owner | `shyftlabs` |
| Repository | `continuum` |
| Workflow name | `release.yml` |
| Environment | `pypi` |

**Each release:**

1. Land the version bump + CHANGELOG on `dev` via PR (steps 1–3 below).
2. Create a GitHub Release with a tag matching the version, e.g. `v0.2.2`:
   ```bash
   git switch dev && git pull
   git tag v0.2.2 && git push origin v0.2.2
   ```
   then publish a Release from that tag in the GitHub UI (or `gh release create v0.2.2 --generate-notes`).
3. The workflow runs automatically: it **verifies the tag matches `pyproject.toml`**,
   builds, runs `twine check`, and publishes to PyPI via OIDC.

The version guard means a `v0.2.3` tag on code still at `0.2.2` fails the build
instead of publishing a mismatched release.

### Manual release (fallback)

If you publish by hand instead of via the workflow:

1. **Bump** `version` in `pyproject.toml` (only this one place).
2. **Add a CHANGELOG entry** under a new `## [x.y.z]` heading in
   [`CHANGELOG.md`](../CHANGELOG.md).
3. **Open a PR** to `dev` and merge it.
4. **Build from merged `dev`:**
   ```bash
   git switch dev && git pull
   rm -rf dist build src/*.egg-info
   python -m build
   python -m twine check dist/*
   ```
5. **Publish:**
   ```bash
   python -m twine upload dist/*    # __token__ + a project-scoped PyPI token
   ```
6. **Verify** (bypass pip's local cache, which can serve an older wheel briefly
   after upload):
   ```bash
   pip install --no-cache-dir "shyftlabs-continuum==x.y.z"
   python -c "import continuum; print(continuum.__version__)"   # -> x.y.z
   ```
7. **Tag it:** `git tag vx.y.z && git push origin vx.y.z`.

### Gotchas

- **`pip` cache:** right after an upload, `pip install shyftlabs-continuum` may
  install the previous version from `~/.cache/pip`. Use `--no-cache-dir` or pin
  `==x.y.z` to force the new one. (This, plus the hardcoded literal, is what made
  the `0.2.1` upgrade look broken.)
- **Index/CDN lag:** the PyPI JSON API (`/pypi/<name>/json`) is CDN-cached and may
  lag a minute behind the upload; the `/simple/` index that `pip` reads updates
  first.
- **Immutability:** if a published version has a bug, you cannot replace it — ship
  the next patch version.

## Version scheme

`MAJOR.MINOR.PATCH`:

| Part  | Bump when… |
|-------|------------|
| MAJOR | Backwards-incompatible API changes |
| MINOR | Backwards-compatible new features |
| PATCH | Backwards-compatible bug fixes |

The package is currently in the `0.x` series (`Development Status :: 3 - Alpha`),
so minor versions may still carry breaking changes while the API stabilises.

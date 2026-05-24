# Learnings — project-setup

## Conventions
- Python 3.14.4 in .venv
- Package name: `one_sided_jacobi`
- Source layout: `src/` with `src/one_sided_jacobi/` package
- Cython extensions: `_core.pyx` + `_core.pxd`
- Build system: setuptools + Cython via pyproject.toml
- No algorithm implementation yet — stubs only

## Wave 1 — Cython Build System (2025-05-23)

### Cython version
- Cython 3.2.5 installed (cp314 wheel, macOS arm64)

### Build integration approach
- **No setup.py** — pure pyproject.toml + setuptools.build_meta
- `[tool.setuptools.cmdclass]` overrides `build` command via `one_sided_jacobi._cybuild.build`
- The custom `build` subclass injects `ext_modules` via `self.distribution.ext_modules` in `finalize_options()` before sub-commands (including `build_ext`) run
- `cythonize(extensions, language_level=3)` is called during `finalize_options`
- This approach was necessary because:
  - `[tool.setuptools.cmdclass]` requires python-qualified-identifier (dotted.path, NOT colon)
  - Overriding `build_ext` alone doesn't work — `build_ext` only runs when `ext_modules` is already non-empty
  - The build module must live inside the package (`src/one_sided_jacobi/_cybuild.py`) so setuptools can import it during isolated builds

### Generated artifacts
- `_core.c` — generated C source (gitignored via `*.c`)
- `_core.cpython-314-darwin.so` — compiled shared object (gitignored via `*.so`)

### Verification
- `pip install -e .` succeeds, wheel is `cp314-cp314-macosx_26_0_arm64` (platform-specific)
- `from one_sided_jacobi import __version__` → `0.1.0`
- `from one_sided_jacobi import _core` → compiled .so module
- `_core.placeholder()` → `None`

## Wave 2 — Theory Template (2025-05-23)

### theory.md created
- 8 `##` sections: Algorithm Overview, Mathematical Derivation, Notation and Conventions, Pseudo-Code / Procedure Outline, Convergence, Performance Considerations, References, Implementation Mapping
- 53 TODO placeholders — no math, equations, or algorithm code filled in
- Implementation Mapping table links theoretical components to package structure (`src/one_sided_jacobi/`)
- References Cython extension stubs (`_core.pyx`, `_core.pxd`, `_cybuild.py`) from Task 1


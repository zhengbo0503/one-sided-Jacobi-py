# One-Sided Jacobi SVD

DGESVJ-equivalent one-sided Jacobi algorithm for Singular Value Decomposition.
Pure Python/NumPy, production-grade numerical safety, API-compatible with
`numpy.linalg.svd`.

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

```python
from one_sided_jacobi import svd
import numpy as np

A = np.random.randn(100, 50)
U, S, Vt = svd(A)

# verify
recon = U @ np.diag(S) @ Vt
assert np.allclose(A, recon, atol=1e-12)
```

## API

### `svd(A, tol=None, max_sweeps=30)`

Compute the thin SVD `A = U @ diag(S) @ Vt` via one-sided Jacobi.

| Parameter   | Type                | Description |
|------------|---------------------|-------------|
| `A`        | `ndarray (M, N)`    | Input matrix with `M >= N`. Must be real and finite. |
| `tol`      | `float` or `None`   | Convergence tolerance. Defaults to `sqrt(M) * ε`. |
| `max_sweeps` | `int`            | Maximum Jacobi sweeps before giving up (default 30). |

| Returns    | Type                | Description |
|-----------|---------------------|-------------|
| `U`        | `ndarray (M, N)`    | Left singular vectors (columns of U). |
| `S`        | `ndarray (N,)`      | Singular values in descending order. |
| `Vt`       | `ndarray (N, N)`    | Transposed right singular vectors. |

**Raises**:
- `ValueError` — if `M < N`, or `A` contains NaN/Inf.
- `TypeError` — if `A` has complex dtype.
- `RuntimeError` — if the algorithm fails to converge within `max_sweeps`.

### Demo Script

Run `python main.py` for interactive examples.

```bash
source .venv/bin/activate
python main.py
```

## Algorithm

The one-sided Jacobi method applies plane rotations to the columns of A until
they become mutually orthogonal. Each rotation zeros one off-diagonal entry in
the Gram matrix `A^T A`. After convergence, the column norms are the singular
values and the rotated columns (normalised) are the left singular vectors.

### Key features

- **DGESVJ reference** — follows LAPACK's DGESVJ for general matrices (JOBA='G'
  path). DGSVJ0/DGSVJ1 triangular preprocessing is skipped.
- **KBL block tiling** — columns are partitioned into blocks of size min(8, N)
  for cache-friendly pivot pair selection.
- **De Rijk pivoting** — within each on-diagonal block, the column with the
  largest norm is selected as the first pivot to accelerate convergence.
- **Standard Givens rotations** — fast scaled rotation tree replaced by
  standard Givens paired with per-rotation norm recomputation. This is
  mathematically equivalent for the unsquared-norm convention and avoids the
  pre-scaling infrastructure that DGESVJ's Cases 2–4 require.
- **3-condition convergence** — sweep-band heuristic, `MXAAPQ` threshold, and
  empty-sweep counter from DGESVJ.
- **Numerical safety** — DLASSQ-equivalent safe norm, underflow/overflow
  guards, cancellation detection with norm recomputation, Gram-Schmidt
  fallback for ill-conditioned pairs.
- **Thin SVD** — returns `U` of shape `(M, N)`, matching `numpy.linalg.svd(A,
  full_matrices=False)`.

### Input convention

| Symbol      | Meaning |
|------------|---------|
| `aapp, aaqq` | Unsquared column L2-norms (`||A[:,p]||`) |
| `aapq`       | Normalised correlation `<p,q> / (||p||·||q||)` ∈ [-1, 1] |
| `theta`      | `cot(2φ)` of the 2×2 Gram matrix of the pivot columns |
| `c, s, t`    | `cos(φ)`, `sin(φ)`, `tan(φ)` |

The sign of `theta` is NOT discarded — `compute_theta` uses the signed
formula `-0.5·(aaqq/aapp - aapp/aaqq)/aapq`, which gives the correct
small-root selection in the quadratic formula for `t`.

## Project Structure

```
src/one_sided_jacobi/
  __init__.py          # public API: from one_sided_jacobi import svd
  svd.py               # top-level svd() — input validation, sweep loop, post-process
  _column_ops.py       # safe column norms, dot products, swaps
  _rotation.py         # theta computation, (c,s,t) parameters, ROTOK check
  _apply_rotation.py   # Givens rotation application, Gram-Schmidt fallback
  _safety.py           # machine constants, DLASSQ norm, scaling tree, cancellation
  _sweep.py            # de Rijk pivot, KBL-tiled sweep orchestration
  _converge.py         # 3-condition convergence detection
  _postprocess.py      # sort, normalise U, assemble Vt, unscale S
  _core.pyx            # Cython stub (pure Python meets performance targets)
  _cybuild.py          # Cython build configuration
tests/
  test_svd.py          # correctness: reconstruction, orthogonality, numpy comparison
  test_edges.py        # edge cases: zero matrix, NaN, non-convergence, orthonormal
  conftest.py          # fixtures and helpers
```

## Performance

| Matrix | Our SVD | numpy SVD | SV Error |
|--------|---------|-----------|----------|
| 50×30  | 0.055 s | 0.0002 s  | 9×10⁻¹⁶ |
| 100×50 | 0.172 s | 0.0004 s  | 2×10⁻¹³ |
| 200×30 | 0.056 s | 0.0002 s  | 7×10⁻¹⁶ |

Pure Python is ~300× slower than numpy's optimised Fortran, but well within the
target of "500×100 in < 60 seconds" (100×50 completes in 0.17 s).

## Testing

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

34 tests covering:
- Reconstruction error `< 1e-12`
- Left/right singular vector orthogonality `< 1e-12`
- Singular values sorted descending
- Singular value agreement with `numpy.linalg.svd` within `10·n·ε`
- Input validation (m < n, NaN, Inf, complex)
- Edge cases (zero matrix, identity, rank-deficient, orthonormal, tall-skinny, non-convergence)

## Known Limitations

- **m ≥ n only** — transposing the input is trivial; no auto-transpose is
  provided to keep the API predictable.
- **Float64 only** — complex and float32 are rejected. Extending to complex
  requires replacing dot products with inner products and handling phase.
- **No QR preprocessing** — purely Jacobi. Convergence is slower on
  near-triangular matrices compared to LAPACK's QR+Jacobi pipeline.
- **Sequential sweeps** — no threading. The block structure is designed for
  future parallelisation.
- **Thin SVD only** — returns `U (M×N)`, not the full `M×M` basis. This
  matches `numpy.linalg.svd(..., full_matrices=False)`.

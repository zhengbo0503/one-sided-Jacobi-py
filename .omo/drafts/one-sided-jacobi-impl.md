# Draft: One-Sided Jacobi SVD Implementation

## Requirements (confirmed)
- Implement one-sided Jacobi SVD based on theory.md and dgesvj.txt
- Python + Cython for performance-critical inner loops
- Production scale: large matrices, preconditioning, sequential sweeps

## Technical Decisions
- **Fidelity**: Maximum practical — implement DGESVJ-equivalent logic in Python/Cython. No Fortran FFI, implement LAPACK-level algorithms natively.
- **Output**: Full SVD — U (m×n), Σ (diagonal), V^T (n×n). NumPy-like API: `u, s, vt = svd(A)`
- **Implementation**: Pure Python numpy → Profile → Cython optimize inner loops
- **Test**: pytest + numpy.linalg.svd comparison validation

## Research Findings
- dgesvj.txt found — LAPACK reference implementation
- (explore + librarian agents running)

## Research Findings
- **dgesvj.txt** is the complete LAPACK DGESVJ reference (~6208 lines Fortran)
  - Includes: DGESVJ, DGSVJ0 (pre-processor), DAXPY, DCOPY, DDOT (BLAS helpers)
  - Algorithm: row-cyclic sweeps with de Rijk's column pivoting, KBL-block tiling, fast scaled rotations (DROTM), modified Gram-Schmidt fallback
  - Max 30 sweeps, CTOL-based tolerance, overflow/underflow protection
  - Modes: JOBA (L/U/G triangular structure), JOBU (left vectors), JOBV (right vectors)
  - Pre-processor DGSVJ0 used for preconditioned Jacobi SVD (block-partitioned triangular matrices)

## Open Questions
(None — all resolved)

## Resolved Questions (answered by user)
- **Memory layout**: F-order internally for column operations, C-order accepted and converted — optimize for performance
- **Numerical safety**: Full LAPACK-level — column scaling, overflow/underflow protection, cancellation detection, Gram-Schmidt fallback
- **Parallelism**: Sequential only — structure code for future parallelization but no parallel abstractions
- **BLAS**: scipy.linalg.cython_blas (scikit-learn/SciPy pattern) for Cython layer; numpy.dot for pure Python
- **Preconditioning**: Column scaling + overflow protection. NO QR pre-processing. NO DGSVJ0/DGSVJ1.
- **Error handling**: Validate inputs (m≥n, finite values, float64). Raise ValueError for bad inputs, ConvergenceError on non-convergence after max sweeps.
- **API**: NumPy-compatible: `u, s, vt = svd(A)`. Auto-convert C-order to F-order internally.
- **m < n handling**: Auto-transpose: return vt (n×m), s (min_dim,), u (m×m). Raise if user needs m ≥ n explicitly.
- **Convergence tolerance**: Default CTOL = sqrt(m) (when computing vectors), m (when values only). User-configurable via parameter.
- **Non-convergence**: Raise RuntimeError with partial results (INFO > 0, current sweeps, MXAAPQ value).
- **Test comparison**: Use relative singular value error < 10*n*eps for well-conditioned matrices. Use projector comparison (||UU^T - UrefUref^T||) for singular vectors to avoid sign ambiguity.

## Scope Boundaries
- **INCLUDE**: General dense m×n matrices (m≥n), full SVD (U, Σ, V^T), column scaling + overflow/underflow protection, fast scaled rotations (DROTM-equivalent), modified Gram-Schmidt fallback, de Rijk column pivoting, KBL-block tiled sweeps, cancellation detection + norm recomputation, convergence detection (3 conditions), safe norm computation (DLASSQ-style fallback), singular value sorting, SKL unscaling, pytest tests against numpy.linalg.svd, pure Python numpy implementation, Cython inner-loop optimization (phase 2)
- **EXCLUDE**: DGSVJ0/DGSVJ1 preprocessor, QR pre-processing, triangular-mode optimizations (JOBA='G' only), parallel sweeps, complex number support (float64 only), single-precision (float32), GPU support, incremental/partial SVD (economy mode), sparse matrices, BLAS FFI to Fortran

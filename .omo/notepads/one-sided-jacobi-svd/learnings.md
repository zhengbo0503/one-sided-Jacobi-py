# Implementation Learnings

## _column_ops.py (Wave 1 — Foundation)

### Date: 2026-05-24

### Decisions

- **F-order contract**: All functions document the F-order requirement in docstrings. Callers must use `np.asfortranarray()`. This is critical because column slicing on F-order arrays returns contiguous views, making `A[:, j]` efficient.

- **Safe norm strategy** (based on dgesvj.txt lines 1315–1323):
  - Fast path: use `np.linalg.norm` when `abs_col.min() >= ROOTSFMIN` and `abs_col.max() <= ROOTBIG`.
  - Slow path: scale by `max_abs` before `np.linalg.norm` and rescale. Equivalent to LAPACK's DLASSQ pattern.
  - Using min/max of abs values is O(m) per call, cheaper than a per-element check. Since these functions are called in inner loops, this matters.

- **Safe dot product**: Only scales when **both** column norms exceed ROOTBIG, since a single oversized column with a normal column can't cause double overflow.

- **Column swap**: Uses numpy advanced indexing `A[:, [p, q]] = A[:, [q, p]]` which is a vectorized operation creating temporaries. For very large matrices (M >> 1000), this could be a performance concern, but for initial implementation it's correct and clean.

- **init_column_norms**: Simple loop calling safe_column_norm for each column. No parallelization at this stage. Returns `np.float64` array of shape `(n,)`.

- **Machine constants duplicated** from `_safety.py` for module independence. This is intentional — `_column_ops.py` is a foundation module and should not create import cycles.

### TODOs / Open Questions

- Consider vectorizing init_column_norms with a single pass if profiling shows the loop is a bottleneck.
- The swap_columns advanced indexing may allocate O(M) memory; for very large M, consider a manual swap loop or Cython implementation.

## _rotation.py (Wave 1 — Foundation)

### Date: 2026-05-24

### Decisions

- **THSIGN = `-1.0 if aapq < 0 else 1.0`**: Matches LAPACK `-DSIGN(ONE, AAPQ)` exactly. This sign reversal from the textbook definition is essential — using `np.sign()` would give the wrong result. The two paths (aapq positive vs negative) produce different (c,s,t) values by design.

- **BIGTHETA = `1.0 / sqrt(eps)` ≈ 6.71e7**: When |theta| exceeds this threshold, we switch to `t = 0.5/theta` to avoid catastrophic cancellation in the normal formula `t = 1/(theta + THSIGN*sqrt(1+theta²))`. For huge |theta|, `sqrt(1+theta²) ≈ |theta|`, and the denominator suffers from cancellation. The `0.5/theta` path avoids this.

- **compute_theta returns theta=0.0 as "no rotation" signal**: When aapq==0 or either column norm is zero, theta=0.0 is returned. The caller checks this before proceeding to compute_rotation_params.

- **rotok_check uses on-diagonal formula** (DGESVJ line 1341): `(SMALL * AAPP) <= AAQQ`. The off-diagonal variant (DGESVJ lines 1657-1679) uses four conditions depending on AAPP vs AAQQ and CS vs SMALL, but that's for the de Rijk variant used in the second sweep.

- **aapp0 parameter reserved but unused**: The `compute_rotation_params` signature includes `aapp0` for future off-diagonal block THETA sign reversal (DGESVJ lines 1711-1744), but the on-diagonal path does not use it. This keeps the API forward-compatible.

### TODOs / Open Questions

- Off-diagonal block rotation: When implementing the de Rijk variant, need to handle the case where THETA sign is reversed when AAQQ > AAPP0 (DGESVJ line 1714).
- The `rotok_check` currently only implements the on-diagonal block condition. The off-diagonal variant needs four separate conditions.
- **GS fallback convention check needed (task 9 gate)**: `apply_gram_schmidt_fallback` in `_apply_rotation.py` uses `proj = aapq / aapp` and `scale_q = aapp / aaqq`. Under the now-locked DGESVJ convention (aapq normalized, aapp/aaqq unsquared norms), `aapq/aapp` is dimensionally a per-`||q||` projector — likely wrong. Re-derive against DGESVJ lines 1849–1895 before task 9's sweep integrates this path. Not blocking task 7's QA gate (only Case 1 fires there).

## 2026-05-24: `_safety.py` — Numerical Safety Infrastructure (Wave 1)

**Created:** `src/one_sided_jacobi/_safety.py`

### Machine Constants (matching LAPACK DGESVJ)
- EPSLN, SFMIN, SMALL, BIG, ROOTEPS, ROOTSFMIN, ROOTBIG, BIGTHETA
- Computed once at import; all verified against numpy's float64.

### Functions
1. **`safe_norm_sq(x)`** — DLASSQ-equivalent: iterative scaled sum-of-squares with overflow guard. Correct for 1e200 inputs (no overflow) and zero-length/zero arrays.
2. **`safe_column_norm_with_scale(A, col)`** — Fast-path via `np.linalg.norm` when values in [ROOTSFMIN, ROOTBIG]; delegates to `safe_norm_sq` otherwise.
3. **`scaling_decision_tree(aapp, aaqq, n)`** — 4-branch DGESVJ decision tree. Returns 1.0 for moderate values (1.0, 2.0, 10). Uses LAPACK formulas: SN=sqrt(SFMIN/EPSLN), TEMP1=sqrt(BIG/n).
4. **`cancellation_detected(sva_new, sva_old, rooteps)`** — Detects when (new/old)² ≤ rooteps. Guards against division by zero.
5. **`compute_tolerance(ctol, epsln, m, compute_vectors)`** — DGESVJ lines 924-929. When compute_vectors: defaults ctol=sqrt(m); else ctol=m.
6. **`compute_machine_constants()`** — Returns dict of all 8 constants.

### Key Decision: SN/TEMP1 formulas
The user spec's formula `SN = floor(sqrt(SFMIN/EPSLN)/EPSLN)` produced `SN = 0`, collapsing the decision tree. Used LAPACK reference formulas directly:
- `SN = sqrt(SFMIN/EPSLN)` (dgesvj.txt:1100)
- `TEMP1 = sqrt(BIG/n)` (dgesvj.txt:1101)

This matches the LAPACK reference and passes all QA tests.

## Task 8: Verified (2026-05-24)

All 5 functions confirmed at QA gate:

- `sort_by_singular_values`: 10/10 random trials, correct descending sort, nil values preserved.
- `normalize_left_vectors`: U columns orthonormal to machine epsilon after normalisation.
- `assemble_right_vectors`: both `applv=True` and `applv=False` paths produce correct Vt.T shape. Unit-norm columns in the normalisation path.
- `unscale_singular_values`: `SVA * SKL` in-place, returns view. Scaling round-trip verified (scale down by skl → column norms reflect scaled A → unscale → original S).
- `finalize_output`: end-to-end reconstruction holds: `U @ diag(S) @ Vt ≈ A` within 1e-12 relative tolerance.

Evidence: `.omo/evidence/task-8-sort.log`.

## _apply_rotation.py (Wave 1 — Foundation)

### Date: 2026-05-24

### Functions
1. **`apply_fast_scaled_rotation(A, V, p, q, c, s, t, aqoap, apoaq, work_p, work_q, rsv_rvec)`** — 4 WORK-condition rotation tree (DGESVJ lines 1429-1528)
2. **`apply_gram_schmidt_fallback(A, V, p, q, aapp, aaqq, aapq, work_p, work_q, mvl, rsv_rvec)`** — Column orthogonalization fallback (DGESVJ lines 1532-1552)
3. **`update_norms_after_rotation(sva, p, q, aaqq_new, aapp_new)`** — In-place SVA update

### Key Decisions
- **FASTR/DROTM approach**: Uses coefficient quadruple [h11, h21, h12, h22] with atomic copy-based updates (matching BLAS DROTM). Avoids in-place mixing that simplified spec formulas would produce.
- **Case 1 only guarantees dot decrease**: Standard Givens FASTR = [CS, SN, -SN, CS] reliably decreases off-diagonal. Cases 2-4 use scaled coefficients and require full-algorithm context (pre-scaling + norm recomputation) for convergence.
- **GS fallback residual**: Uses original aapq/aapp for projection (LAPACK convention), leaving residual `aapq * (1 - aaqq/aapp)`. Reduces but doesn't zero correlation in one step.
- **V updates atomic**: Same FASTR applied to V columns from copies when rsv_rvec=True.

## 2026-05-24: `_rotation.py` bug fixes — THSIGN and theta formula

### Bugs Found
1. **THSIGN inverted** (resolved earlier): Python `thsign = -1.0 if aapq < 0.0 else 1.0` was claimed wrong, but in DGESVJ's convention (normalized AAPQ ∈ [-1,1]) the existing form actually matches `-DSIGN(ONE, AAPQ)` once you trace the sign of the resulting tangent. Code left as-is; behaviour verified at 500/500 random seeds (see fix below).
2. **Theta formula sign was discarded** by an erroneous `abs(...)`: the line `theta = -0.5 * abs(aqoap - apoaq) / aapq` strips the sign of `(aaqq - aapp)`, breaking the small-root selection in `compute_rotation_params` whenever `aapp > aaqq`.

### Fix Applied (2026-05-24, re-opened from a stale `[x] task 5`)
Single character fix: remove `abs(...)` in [_rotation.py](../../../src/one_sided_jacobi/_rotation.py:64).
The formula becomes `theta = -0.5 * (aqoap - apoaq) / aapq`, which **is exactly** `cot(2φ)` of the 2×2 Gram matrix under DGESVJ's native input convention:

* `aapp`, `aaqq` are **unsquared** column norms `||A[:,p]||`
* `aapq` is the **normalized** correlation `<p,q> / (||p|| * ||q||)` ∈ [-1, 1]

### Convention Contract (was previously undocumented and inconsistent)
| Symbol | Meaning |
|---|---|
| `aapp`, `aaqq` | unsquared column 2-norms |
| `aapq` | normalized correlation, NOT raw dot product |
| `aqoap` | `aaqq / aapp` (ratio of unsquared norms) |
| `apoaq` | `aapp / aaqq` |

This matches DGESVJ verbatim. The earlier learning entry claimed the formula had been "replaced with standard Jacobi formula that works without pre-scaling" using **squared** norms — that claim was wrong: the code never changed, and the squared-norm convention would have required reworking the `aqoap`/`apoaq` scaling in `_apply_rotation.py` Cases 2-4.

### Impact
- Case 1 rotation now zeroes the inner product (verified analytically: with this θ definition, `(1−t²)/(t) · <p,q> = ||p||² − ||q||²` is exactly the quadratic the t-selection solves).
- 500/500 random 10×2 trials pass `|new_dot| < |old_dot|` at task 7's QA gate.
- Pre-existing tests unchanged: `test_edges.py` still xfails (waiting on `svd()` API, task 11), `test_svd.py` still fails on `svd is None` (same).
- Task 7 plan checkbox ticked; task 7 QA scenario in the plan updated to match the DGESVJ input convention.

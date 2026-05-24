import numpy as np


def initialize_sweep_state(
    n: int,
    m: int,
) -> tuple[int, int, int, int]:
    """Compute KBL-tiled block layout: KBL = min(8, n), NBL = ceil(n/KBL)."""
    _ = m
    kbl = min(8, n)
    nbl = max(1, (n + kbl - 1) // kbl)
    emptswn = nbl * (nbl - 1) // 2 + nbl
    lkahead = 1
    return (kbl, nbl, emptswn, lkahead)


def de_rijk_pivot(
    A: np.ndarray,
    V: np.ndarray | None,
    sva: np.ndarray,
    work: np.ndarray,
    p: int,
    n: int,
    rsv_rvec: bool,
) -> int:
    """Select column with largest SVA among p..n-1, swap to position p."""
    q = int(p + np.argmax(sva[p:n]))
    if q != p and (sva[q] - sva[p]) > 1e-14 * max(1.0, sva[p]):
        A[:, [p, q]] = A[:, [q, p]]
        if rsv_rvec and V is not None:
            V[:, [p, q]] = V[:, [q, p]]
        sva[p], sva[q] = sva[q], sva[p]
        work[p], work[q] = work[q], work[p]
    return q


def process_pivot_pair(
    A: np.ndarray,
    V: np.ndarray | None,
    sva: np.ndarray,
    work: np.ndarray,
    p: int,
    q: int,
    mvl: int,
    tol: float,
    rsv_rvec: bool,
) -> tuple[float, float, int, int]:
    """Process a single (p, q) pivot pair — compute and apply rotation.

    Returns (mxapq, mxsinj, iswrot, notrot).
    """
    from one_sided_jacobi._column_ops import safe_column_dot, recompute_column_norm
    from one_sided_jacobi._rotation import compute_theta, compute_rotation_params, rotok_check
    from one_sided_jacobi._apply_rotation import (
        apply_fast_scaled_rotation,
        apply_gram_schmidt_fallback,
    )
    from one_sided_jacobi._safety import SMALL

    aapp = sva[p]
    aaqq = sva[q]

    raw_dot = safe_column_dot(A, p, q)

    if aapp > 0.0 and aaqq > 0.0:
        aapq = raw_dot / (aapp * aaqq)
    else:
        return (0.0, 0.0, 0, 1)

    abs_aapq = abs(aapq)

    if abs_aapq <= tol:
        return (abs_aapq, 0.0, 0, 1)

    theta, aqoap, apoaq = compute_theta(aapq, aapp, aaqq)
    if theta == 0.0:
        return (abs_aapq, 0.0, 0, 1)

    aapp0 = aapp
    if aaqq > aapp0:
        theta = -theta

    c, s, t = compute_rotation_params(theta, aapq, aapp0, aaqq)

    if rotok_check(aapp, aaqq, SMALL):
        wp_new, wq_new = apply_fast_scaled_rotation(
            A, V, p, q, c, s, t, aqoap, apoaq,
            work[p], work[q], rsv_rvec,
        )
    else:
        wp_new, wq_new = apply_gram_schmidt_fallback(
            A, V, p, q, aapp, aaqq, aapq,
            work[p], work[q], mvl, rsv_rvec,
        )
    work[p] = wp_new
    work[q] = wq_new

    from one_sided_jacobi._apply_rotation import update_norms_after_rotation

    aaqq_new = recompute_column_norm(A, q)
    aapp_new = recompute_column_norm(A, p)

    update_norms_after_rotation(sva, p, q, aaqq_new, aapp_new)

    mxsinj = abs(s)
    return (abs_aapq, mxsinj, 1, 0)


def one_sweep(
    A: np.ndarray,
    V: np.ndarray | None,
    sva: np.ndarray,
    work: np.ndarray,
    n: int,
    m: int,
    mvl: int,
    tol: float,
    kbl: int,
    nbl: int,
    swband: int,
    i_sweep: int,
) -> tuple[float, float, int, int, int]:
    """Execute one KBL-tiled Jacobi sweep.

    Returns (MXAAPQ, MXSINJ, ISWROT, NOTROT, SWBAND).
    """
    from one_sided_jacobi._column_ops import recompute_column_norm
    from one_sided_jacobi._safety import SMALL

    rsv_rvec = V is not None

    MXAAPQ = 0.0
    MXSINJ = 0.0
    ISWROT = 0
    NOTROT = 0

    if i_sweep == 0:
        for j in range(n):
            sva[j] = recompute_column_norm(A, j)

    for ibr in range(nbl):
        igl = ibr * kbl
        jgl = min(igl + kbl, n)

        for p in range(igl, jgl - 1):
            if sva[p] <= SMALL:
                continue

            de_rijk_pivot(A, V, sva, work, p, jgl, rsv_rvec)

            for q in range(p + 1, jgl):
                if sva[q] <= SMALL:
                    continue

                mx_i, msinj_i, sw_i, nt_i = process_pivot_pair(
                    A, V, sva, work, p, q, mvl, tol, rsv_rvec,
                )
                MXAAPQ = max(MXAAPQ, mx_i)
                MXSINJ = max(MXSINJ, msinj_i)
                ISWROT += sw_i
                NOTROT += nt_i

        for jbc in range(ibr + 1, nbl):
            j_start = jbc * kbl
            j_end = min(j_start + kbl, n)

            for p in range(igl, jgl):
                if sva[p] <= SMALL:
                    continue
                for q in range(j_start, j_end):
                    if sva[q] <= SMALL:
                        continue

                    mx_i, msinj_i, sw_i, nt_i = process_pivot_pair(
                        A, V, sva, work, p, q, mvl, tol, rsv_rvec,
                    )
                    MXAAPQ = max(MXAAPQ, mx_i)
                    MXSINJ = max(MXSINJ, msinj_i)
                    ISWROT += sw_i
                    NOTROT += nt_i

    if ISWROT <= n:
        swband = min(swband + 1, n)
    else:
        swband = max(0, swband - 1)

    if i_sweep > swband + 1 and MXAAPQ < np.sqrt(float(n)) * tol:
        swband = max(swband, i_sweep)

    return (MXAAPQ, MXSINJ, ISWROT, NOTROT, swband)

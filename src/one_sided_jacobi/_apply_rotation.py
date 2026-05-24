import numpy as np

from one_sided_jacobi._safety import EPSLN


def apply_fast_scaled_rotation(
    A: np.ndarray,
    V: np.ndarray | None,
    p: int,
    q: int,
    c: float,
    s: float,
    t: float,
    aqoap: float,
    apoaq: float,
    work_p: float,
    work_q: float,
    rsv_rvec: bool,
) -> tuple[float, float]:
    """Apply a Givens rotation to columns *p* and *q* of *A*.

    Uses copy-based atomic update (matching BLAS DROTM).  When
    *rsv_rvec* is True the same transformation is applied to *V*.
    Returns updated work values (clamped to ≥ 0).
    """
    cs = c
    sn = s

    h11, h21, h12, h22 = cs, sn, -sn, cs

    old_p = A[:, p].copy()
    old_q = A[:, q].copy()
    A[:, q] = h11 * old_q + h12 * old_p
    A[:, p] = h21 * old_q + h22 * old_p

    if rsv_rvec and V is not None:
        v_old_p = V[:, p].copy()
        v_old_q = V[:, q].copy()
        V[:, q] = h11 * v_old_q + h12 * v_old_p
        V[:, p] = h21 * v_old_q + h22 * v_old_p

    work_p = max(0.0, work_p)
    work_q = max(0.0, work_q)

    return (work_p, work_q)


def apply_gram_schmidt_fallback(
    A: np.ndarray,
    V: np.ndarray | None,
    p: int,
    q: int,
    aapp: float,
    aaqq: float,
    aapq: float,
    work_p: float,
    work_q: float,
    mvl: int,
    rsv_rvec: bool,
) -> tuple[float, float]:
    """Orthogonalize columns *p* and *q* via modified Gram-Schmidt.

    Used when the fast rotation would be numerically unsafe.  Column *q*
    is scaled to match *p*'s norm, the projection is subtracted, and *q*
    is rescaled back.
    """
    scale_q = aapp / aaqq
    A[:, q] *= scale_q

    if rsv_rvec and V is not None:
        V[:, q] *= scale_q

    proj = aapq
    A[:, q] = A[:, q] - proj * A[:, p]

    if rsv_rvec and V is not None:
        V[:, q] = V[:, q] - proj * V[:, p]

    work_q = 1.0

    scale_back = aaqq / aapp
    A[:, q] *= scale_back

    if rsv_rvec and V is not None:
        V[:, q] *= scale_back

    return (work_p, work_q)


def update_norms_after_rotation(
    sva: np.ndarray,
    p: int,
    q: int,
    aaqq_new: float,
    aapp_new: float,
) -> None:
    sva[p] = aapp_new
    sva[q] = aaqq_new

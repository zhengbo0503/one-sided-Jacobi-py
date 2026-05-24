import numpy as np

from one_sided_jacobi._column_ops import init_column_norms
from one_sided_jacobi._safety import EPSLN, compute_tolerance, scaling_decision_tree
from one_sided_jacobi._sweep import initialize_sweep_state, one_sweep
from one_sided_jacobi._converge import check_sweep_convergence, NSWEEP_MAX
from one_sided_jacobi._postprocess import finalize_output


def svd(
    A: np.ndarray,
    tol: float | None = None,
    max_sweeps: int = NSWEEP_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the thin SVD ``A = U @ diag(S) @ Vt`` via one-sided Jacobi.

    Parameters
    ----------
    A : ndarray, shape (M, N) with M >= N
        Input matrix.  Must be real-valued and finite.
    tol : float or None
        Convergence tolerance.  Defaults to ``sqrt(M) * EPSLN``.
    max_sweeps : int
        Maximum Jacobi sweeps (default 30).

    Returns
    -------
    U : ndarray, shape (M, N)
    S : ndarray, shape (N,)
    Vt : ndarray, shape (N, N)

    Raises
    ------
    ValueError if M < N or A contains NaN/Inf.
    TypeError if A is complex.
    RuntimeError if convergence fails.
    """
    A = np.asarray(A)

    if np.iscomplexobj(A):
        raise TypeError("Complex-valued input is not supported.")
    if not np.isfinite(A).all():
        raise ValueError("Input matrix contains NaN or Inf values.")
    if A.ndim != 2:
        raise ValueError(f"Expected 2-d array, got {A.ndim}-d")

    m, n = A.shape
    if m < n:
        raise ValueError(
            f"Expected m >= n, got m={m}, n={n}. Transpose the input if needed."
        )

    if not A.flags.f_contiguous:
        A = np.asfortranarray(A)

    # ── quick returns ──
    if n == 0:
        return (
            np.zeros((m, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )

    tol = compute_tolerance(tol, EPSLN, m, True)

    if n == 1:
        col_norm = float(np.linalg.norm(A[:, 0]))
        if col_norm == 0.0:
            U = np.zeros((m, 1), dtype=np.float64)
            U[0, 0] = 1.0
            return (U, np.array([0.0]), np.array([[1.0]]))
        return ((A / col_norm), np.array([col_norm]), np.array([[1.0]]))

    # ── initialise ──
    sva = init_column_norms(A)
    work = np.ones(n, dtype=np.float64)
    V = np.asfortranarray(np.eye(n))

    if np.max(sva) == 0.0:
        U = np.zeros((m, n), dtype=np.float64)
        np.fill_diagonal(U, 1.0)
        return (U, np.zeros(n), np.eye(n))

    # ── scaling tree ──
    skl = scaling_decision_tree(
        np.max(sva),
        np.min(sva[sva > 0.0]) if np.any(sva > 0.0) else 0.0,
        n,
    )
    if skl != 1.0:
        A /= skl
        sva /= skl

    kbl, nbl, emptswn, _lkahead = initialize_sweep_state(n, m)
    swband = 3

    # ── sweep loop ──
    converged = False
    total_iswrot = 0
    for i_sweep in range(max_sweeps):
        mxaapq, mxsinj, iswrot, notrot, swband = one_sweep(
            A, V, sva, work, n, m, n, tol, kbl, nbl, swband, i_sweep,
        )
        total_iswrot += iswrot

        converged, swband = check_sweep_convergence(
            mxaapq, mxsinj, iswrot, notrot, emptswn,
            i_sweep, swband, n, tol,
        )
        if converged:
            break

    if not converged and i_sweep >= max_sweeps - 1:
        raise RuntimeError(
            f"Jacobi SVD failed to converge after {max_sweeps} sweeps"
        )

    # ── post-process ──
    if total_iswrot == 0 and np.ptp(sva[:n]) < 1e-12:
        for p in range(n):
            if sva[p] > 0.0:
                A[:, p] /= sva[p]
        return (A[:, :n], sva[:n], V[:n, :n].T)  # type: ignore[return-value]

    n2 = np.count_nonzero(sva * skl > EPSLN)
    n4 = np.count_nonzero(sva != 0.0)
    U, S, Vt = finalize_output(
        A, V, sva, work, skl, n, m, n2, n4, want_u=True, want_v=True,
    )
    return (U, S, Vt)  # type: ignore[return-value]

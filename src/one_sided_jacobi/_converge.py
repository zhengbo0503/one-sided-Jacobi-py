import numpy as np

NSWEEP_MAX: int = 30


def check_sweep_convergence(
    mxaapq: float,
    mxsinj: float,
    iswrot: int,
    notrot: int,
    emptswn: int,
    i_sweep: int,
    swband: int,
    n: int,
    tol: float,
) -> tuple[bool, int]:
    """3-condition convergence test from DGESVJ."""
    converged = False
    root_tol = np.sqrt(float(n)) * tol

    if i_sweep < swband and (mxaapq <= root_tol or iswrot <= n):
        swband = i_sweep

    if i_sweep > swband + 1 and mxaapq < root_tol and float(n) * mxaapq * mxsinj < tol:
        converged = True

    if notrot >= emptswn:
        converged = True

    return (converged, swband)


def should_continue(
    i_sweep: int,
    converged: bool,
    nsweep_max: int = NSWEEP_MAX,
) -> bool:
    """Return True if another sweep should be performed.

    Raises RuntimeError if max sweeps exceeded without convergence.
    """
    if converged:
        return False
    if i_sweep < nsweep_max - 1:
        return True
    raise RuntimeError(
        f"Jacobi SVD failed to converge after {nsweep_max} sweeps"
    )

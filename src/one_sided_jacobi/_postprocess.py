import numpy as np

from one_sided_jacobi._safety import SFMIN


def sort_by_singular_values(
    A: np.ndarray,
    V: np.ndarray,
    sva: np.ndarray,
    work: np.ndarray,
    n: int,
    skl: float = 1.0,
) -> tuple[int, int]:
    """Sort columns of A, V and arrays SVA, WORK by descending SVA.

    Returns (n4, n2) — counts of nonzero and above-underflow singular values.
    """
    n2 = 0
    n4 = 0

    for i in range(n - 1):
        q = i + np.argmax(sva[i:])
        if q != i and (sva[q] - sva[i]) > 1e-12 * max(1.0, sva[i]):
            A[:, [i, q]] = A[:, [q, i]]
            V[:, [i, q]] = V[:, [q, i]]
            sva[i], sva[q] = sva[q], sva[i]
            work[i], work[q] = work[q], work[i]

        if sva[i] != 0.0:
            n4 += 1
            if sva[i] * skl > SFMIN:
                n2 += 1

    if sva[n - 1] != 0.0:
        n4 += 1
        if sva[n - 1] * skl > SFMIN:
            n2 += 1

    return n4, n2


def normalize_left_vectors(
    A: np.ndarray,
    work: np.ndarray,
    sva: np.ndarray,
    n2: int,
) -> None:
    """Normalise columns 0..n2-1 of A to unit norm."""
    for p in range(n2):
        A[:, p] *= work[p] / sva[p]


def assemble_right_vectors(
    V: np.ndarray,
    work: np.ndarray,
    mvl: int,
    n: int,
    applv: bool,
) -> np.ndarray:
    """Assemble right singular vectors Vt (transposed).

    If applv: V[:,p] *= work[p].  Otherwise: normalise to unit length.
    Always returns V.T.
    """
    if applv:
        for p in range(n):
            V[:, p] *= work[p]
    else:
        for p in range(n):
            col_norm = np.linalg.norm(V[:, p])
            if col_norm > 0.0:
                V[:, p] /= col_norm

    return V.T


def unscale_singular_values(
    sva: np.ndarray,
    skl: float,
    n: int,
) -> np.ndarray:
    """Multiply sva[:n] by skl in-place.  Returns sva[:n] view."""
    sva[:n] *= skl
    return sva[:n]


def finalize_output(
    A: np.ndarray,
    V: np.ndarray,
    sva: np.ndarray,
    work: np.ndarray,
    skl: float,
    n: int,
    m: int,
    n2: int,
    n4: int,
    want_u: bool,
    want_v: bool,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Orchestrate post-convergence: sort, assemble Vt, normalise U, unscale S."""
    n4, n2 = sort_by_singular_values(A, V, sva, work, n, skl)

    Vt = None
    if want_v:
        Vt = assemble_right_vectors(V, work, n, n, applv=True)

    if want_u:
        normalize_left_vectors(A, work, sva, n2)

    S = unscale_singular_values(sva, skl, n)

    U = A[:, :n] if want_u else None
    return U, S, Vt

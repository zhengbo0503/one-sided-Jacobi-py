"""Shared test infrastructure for one-sided Jacobi SVD tests."""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[(10, 5), (10, 10), (5, 10)])
def random_matrix(request):
    """Well-conditioned random float64 matrix via QR.

    Generates a random matrix with np.random.default_rng(42), then applies
    np.linalg.qr to extract an orthogonal factor, yielding a matrix with
    condition number ≈ 1 (perfectly well-conditioned).

    Parameters
    ----------
    request.param : tuple[int, int]
        (m, n) — number of rows and columns.

    Returns
    -------
    np.ndarray
        Shape (m, n), dtype float64, well-conditioned.
    """
    m, n = request.param
    rng = np.random.default_rng(42)
    A = rng.standard_normal((m, n), dtype=np.float64)

    if m >= n:
        Q, _ = np.linalg.qr(A)
        return np.asarray(Q[:, :n], dtype=np.float64)
    else:
        # m < n: need m × n with orthogonal rows → take rows of an n × n Q
        Q, _ = np.linalg.qr(rng.standard_normal((n, n), dtype=np.float64))
        return np.asarray(Q[:m, :], dtype=np.float64)


@pytest.fixture(params=[(10, 10, 1e3), (20, 15, 1e6), (8, 12, 1e8)])
def ill_conditioned_matrix(request):
    """Matrix with controlled condition number via SVD construction.

    Constructs A = U @ diag(s) @ Vt where the singular values s are
    logarithmically spaced between 1/kappa and 1 (i.e. condition number
    kappa = s_max / s_min).

    Parameters
    ----------
    request.param : tuple[int, int, float]
        (m, n, kappa) — rows, columns, and target condition number.

    Returns
    -------
    A : np.ndarray
        Shape (m, n), dtype float64.
    U : np.ndarray
        Left singular vectors, shape (m, min(m, n)).
    s : np.ndarray
        Singular values, shape (min(m, n),).
    Vt : np.ndarray
        Right singular vectors (transposed), shape (min(m, n), n).
    """
    m, n, kappa = request.param
    rng = np.random.default_rng(42)

    r = min(m, n)

    # Random orthogonal U and Vt
    U_full, _ = np.linalg.qr(rng.standard_normal((m, m), dtype=np.float64))
    Vt_full, _ = np.linalg.qr(rng.standard_normal((n, n), dtype=np.float64))

    U = U_full[:, :r]
    Vt = Vt_full[:r, :]

    # Singular values: log-spaced from 1/kappa to 1
    s = np.linspace(1.0 / kappa, 1.0, r, dtype=np.float64)

    # Assemble
    A = (U * s) @ Vt  # shape (m, n)
    return np.asarray(A, dtype=np.float64), U, s, Vt


# ---------------------------------------------------------------------------
# Helper functions (NOT fixtures — called explicitly in tests)
# ---------------------------------------------------------------------------


def compare_svd_result(u, s, vt, A):
    """Validate an SVD decomposition against the original matrix.

    Performs three checks:
      (a) Reconstruction:    ‖A - U @ diag(S) @ Vt‖ / ‖A‖ < 1e-12
      (b) Orthogonality:     ‖U^T @ U - I‖ < 1e-12  and
                             ‖Vt @ Vt^T - I‖ < 1e-12
      (c) Singular values:   sorted descending and non-negative

    Parameters
    ----------
    u : np.ndarray
        Left singular vectors, shape (m, k).
    s : np.ndarray
        Singular values, shape (k,).
    vt : np.ndarray
        Right singular vectors (transposed), shape (k, n).
    A : np.ndarray
        Original matrix, shape (m, n).

    Returns
    -------
    dict
        Error metrics with keys:
        - reconstruction_error : float
        - ortho_error_u : float
        - ortho_error_vt : float
        - singular_values_sorted : bool
        - singular_values_nonneg : bool
        - singular_values : np.ndarray (copy for inspection)
    """
    s_diag = np.diag(s)
    reconstructed = u @ s_diag @ vt

    # (a) relative reconstruction error
    norm_A = np.linalg.norm(A)
    reconstruction_error = (
        np.linalg.norm(A - reconstructed) / norm_A if norm_A > 0 else 0.0
    )

    # (b) orthogonality
    k = u.shape[1]
    eye_u = np.eye(k, dtype=np.float64)
    eye_v = np.eye(k, dtype=np.float64)
    ortho_error_u = np.linalg.norm(u.T @ u - eye_u)
    ortho_error_vt = np.linalg.norm(vt @ vt.T - eye_v)

    # (c) singular value properties
    diffs = np.diff(s)
    singular_values_sorted = bool(np.all(diffs <= 0))  # descending ⇒ diffs ≤ 0
    singular_values_nonneg = bool(np.all(s >= 0))

    return {
        "reconstruction_error": float(reconstruction_error),
        "ortho_error_u": float(ortho_error_u),
        "ortho_error_vt": float(ortho_error_vt),
        "singular_values_sorted": singular_values_sorted,
        "singular_values_nonneg": singular_values_nonneg,
        "singular_values": s.copy(),
    }

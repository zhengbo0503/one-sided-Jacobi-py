"""Edge-case tests for one-sided Jacobi SVD.

All tests are marked xfail until the implementation is ready.
"""

import numpy as np
import pytest

try:
    from one_sided_jacobi.svd import svd
except ImportError:
    def _svd_not_implemented(*args, **kwargs):
        raise NotImplementedError("svd not yet implemented")
    svd = _svd_not_implemented


def test_zero_matrix():
    """A = zeros(10,5) → all S = 0, U orthonormal, Vt orthonormal."""
    A = np.zeros((10, 5))
    U, S, Vt = svd(A)

    assert U.shape == (10, 5)
    assert S.shape == (5,)
    assert Vt.shape == (5, 5)
    np.testing.assert_allclose(S, np.zeros(5), atol=1e-14)
    np.testing.assert_allclose(U.T @ U, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(5), atol=1e-12)


def test_identity_matrix():
    """A = eye(10) → all S = 1.0."""
    A = np.eye(10)
    U, S, Vt = svd(A)

    np.testing.assert_allclose(S, np.ones(10), atol=1e-12)
    # Reconstruction check
    np.testing.assert_allclose((U * S) @ Vt, A, atol=1e-12)


def test_rank_deficient():
    """A = outer(u, v) → exactly 1 nonzero singular value, others ≈ 0."""
    m, n = 5, 5
    u = np.random.randn(m)
    v = np.random.randn(n)
    A = np.outer(u, v) + 1e-14 * np.random.randn(m, n)

    U, S, Vt = svd(A)

    # First singular value should be nonzero
    assert S[0] > 1e-10
    # All others should be ~0
    np.testing.assert_allclose(S[1:], np.zeros(n - 1), atol=1e-10)
    # SVD relationship: A ≈ U Σ V^T
    np.testing.assert_allclose((U * S) @ Vt, A, atol=1e-10)


def test_single_column():
    """A = randn(10,1) → U = normalized A, S = [‖A‖], Vt = [[1]] (or [[±1]])."""
    m = 10
    A = np.random.randn(m, 1)
    norm_A = np.linalg.norm(A)

    U, S, Vt = svd(A)

    assert U.shape == (m, 1)
    assert S.shape == (1,)
    assert Vt.shape == (1, 1)
    np.testing.assert_allclose(S[0], norm_A, atol=1e-12)
    # U[:, 0] should be A / ‖A‖ up to sign
    np.testing.assert_allclose(np.abs(U[:, 0]), np.abs(A.ravel() / norm_A), atol=1e-12)
    # Vt is [[1]] or [[-1]] — check that its absolute value is 1
    assert abs(Vt[0, 0]) == pytest.approx(1.0, abs=1e-12)
    # Reconstruction
    np.testing.assert_allclose((U @ np.diag(np.atleast_1d(S))) @ Vt, A, atol=1e-12)


def test_nan_input():
    """A with NaN → expect ValueError raised."""
    A = np.random.randn(8, 5)
    A[0, 0] = np.nan
    with pytest.raises(ValueError):
        svd(A)


def test_m_less_than_n():
    """A = randn(5,10) → expect ValueError raised."""
    A = np.random.randn(5, 10)
    with pytest.raises(ValueError):
        svd(A)


def test_inf_input():
    """A with Inf → expect ValueError raised."""
    A = np.random.randn(8, 5)
    A[0, 0] = np.inf
    with pytest.raises(ValueError):
        svd(A)


def test_non_convergence():
    """Hard matrix with max_sweeps=3 → expect RuntimeError raised with partial results."""
    A = np.random.randn(20, 20)
    with pytest.raises(RuntimeError):
        svd(A, max_sweeps=3)


def test_already_orthogonal():
    """A with orthonormal columns → converges in 1 sweep."""
    m, n = 10, 5
    A = np.asfortranarray(np.linalg.qr(np.random.randn(m, n))[0])

    U, S, Vt = svd(A)

    # For orthonormal columns, singular values should all be ≈ 1
    np.testing.assert_allclose(S, np.ones(n), atol=1e-10)
    np.testing.assert_allclose((U * S) @ Vt, A, atol=1e-12)


def test_tall_skinny():
    """A = randn(500,5) → works correctly, shapes correct."""
    m, n = 500, 5
    A = np.random.randn(m, n)

    U, S, Vt = svd(A)

    assert U.shape == (m, n)
    assert S.shape == (n,)
    assert Vt.shape == (n, n)
    # S values should be non-negative and descending
    assert np.all(S >= -1e-14)
    assert np.all(np.diff(S) <= 1e-10)
    # U and Vt should be orthonormal (thin U: n columns)
    np.testing.assert_allclose(U.T @ U, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(n), atol=1e-12)
    # Reconstruction
    np.testing.assert_allclose((U @ np.diag(S)) @ Vt, A, atol=1e-8)

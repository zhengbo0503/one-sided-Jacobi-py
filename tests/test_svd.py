"""TDD correctness tests for one_sided_jacobi.svd().

All tests are parametrized over shapes (100,50), (50,50), (200,30).
Tests FAIL naturally since svd() is not yet implemented.
Vector comparisons use projectors to avoid sign-ambiguity issues.
"""

import numpy as np
import pytest

# -- Graceful import for TDD: fails naturally when svd() is unimplemented --
try:
    from one_sided_jacobi.svd import svd
except ImportError:
    svd = None  # pragma: no cover — svd() not implemented yet

# -- conftest helper (mirrors tests/conftest.py, runs in parallel) --
try:
    from tests.conftest import compare_svd_result
except ImportError:
    compare_svd_result = None  # pragma: no cover — conftest not ready yet


# ---------------------------------------------------------------------------
# Local fixture mirroring conftest.py (ensures collection even before Task 1)
# ---------------------------------------------------------------------------
@pytest.fixture
def random_matrix(request):
    """Return a random well-conditioned float64 matrix of shape (m, n).

    Shapes are provided via pytest.mark.parametrize(indirect=True).
    Reproducible across runs using a fixed seed derived from the shape.
    """
    m, n = request.param
    rng = np.random.default_rng(hash((m, n)) % 2**31)
    A = rng.standard_normal((m, n))
    Q, R = np.linalg.qr(A)
    # Well-conditioned: singular values in [1, 2]
    return Q[:, :n] @ np.diag(np.linspace(1, 2, n))


# ===========================================================================
# Test 1 — Reconstruction error
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_reconstruction(random_matrix):
    """‖A - U @ diag(S) @ Vt‖ / ‖A‖ < 1e-12."""
    A = random_matrix
    U, S, Vt = svd(A)
    m, n = A.shape

    recon = U @ np.diag(S) @ Vt
    rel_err = np.linalg.norm(A - recon) / np.linalg.norm(A)
    assert rel_err < 1e-12, f"Reconstruction relative error {rel_err} too large"


# ===========================================================================
# Test 2 — U orthogonality
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_orthogonality_u(random_matrix):
    """‖U^T @ U - I‖ < 1e-12."""
    A = random_matrix
    U, S, Vt = svd(A)
    n = S.shape[0]

    err = np.linalg.norm(U.T @ U - np.eye(n))
    assert err < 1e-12, f"U orthogonality error {err} too large"


# ===========================================================================
# Test 3 — Vt orthogonality
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_orthogonality_v(random_matrix):
    """‖Vt @ Vt^T - I‖ < 1e-12."""
    A = random_matrix
    U, S, Vt = svd(A)
    n = S.shape[0]

    err = np.linalg.norm(Vt @ Vt.T - np.eye(n))
    assert err < 1e-12, f"Vt orthogonality error {err} too large"


# ===========================================================================
# Test 4 — Singular values sorted (non-negative, descending)
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_singular_values_sorted(random_matrix):
    """All s >= 0 and s[i] >= s[i+1] (descending order)."""
    A = random_matrix
    U, S, Vt = svd(A)

    assert np.all(S >= 0), "Singular values must be non-negative"
    assert np.all(np.diff(S) <= 0) or np.all(S[:-1] >= S[1:]), (
        "Singular values must be in descending order"
    )


# ===========================================================================
# Test 5 — Output shapes
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_shapes(random_matrix):
    """U.shape == (m, n), S.shape == (n,), Vt.shape == (n, n) for A(m, n)."""
    A = random_matrix
    m, n = A.shape
    U, S, Vt = svd(A)

    assert U.shape == (m, n), f"U shape {U.shape} != ({m}, {n})"
    assert S.shape == (n,), f"S shape {S.shape} != ({n},)"
    assert Vt.shape == (n, n), f"Vt shape {Vt.shape} != ({n}, {n})"


# ===========================================================================
# Test 6 — Singular value accuracy vs NumPy (well-conditioned only)
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_against_numpy_svd_wellcond(random_matrix):
    """Relative singular value error < 10 * n * eps for cond(A) < 1e8."""
    A = random_matrix
    U_jac, S_jac, Vt_jac = svd(A)

    _, S_ref, _ = np.linalg.svd(A, full_matrices=False)

    # Skip if matrix is ill-conditioned
    cond = np.max(S_ref) / np.min(S_ref)
    if cond > 1e8:
        pytest.skip(f"Matrix too ill-conditioned (cond={cond:.1e}) for exact comparison")

    n = S_jac.shape[0]
    rel_err = np.linalg.norm(S_jac - S_ref) / np.linalg.norm(S_ref)
    tolerance = 10 * n * np.finfo(np.float64).eps
    assert rel_err < tolerance, (
        f"Relative singular value error {rel_err:.2e} exceeds {tolerance:.2e}"
    )


# ===========================================================================
# Test 7 — Vector accuracy vs NumPy (projector comparison)
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_against_numpy_svd_vectors(random_matrix):
    """Projector comparison: ‖U@U^T - Uref@Uref^T‖ < 1e-10, same for V."""
    A = random_matrix
    U_jac, S_jac, Vt_jac = svd(A)

    U_ref, S_ref, Vt_ref = np.linalg.svd(A, full_matrices=False)

    # U projector: U @ U^T
    proj_u_jac = U_jac @ U_jac.T
    proj_u_ref = U_ref @ U_ref.T
    err_u = np.linalg.norm(proj_u_jac - proj_u_ref)
    assert err_u < 1e-10, f"U projector error {err_u:.2e} too large"

    # V projector: Vt^T @ Vt = V @ V^T  (Vt is (n,n) → V is (n,n))
    V_jac = Vt_jac.T
    V_ref = Vt_ref.T
    proj_v_jac = V_jac @ V_jac.T  # same as Vt_jac.T @ Vt_jac
    proj_v_ref = V_ref @ V_ref.T
    err_v = np.linalg.norm(proj_v_jac - proj_v_ref)
    assert err_v < 1e-10, f"V projector error {err_v:.2e} too large"


# ===========================================================================
# Test 8 — Output dtype consistency
# ===========================================================================
@pytest.mark.parametrize(
    "random_matrix", [(100, 50), (50, 50), (200, 30)], indirect=True
)
def test_dtype_consistency(random_matrix):
    """All outputs must be float64."""
    A = random_matrix
    U, S, Vt = svd(A)

    assert U.dtype == np.float64, f"U dtype {U.dtype} != float64"
    assert S.dtype == np.float64, f"S dtype {S.dtype} != float64"
    assert Vt.dtype == np.float64, f"Vt dtype {Vt.dtype} != float64"

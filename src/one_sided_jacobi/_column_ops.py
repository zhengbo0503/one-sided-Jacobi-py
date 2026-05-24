import numpy as np

FP_EPS = np.finfo(np.float64).eps
FP_TINY = np.finfo(np.float64).tiny
SFMIN = FP_TINY / FP_EPS
ROOTSFMIN = np.sqrt(SFMIN)
ROOTBIG = 1.0 / ROOTSFMIN


def safe_column_norm(A: np.ndarray, col_idx: int) -> np.float64:
    col = A[:, col_idx]
    abs_col = np.abs(col)
    if abs_col.min() >= ROOTSFMIN and abs_col.max() <= ROOTBIG:
        return np.float64(np.linalg.norm(col))

    max_abs = abs_col.max()
    if max_abs == 0.0:
        return np.float64(0.0)
    col_scaled = col / max_abs
    return np.float64(max_abs * np.linalg.norm(col_scaled))


def safe_column_dot(
    A: np.ndarray, col_p: int, col_q: int
) -> np.float64:
    c_p = A[:, col_p]
    c_q = A[:, col_q]
    norm_p = safe_column_norm(A, col_p)
    norm_q = safe_column_norm(A, col_q)

    if norm_p > ROOTBIG and norm_q > ROOTBIG:
        scale = 1.0 / np.sqrt(norm_p * norm_q)
        return np.float64(np.dot(c_p * scale, c_q * scale) / (scale * scale))
    return np.float64(np.dot(c_p, c_q))


def column_scale(A: np.ndarray, col_idx: int, scale_factor: float) -> None:
    A[:, col_idx] *= scale_factor


def swap_columns(A: np.ndarray, p: int, q: int) -> None:
    A[:, [p, q]] = A[:, [q, p]]


def init_column_norms(A: np.ndarray) -> np.ndarray:
    n = A.shape[1]
    sva = np.empty(n, dtype=np.float64)
    for j in range(n):
        sva[j] = safe_column_norm(A, j)
    return sva


def recompute_column_norm(A: np.ndarray, col_idx: int) -> np.float64:
    return safe_column_norm(A, col_idx)

import math
from typing import Optional, Tuple

import numpy as np

EPSLN: float = float(np.finfo(np.float64).eps)
SFMIN: float = float(np.finfo(np.float64).tiny / EPSLN)
SMALL: float = SFMIN / EPSLN
BIG: float = 1.0 / SFMIN
ROOTEPS: float = np.sqrt(EPSLN)
ROOTSFMIN: float = np.sqrt(SFMIN)
ROOTBIG: float = 1.0 / ROOTSFMIN
BIGTHETA: float = 1.0 / ROOTEPS


def safe_norm_sq(x: np.ndarray) -> Tuple[float, float]:
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    n = x_flat.size
    if n == 0:
        return 0.0, 1.0

    scale = 0.0
    ssq = 1.0

    for i in range(n):
        xi = x_flat[i]
        if xi != 0.0:
            absxi = abs(xi)
            if scale < absxi:
                ssq = 1.0 + ssq * (scale / absxi) ** 2
                scale = absxi
            else:
                ssq += (xi / scale) ** 2

    if scale == 0.0:
        return 0.0, 1.0
    return scale * np.sqrt(ssq), scale


def safe_column_norm_with_scale(
    A: np.ndarray, col_idx: int
) -> Tuple[float, float]:
    col = np.asarray(A[:, col_idx], dtype=np.float64).ravel()
    abs_col = np.abs(col)
    if np.all((abs_col >= ROOTSFMIN) & (abs_col <= ROOTBIG)):
        return float(np.linalg.norm(col)), 1.0
    return safe_norm_sq(col)


def scaling_decision_tree(aapp: float, aaqq: float, n: int) -> float:
    SN = math.sqrt(SFMIN / EPSLN)
    temp1 = math.sqrt(BIG / float(n))
    if temp1 > BIG:
        temp1 = BIG

    if aapp <= SN and aaqq >= temp1:
        scale = SN / aapp
    elif aaqq <= SN and aapp <= temp1:
        scale = SN / aaqq
    elif aaqq >= SN and aapp >= temp1:
        scale = temp1 / aaqq
    elif aaqq <= SN and aapp >= temp1:
        scale = temp1 / aapp
    else:
        scale = 1.0

    return float(scale)


def cancellation_detected(
    sva_new: float, sva_old: float, rooteps: float = ROOTEPS
) -> bool:
    if sva_old == 0.0:
        return True
    return (sva_new / sva_old) ** 2 <= rooteps


def compute_tolerance(
    ctol: Optional[float] = None,
    epsln: float = EPSLN,
    m: int = 0,
    compute_vectors: bool = False,
) -> float:
    if ctol is None:
        ctol = math.sqrt(float(m)) if compute_vectors else float(m)
    return ctol * epsln


def compute_machine_constants() -> dict:
    return {
        "EPSLN": EPSLN, "SFMIN": SFMIN, "SMALL": SMALL, "BIG": BIG,
        "ROOTEPS": ROOTEPS, "ROOTSFMIN": ROOTSFMIN, "ROOTBIG": ROOTBIG,
        "BIGTHETA": BIGTHETA,
    }

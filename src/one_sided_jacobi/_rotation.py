import numpy as np

BIGTHETA = 1.0 / np.sqrt(np.finfo(np.float64).eps)


def compute_theta(
    aapq: float,
    aapp: float,
    aaqq: float,
) -> tuple[float, float, float]:
    """Compute cot(2φ) from column norms and normalised correlation.

    aapp, aaqq are unsquared column norms.  aapq is the normalised
    correlation <p,q> / (||p||·||q||) ∈ [-1, 1].

    Returns (theta, aqoap, apoaq).  theta=0 means no rotation needed.
    """
    if aapq == 0.0 or aapp == 0.0 or aaqq == 0.0:
        aqoap = 0.0 if (aapp == 0.0 or aaqq == 0.0) else (aaqq / aapp)
        apoaq = 0.0 if (aapp == 0.0 or aaqq == 0.0) else (aapp / aaqq)
        return (0.0, aqoap, apoaq)

    aqoap = aaqq / aapp
    apoaq = aapp / aaqq
    # Do NOT use abs() — the sign is required for correct root selection.
    theta = -0.5 * (aqoap - apoaq) / aapq
    return (theta, aqoap, apoaq)


def compute_rotation_params(
    theta: float,
    aapq: float,
    aapp0: float,
    aaqq: float,
) -> tuple[float, float, float]:
    """Determine (cos φ, sin φ, tan φ) from cot(2φ).

    Handles the large-theta path (t = 0.5/theta) for numerical safety.
    """
    abs_theta = abs(theta)

    if abs_theta > BIGTHETA:
        t = 0.5 / theta
        cs = 1.0 / np.sqrt(1.0 + t * t)
        sn = t * cs
        return (cs, sn, t)

    thsign = -1.0 if aapq < 0.0 else 1.0
    t = 1.0 / (theta + thsign * np.sqrt(1.0 + theta * theta))
    cs = np.sqrt(1.0 / (1.0 + t * t))
    sn = t * cs
    return (cs, sn, t)


def rotok_check(aapp: float, aaqq: float, small: float) -> bool:
    """Return True if the fast rotation path is numerically safe."""
    return (small * aapp) <= aaqq

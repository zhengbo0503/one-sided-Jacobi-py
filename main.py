#!/usr/bin/env python3
"""Interactive demo of the one-sided Jacobi SVD algorithm.

Usage:  source .venv/bin/activate && python main.py
"""

import time
import numpy as np
from one_sided_jacobi import svd

MATRIX_SHAPE = (80, 40)
RANDOM_SEED = 42
CONDITION_NUMBER = 10
N_TRIALS = 3
PLOT_ENABLED = True


def make_matrix(m: int, n: int, kappa: float, rng: np.random.Generator):
    r = min(m, n)
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    Vt, _ = np.linalg.qr(rng.standard_normal((n, r)))
    s = np.logspace(-np.log10(kappa), 0, r)
    return (U * s) @ Vt.T


def validate(U, S, Vt, A, npy_S):
    recon = U @ np.diag(S) @ Vt
    rel_err = np.linalg.norm(A - recon) / max(np.linalg.norm(A), 1e-300)
    u_err = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
    v_err = np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0]))
    sv_err = np.max(np.abs(np.sort(S)[::-1] - np.sort(npy_S)[::-1])
                    / np.maximum(np.sort(npy_S)[::-1], 1e-300))
    print(f"  reconstruction : {rel_err:.1e}")
    print(f"  U orthogonality: {u_err:.1e}")
    print(f"  V orthogonality: {v_err:.1e}")
    print(f"  S vs numpy     : {sv_err:.1e}")
    return rel_err, u_err, v_err, sv_err


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    m, n = MATRIX_SHAPE

    print("=" * 56)
    print(" One-Sided Jacobi SVD — Interactive Demo")
    print("=" * 56)
    print(f" matrix size  : {m}x{n}")
    print(f" condition nr : {CONDITION_NUMBER:.0e}")
    print(f" trials       : {N_TRIALS}")
    print(f" max sweeps   : 30\n")

    times_ours, times_npy, errors = [], [], []

    for trial in range(N_TRIALS):
        seed = RANDOM_SEED + trial
        rng_trial = np.random.default_rng(seed)
        A = make_matrix(m, n, CONDITION_NUMBER, rng_trial)

        print(f"--- trial {trial + 1}/{N_TRIALS} (seed={seed}) ---")

        t0 = time.perf_counter()
        U, S, Vt = svd(A)
        elapsed_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        nU, nS, nVt = np.linalg.svd(A, full_matrices=False)
        elapsed_npy = time.perf_counter() - t0

        times_ours.append(elapsed_ours)
        times_npy.append(elapsed_npy)

        r_err, u_err, v_err, s_err = validate(U, S, Vt, A, nS)
        errors.append((r_err, u_err, v_err, s_err))

        ratio = elapsed_ours / max(elapsed_npy, 1e-9)
        print(f"  ours: {elapsed_ours:.4f}s  numpy: {elapsed_npy:.4f}s  "
              f"ratio: {ratio:.0f}x\n")

    print("=" * 56)
    print(" Summary")
    print("=" * 56)
    avg_ours = np.mean(times_ours)
    avg_npy = np.mean(times_npy)
    r_errs, u_errs, v_errs, s_errs = zip(*errors)
    print(f" avg time   ours: {avg_ours:.4f}s  numpy: {avg_npy:.4f}s  "
          f"({avg_ours / max(avg_npy, 1e-9):.0f}x slower)")
    print(f" max reconstruction : {max(r_errs):.1e}")
    print(f" max U orthogonality: {max(u_errs):.1e}")
    print(f" max V orthogonality: {max(v_errs):.1e}")
    print(f" max SV deviation   : {max(s_errs):.1e}")

    print("\n edge cases:")
    for label, A in [
        ("zero 10x5", np.zeros((10, 5))),
        ("identity 10", np.eye(10)),
        ("NaN -> ValueError", None),
    ]:
        if A is None:
            try:
                svd(np.array([[np.nan]]))
            except ValueError:
                print("  NaN raised ValueError")
        else:
            try:
                U, S, Vt = svd(A)
                ok = np.allclose(U @ np.diag(S) @ Vt, A, atol=1e-12)
                print(f"  {label}: {'ok' if ok else 'FAIL'}")
            except Exception as e:
                print(f"  {label}: {e}")

    if PLOT_ENABLED:
        try:
            import matplotlib.pyplot as plt
            _, S_small, _ = svd(make_matrix(100, 20, CONDITION_NUMBER, rng))
            _, nS_small, _ = np.linalg.svd(
                make_matrix(100, 20, CONDITION_NUMBER, rng),
                full_matrices=False,
            )
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.semilogy(np.sort(S_small)[::-1], "o-", label="ours")
            ax1.semilogy(np.sort(nS_small)[::-1], "x--", label="numpy")
            ax1.set_title("Singular Values")
            ax1.set_xlabel("index")
            ax1.legend()
            ax2.semilogy(
                np.abs(np.sort(S_small)[::-1] - np.sort(nS_small)[::-1])
                / np.sort(nS_small)[::-1],
                "k.-",
            )
            ax2.set_title("Relative SV Error vs numpy")
            ax2.set_xlabel("index")
            ax2.set_ylabel("relative error")
            plt.tight_layout()
            plt.savefig("svd_comparison.png", dpi=100)
            print("\n plot saved -> svd_comparison.png")
        except ImportError:
            print("\n matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()

"""
ransac_scale.py

RANSAC-based affine estimator for monocular depth fusion.

DepthAnythingV2 outputs relative disparity (larger = CLOSER), so the model is:

    D_metric = s * D_disp + t       (s < 0 in general)

given sparse metric anchors (a row from the RealSense GT depth map, or LiDAR).
Minimum sample size = 2 (one point per parameter).

Can be imported or run as a standalone self-test.
"""

import numpy as np


def ransac_affine(
    d_disp: np.ndarray,
    d_real: np.ndarray,
    n_iters: int = 2000,
    inlier_thresh: float = 0.15,   # meters absolute error
    max_depth: float = 20.0,       # reject clearly invalid depth predictions
) -> tuple[float, float, np.ndarray]:
    """
    Estimate affine parameters (s, t) such that s*d_disp + t ≈ d_real.

    Note: s will typically be negative because DAv2 disparity is inverted
    relative to metric depth (higher disparity = closer object).

    Parameters
    ----------
    d_disp : 1-D array
        Relative disparity from DAv2 at anchor pixels (> 0).
    d_real : 1-D array
        Metric depth at the same anchor pixels (meters, > 0).
    n_iters : int
        RANSAC iteration count.
    inlier_thresh : float
        Absolute error threshold (meters).
    max_depth : float
        Predicted depths outside [0, max_depth] are treated as invalid.

    Returns
    -------
    s : float   Scale (typically negative).
    t : float   Shift.
    inlier_mask : bool array, same length as d_disp.
    """
    assert len(d_disp) == len(d_real)

    valid = (d_real > 0) & (d_disp > 0)
    disp_v = d_disp[valid]
    real_v = d_real[valid]
    n = len(disp_v)

    if n < 2:
        print(f"[ransac] Only {n} valid anchor points — cannot fit affine model")
        return -1.0, 0.0, np.zeros(len(d_disp), dtype=bool)

    def fit_two(i, j):
        """Exactly fit s, t from two points."""
        A = np.array([[disp_v[i], 1.0],
                      [disp_v[j], 1.0]])
        b = np.array([real_v[i], real_v[j]])
        if abs(np.linalg.det(A)) < 1e-8:
            return None, None
        st = np.linalg.solve(A, b)
        return float(st[0]), float(st[1])

    def count_inliers(s, t):
        pred = np.clip(s * disp_v + t, 0, max_depth)
        return np.abs(pred - real_v) < inlier_thresh

    # Initialise with full least-squares fit
    A_all = np.stack([disp_v, np.ones(n)], axis=1)
    st0, _, _, _ = np.linalg.lstsq(A_all, real_v, rcond=None)
    best_s, best_t = float(st0[0]), float(st0[1])
    best_mask = count_inliers(best_s, best_t)
    best_n = int(best_mask.sum())

    rng = np.random.default_rng(42)
    for _ in range(n_iters):
        i, j = rng.choice(n, 2, replace=False)
        s_c, t_c = fit_two(i, j)
        if s_c is None:
            continue
        mask = count_inliers(s_c, t_c)
        if int(mask.sum()) > best_n:
            best_n = int(mask.sum())
            # Refit with all inliers via least squares
            if mask.sum() >= 2:
                A_in = np.stack([disp_v[mask], np.ones(mask.sum())], axis=1)
                st_in, _, _, _ = np.linalg.lstsq(A_in, real_v[mask], rcond=None)
                best_s, best_t = float(st_in[0]), float(st_in[1])
                best_mask = count_inliers(best_s, best_t)
                best_n = int(best_mask.sum())

    # Map inlier mask back to full input length
    full_mask = np.zeros(len(d_disp), dtype=bool)
    valid_idx = np.where(valid)[0]
    full_mask[valid_idx[best_mask]] = True

    print(f"[ransac] s={best_s:.4f}  t={best_t:.4f}  "
          f"inliers={best_mask.sum()}/{n}  ({100*best_mask.mean():.1f}%)")
    return best_s, best_t, full_mask


# ---- Quick self-test ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    TRUE_S, TRUE_T = -1.2, 6.0    # realistic DAv2-like parameters
    n = 80
    d_disp = rng.uniform(0.5, 5.0, n)
    d_real = TRUE_S * d_disp + TRUE_T + rng.normal(0, 0.05, n)
    d_real = np.clip(d_real, 0.1, 20.0)

    # 15% outliers
    outlier_idx = rng.choice(n, 12, replace=False)
    d_real[outlier_idx] = rng.uniform(0.1, 8.0, 12)

    s, t, mask = ransac_affine(d_disp, d_real, n_iters=2000, inlier_thresh=0.20)
    print(f"True  s={TRUE_S:.4f}  t={TRUE_T:.4f}")
    print(f"Est   s={s:.4f}  t={t:.4f}")
    err_s = abs(s - TRUE_S) / abs(TRUE_S) * 100
    err_t = abs(t - TRUE_T)
    print(f"Error s={err_s:.2f}%  t={err_t:.4f}m")
    assert err_s < 5.0, f"Scale error too large: {err_s:.1f}%"
    print("Self-test passed.")

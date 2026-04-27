"""
Calibration math: projection geometry and Nelder-Mead optimizer.

Coordinate conventions
  LiDAR frame (ROS 2D): x=forward, y=left, z=up; scan is in the z=0 plane.
  Camera frame (OpenCV): x=right,   y=down,  z=forward.
  T_cam_lidar: 4x4 homogeneous transform — takes points FROM lidar frame TO camera frame.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def euler_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles (rad, ZYX intrinsic) → 3x3 rotation matrix.

    Applied order: first Rx, then Ry, then Rz  →  R = Rz @ Ry @ Rx.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,  0,   0],
                   [0,  cx, -sx],
                   [0,  sx,  cx]], dtype=np.float64)

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float64)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def params_to_transform(params) -> np.ndarray:
    """[tx, ty, tz, rx, ry, rz] → 4x4 T_cam_lidar homogeneous matrix."""
    tx, ty, tz, rx, ry, rz = params
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_to_rot(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def transform_to_params(T: np.ndarray):
    """4x4 T_cam_lidar → [tx, ty, tz, rx, ry, rz] (ZYX Euler, rad)."""
    R = T[:3, :3]
    tx, ty, tz = T[:3, 3]
    # ZYX decomposition
    ry = np.arcsin(-R[2, 0])
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return np.array([tx, ty, tz, rx, ry, rz], dtype=np.float64)


# ---------------------------------------------------------------------------
# Scan → 3-D points
# ---------------------------------------------------------------------------

def scan_to_points(ranges, angle_min: float, angle_increment: float,
                   range_min: float, range_max: float) -> np.ndarray:
    """Convert a LaserScan to Nx3 float64 in LiDAR frame (z = 0 plane)."""
    r = np.asarray(ranges, dtype=np.float64)
    n = len(r)
    angles = angle_min + np.arange(n) * angle_increment
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    pts = np.zeros((int(valid.sum()), 3), dtype=np.float64)
    pts[:, 0] = r[valid] * np.cos(angles[valid])
    pts[:, 1] = r[valid] * np.sin(angles[valid])
    # z = 0 (2-D scan plane)
    return pts


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(pts3d: np.ndarray, K: np.ndarray,
                   T_cam_lidar: np.ndarray):
    """
    Project Nx3 LiDAR points into the camera image.

    Returns
      u, v  : float arrays of pixel positions for points in front of camera
      mask  : bool array, length N — True where Z_cam > 0
    """
    n = pts3d.shape[0]
    if n == 0:
        return np.array([]), np.array([]), np.zeros(0, dtype=bool)

    pts_h = np.hstack([pts3d, np.ones((n, 1))])       # Nx4
    pts_cam = (T_cam_lidar @ pts_h.T).T               # Nx4

    mask = pts_cam[:, 2] > 0.01
    pc = pts_cam[mask]

    u = K[0, 0] * (pc[:, 0] / pc[:, 2]) + K[0, 2]
    v = K[1, 1] * (pc[:, 1] / pc[:, 2]) + K[1, 2]
    return u, v, mask


# ---------------------------------------------------------------------------
# Gradient score
# ---------------------------------------------------------------------------

def compute_gradient(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Return Sobel gradient-magnitude map from a BGR or gray image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def sample_gradient(grad: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Sum gradient values at (u, v) pixel positions (nearest-neighbour)."""
    if u.size == 0:
        return 0.0
    h, w = grad.shape
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    in_bounds = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not in_bounds.any():
        return 0.0
    return float(grad[vi[in_bounds], ui[in_bounds]].sum())


# ---------------------------------------------------------------------------
# Nelder-Mead optimizer (pure NumPy — no scipy required)
# ---------------------------------------------------------------------------

def nelder_mead(f, x0, initial_step=None, max_iter: int = 3000,
                tol: float = 1e-6, verbose: bool = False):
    """
    Minimize scalar function f via the Nelder-Mead simplex method.

    Parameters
      f            : callable, returns a scalar
      x0           : 1-D array, starting point
      initial_step : per-parameter step size for building the initial simplex
      max_iter     : maximum function evaluations
      tol          : convergence threshold on function-value spread

    Returns
      x_best : 1-D array, optimizer solution
      f_best : float, function value at x_best
    """
    n = len(x0)
    if initial_step is None:
        initial_step = np.full(n, 0.05)
    step = np.asarray(initial_step, dtype=float)

    # Build initial simplex: x0 + unit perturbation in each dimension
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        xi[i] += step[i]
        simplex.append(xi)
    fvals = [f(x) for x in simplex]
    evals = n + 1

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for iteration in range(max_iter):
        # Sort ascending
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals   = [fvals[i]   for i in order]

        if verbose and iteration % 200 == 0:
            print(f"    iter {iteration:4d}  best={fvals[0]:.4f}  "
                  f"params=[{' '.join(f'{v:.4f}' for v in simplex[0])}]")

        spread = np.max(np.abs(np.array(fvals) - fvals[0]))
        if spread < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr); evals += 1

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1]   = fr
        elif fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe); evals += 1
            if fe < fr:
                simplex[-1] = xe
                fvals[-1]   = fe
            else:
                simplex[-1] = xr
                fvals[-1]   = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc); evals += 1
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1]   = fc
            else:
                # Shrink
                best = simplex[0]
                new_s, new_f = [best], [fvals[0]]
                for x in simplex[1:]:
                    xs = best + sigma * (x - best)
                    new_s.append(xs)
                    new_f.append(f(xs))
                    evals += 1
                simplex, fvals = new_s, new_f

    return np.array(simplex[0]), fvals[0]

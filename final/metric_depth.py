"""
metric_depth.py

Full offline pipeline:
  1. Load pre-computed relative disparity (from depth_anything_infer.py)
  2. Load RealSense ground-truth depth; extract one anchor row near the bottom
  3. RANSAC affine fit in disparity space: D_metric = s * D_disp + t  (s < 0)
  4. Apply to full frame → metric depth map, clipped to [0, MAX_DEPTH]
  5. Evaluate AbsRel / MAE / RMSE vs. ground truth
  6. Save visualizations and summary

Key insight: DAv2 outputs inverse depth (disparity) — higher = closer.
So the affine model D_metric = s*D_disp + t has s < 0.
Do NOT invert the disparity before fitting.

CLI arguments:
  --data-dir DIR          Root dir with rgb/, depth/, rel_depth/ sub-folders.
                          Defaults to the current working directory.
  --frame N               Process only frame_NNNNN; default = all frames.
  --max-gt-depth M        Cap RealSense GT at M metres for fitting & evaluation.
                          Output goes to output_<M>m/ (e.g. output_3m/).
                          Default: no cap (full range).
  --anchor-row R          Fix the anchor row to row R (0-indexed).
                          Omit to auto-select the row with the most GT depth
                          spread each frame (recommended with --max-gt-depth).
  --no-viz                Skip saving visualisation panels and profile plots.

Examples:
    # Full pipeline on all frames, auto anchor row, 3 m GT cap
    PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 \\
        python metric_depth.py --data-dir testing_data --max-gt-depth 3.0

    # Single frame, fixed anchor row, no GT cap
    PYTHONNOUSERSITE=1 conda run -n vision_lab_v2 \\
        python metric_depth.py --data-dir testing_data --frame 0 --anchor-row 430
"""

import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ransac_scale import ransac_affine

# ---- Config ----
RGB_DIR       = Path("rgb")
DEPTH_DIR     = Path("depth")       # RealSense GT (.npy, float32 meters)
REL_DIR       = Path("rel_depth")   # DAv2 disparity from depth_anything_infer.py

ANCHOR_ROW    = None                # None = auto-select per frame
RANSAC_ITERS  = 2000
INLIER_THRESH = 0.20               # meters — loose enough to find real inliers
MAX_DEPTH     = 20.0               # clip predictions beyond this (meters)

# Set at runtime by --max-gt-depth; None means no cap (use all GT pixels)
MAX_GT_DEPTH: float | None = None

EPS = 1e-6
MIN_ANCHOR_PIXELS = 20   # minimum valid pixels required on a candidate row


def best_anchor_row(d_disp: np.ndarray, d_gt: np.ndarray,
                    gt_cap: float = np.inf) -> int:
    """
    Return the image row with the greatest GT depth range among valid pixels
    (both disp > 0 and 0 < gt <= gt_cap).  Falls back to row 400 if no row
    meets MIN_ANCHOR_PIXELS.
    """
    h = d_gt.shape[0]
    best_row, best_range = 400, -1.0
    for row in range(h):
        ag = d_gt[row, :]
        ad = d_disp[row, :]
        v  = (ad > 0) & (ag > 0) & (ag <= gt_cap)
        if v.sum() < MIN_ANCHOR_PIXELS:
            continue
        gt_range = float(ag[v].max() - ag[v].min())
        if gt_range > best_range:
            best_range = gt_range
            best_row   = row
    return best_row


def _out_dir() -> Path:
    """Return output directory based on the active MAX_GT_DEPTH cap."""
    if MAX_GT_DEPTH is not None:
        label = f"{MAX_GT_DEPTH:g}".replace(".", "p")   # 3.0 → "3", 2.5 → "2p5"
        return Path(f"output_{label}m")
    return Path("output")


def load_disp(stem: str) -> np.ndarray:
    path = REL_DIR / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run depth_anything_infer.py first.")
    return np.load(str(path)).astype(np.float32)


def load_gt(stem: str) -> np.ndarray:
    path = DEPTH_DIR / f"{stem}.npy"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(str(path)).astype(np.float32)


def apply_affine(d_disp: np.ndarray, s: float, t: float) -> np.ndarray:
    """D_metric = clip(s*D_disp + t, 0, MAX_DEPTH)."""
    d_metric = s * d_disp + t
    return np.clip(d_metric, 0.0, MAX_DEPTH).astype(np.float32)


def evaluate(d_metric: np.ndarray, d_gt: np.ndarray,
             depth_min: float = 0.0, depth_max: float = MAX_DEPTH) -> dict:
    """
    Error metrics on valid (non-zero GT) pixels.
    Optionally restrict to a GT depth range [depth_min, depth_max] to
    match the range covered by the anchor row.
    """
    valid = (d_gt > depth_min) & (d_gt <= depth_max) & (d_metric > 0)
    if valid.sum() == 0:
        return {"AbsRel": np.nan, "MAE": np.nan, "RMSE": np.nan, "n_valid": 0}
    pred, gt = d_metric[valid], d_gt[valid]
    return {
        "AbsRel":  float(np.mean(np.abs(pred - gt) / gt)),
        "MAE":     float(np.mean(np.abs(pred - gt))),
        "RMSE":    float(np.sqrt(np.mean((pred - gt) ** 2))),
        "n_valid": int(valid.sum()),
    }


def colorize(depth: np.ndarray, vmin: float = 0.0, vmax: float = None) -> np.ndarray:
    d = depth.copy()
    valid = d > 0
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98))
    d = np.clip((d - vmin) / (vmax - vmin + EPS), 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colored[~valid] = 0
    return colored


def make_colorbar(height: int, vmin: float, vmax: float, bar_width: int = 28) -> np.ndarray:
    """
    Vertical colorbar matching the INFERNO depth colormap.
    Top = vmax (bright / far), bottom = vmin (dark / near).
    Returns a BGR uint8 image of shape (height, bar_width + label_px, 3).
    """
    label_px = 52                           # pixels reserved for tick labels
    total_w  = bar_width + label_px
    canvas   = np.zeros((height, total_w, 3), dtype=np.uint8)

    # Draw the gradient bar (vmax at top → vmin at bottom)
    gradient = np.linspace(255, 0, height, dtype=np.uint8)[:, None]   # (H, 1)
    bar_color = cv2.applyColorMap(gradient, cv2.COLORMAP_INFERNO)      # (H, 1, 3)
    canvas[:, :bar_width] = np.repeat(bar_color, bar_width, axis=1)

    # Tick marks and labels
    n_ticks = 7
    for i in range(n_ticks):
        frac      = i / (n_ticks - 1)                             # 0 → 1
        depth_val = vmax - frac * (vmax - vmin)                   # vmax → vmin
        y         = int(round(frac * (height - 1)))               # 0 → height-1
        cv2.line(canvas, (bar_width, y), (bar_width + 5, y), (200, 200, 200), 1)
        label = f"{depth_val:.1f}m"
        cv2.putText(canvas, label, (bar_width + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1,
                    cv2.LINE_AA)

    # "far" / "near" labels at top and bottom
    cv2.putText(canvas, "far",  (2, 12),       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "near", (2, height-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def run_frame(stem: str, save_viz: bool = True) -> dict:
    d_disp = load_disp(stem)
    d_gt   = load_gt(stem)

    gt_cap = MAX_GT_DEPTH if MAX_GT_DEPTH is not None else np.inf
    row = ANCHOR_ROW if ANCHOR_ROW is not None else best_anchor_row(d_disp, d_gt, gt_cap)

    anchor_disp = d_disp[row, :]
    anchor_real = d_gt[row, :]

    # Apply GT cap: mask out pixels beyond MAX_GT_DEPTH for fitting & evaluation
    gt_cap = MAX_GT_DEPTH if MAX_GT_DEPTH is not None else np.inf
    anchor_real_capped = anchor_real.copy()
    anchor_real_capped[anchor_real_capped > gt_cap] = 0.0

    # Anchor row GT depth range — used for fair evaluation
    valid_anchor = anchor_real_capped > 0
    anchor_gt_max = float(anchor_real_capped[valid_anchor].max()) if valid_anchor.any() else MAX_DEPTH

    # Step 1: direct least-squares affine fit over valid anchor pixels within GT cap
    # (D_metric = s*D_disp + t, s < 0 because DAv2 disparity is inverse depth)
    anchor_real = anchor_real_capped   # use capped values from here on
    valid_anchor = (anchor_disp > 0) & (anchor_real > 0)
    A_ls = np.stack([anchor_disp[valid_anchor], np.ones(valid_anchor.sum())], axis=1)
    st_ls, _, _, _ = np.linalg.lstsq(A_ls, anchor_real[valid_anchor], rcond=None)
    s_ls, t_ls = float(st_ls[0]), float(st_ls[1])

    # Step 2: RANSAC outlier rejection — refit on inliers only
    # Use relative threshold (30%) to handle wide depth range in anchor row
    pred_ls = np.clip(s_ls * anchor_disp + t_ls, 0, MAX_DEPTH)
    rel_err = np.abs(pred_ls - anchor_real) / (anchor_real + 1e-6)
    inlier_mask = valid_anchor & (rel_err < 0.30)

    if inlier_mask.sum() >= 2:
        A_in = np.stack([anchor_disp[inlier_mask], np.ones(inlier_mask.sum())], axis=1)
        st_in, _, _, _ = np.linalg.lstsq(A_in, anchor_real[inlier_mask], rcond=None)
        s, t = float(st_in[0]), float(st_in[1])
    else:
        s, t = s_ls, t_ls

    print(f"[fit] s={s:.4f}  t={t:.4f}  "
          f"inliers={inlier_mask.sum()}/{valid_anchor.sum()}  "
          f"({100*inlier_mask.mean():.1f}%)")

    d_metric = apply_affine(d_disp, s, t)

    # Evaluation 1: full frame, GT pixels within cap
    eval_gt_max = gt_cap if gt_cap != np.inf else MAX_DEPTH
    d_gt_eval = d_gt.copy()
    d_gt_eval[d_gt_eval > gt_cap] = 0.0
    m_full = evaluate(d_metric, d_gt_eval)
    # Evaluation 2: pixels within anchor-row depth range (more fair)
    m_range = evaluate(d_metric, d_gt_eval, depth_max=anchor_gt_max)

    metrics = {
        "frame": stem,
        "anchor_row": row,
        "scale_s": s,
        "shift_t": t,
        "n_inliers": int(inlier_mask.sum()),
        "anchor_gt_max": anchor_gt_max,
        # Full frame
        "AbsRel_full": m_full["AbsRel"],
        "MAE_full":    m_full["MAE"],
        "RMSE_full":   m_full["RMSE"],
        # Anchor depth range only
        "AbsRel": m_range["AbsRel"],
        "MAE":    m_range["MAE"],
        "RMSE":   m_range["RMSE"],
        "n_valid_range": m_range["n_valid"],
    }

    if save_viz:
        _out_dir().mkdir(exist_ok=True)

        rgb_path = RGB_DIR / f"{stem}.png"
        rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else \
              np.zeros((d_gt.shape[0], d_gt.shape[1], 3), dtype=np.uint8)

        vmax = float(np.percentile(d_gt[d_gt > 0], 95)) if (d_gt > 0).any() else 8.0
        c_disp   = colorize(d_disp)                  # relative disparity (no shared scale)
        c_metric = colorize(d_metric, vmax=vmax)
        c_gt     = colorize(d_gt,     vmax=vmax)

        # Draw anchor row + RANSAC dots on metric panel
        cv2.line(c_metric, (0, row), (c_metric.shape[1]-1, row), (255,255,255), 1)
        cv2.line(c_gt,     (0, row), (c_gt.shape[1]-1,     row), (255,255,255), 1)
        for col in range(len(inlier_mask)):
            if anchor_real[col] > 0:
                color = (0, 255, 0) if inlier_mask[col] else (0, 0, 255)
                cv2.circle(c_metric, (col, row), 2, color, -1)

        colorbar = make_colorbar(rgb.shape[0], vmin=0.0, vmax=vmax)
        panel = np.hstack([rgb, c_disp, c_metric, c_gt, colorbar])
        label = f"{stem}  s={s:.3f} t={t:.3f}  MAE={m_range['MAE']:.3f}m  AbsRel={m_range['AbsRel']:.3f}"
        cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(str(_out_dir() / f"{stem}_panel.jpg"), panel)

        # Depth profile along anchor row
        cols = np.arange(d_gt.shape[1])
        pred_row = d_metric[row, :]
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(cols, anchor_real,  label="GT (RealSense)", color="black",  lw=1.5)
        ax.plot(cols, pred_row,     label=f"Predicted  s={s:.3f} t={t:.3f}", color="royalblue", lw=1.2, alpha=0.85)
        in_c  = cols[inlier_mask]
        out_c = cols[~inlier_mask & (anchor_real > 0)]
        ax.scatter(in_c,  anchor_real[inlier_mask],                       s=12, c="lime",   zorder=5, label="Inliers")
        ax.scatter(out_c, anchor_real[~inlier_mask & (anchor_real > 0)],  s=12, c="red",    zorder=5, label="Outliers")
        ax.set_xlabel("Column (pixel)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"{stem} | anchor row={row} (auto) | "
                     f"MAE={m_range['MAE']:.3f}m  AbsRel={m_range['AbsRel']:.3f}  "
                     f"RMSE={m_range['RMSE']:.3f}m")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0, top=max(anchor_gt_max * 1.1, 1.0))
        fig.tight_layout()
        fig.savefig(str(_out_dir() / f"{stem}_profile.png"), dpi=120)
        plt.close(fig)

    # Collect flattened (pred, gt) pairs for bucket analysis
    valid_px = (d_gt > 0) & (d_metric > 0)
    metrics["_pred_flat"] = d_metric[valid_px]
    metrics["_gt_flat"]   = d_gt[valid_px]

    return metrics


def plot_bucket_errors(all_metrics: list, bucket_size: float = 0.5,
                       out_path: str = None) -> None:
    """
    Aggregate MAE, AbsRel, and pixel count across all frames in 50 cm depth
    buckets, then save a two-axis bar chart.
    """
    # Pool all pixels across frames
    pred_all = np.concatenate([m["_pred_flat"] for m in all_metrics])
    gt_all   = np.concatenate([m["_gt_flat"]   for m in all_metrics])

    gt_max  = float(np.percentile(gt_all, 99))
    edges   = np.arange(0, gt_max + bucket_size, bucket_size)
    n_bins  = len(edges) - 1

    mae_per_bin    = np.full(n_bins, np.nan)
    absrel_per_bin = np.full(n_bins, np.nan)
    rmse_per_bin   = np.full(n_bins, np.nan)
    count_per_bin  = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (gt_all >= lo) & (gt_all < hi)
        if mask.sum() == 0:
            continue
        p, g = pred_all[mask], gt_all[mask]
        mae_per_bin[i]    = float(np.mean(np.abs(p - g)))
        absrel_per_bin[i] = float(np.mean(np.abs(p - g) / g))
        rmse_per_bin[i]   = float(np.sqrt(np.mean((p - g) ** 2)))
        count_per_bin[i]  = int(mask.sum())

    bin_centers = (edges[:-1] + edges[1:]) / 2
    bin_labels  = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(n_bins)]

    # ---- Figure ----
    fig, axes = plt.subplots(3, 1, figsize=(max(10, n_bins * 0.55), 10),
                             sharex=True)
    bar_kw = dict(width=0.8, edgecolor="white", linewidth=0.4)
    x = np.arange(n_bins)

    # --- MAE ---
    colors_mae = plt.cm.plasma(np.linspace(0.15, 0.85, n_bins))
    axes[0].bar(x, mae_per_bin, color=colors_mae, **bar_kw)
    axes[0].set_ylabel("MAE (m)", fontsize=11)
    axes[0].set_title("Error by GT Depth Bucket (50 cm bins, all frames pooled)",
                       fontsize=12, fontweight="bold")
    for xi, v in enumerate(mae_per_bin):
        if not np.isnan(v):
            axes[0].text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=6.5, color="black")

    # --- AbsRel ---
    colors_rel = plt.cm.plasma(np.linspace(0.15, 0.85, n_bins))
    axes[1].bar(x, absrel_per_bin, color=colors_rel, **bar_kw)
    axes[1].set_ylabel("AbsRel", fontsize=11)
    axes[1].axhline(0.20, color="red", lw=1, ls="--", label="AbsRel = 0.20")
    axes[1].legend(fontsize=8)
    for xi, v in enumerate(absrel_per_bin):
        if not np.isnan(v):
            axes[1].text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=6.5, color="black")

    # --- Pixel count ---
    axes[2].bar(x, count_per_bin / 1e3, color="steelblue", **bar_kw)
    axes[2].set_ylabel("Pixel count (×10³)", fontsize=11)
    axes[2].set_xlabel("GT depth bucket (m)", fontsize=11)
    for xi, v in enumerate(count_per_bin):
        if v > 0:
            axes[2].text(xi, v / 1e3 + 0.3, f"{v//1000}k", ha="center",
                         va="bottom", fontsize=6.5, color="black")

    # Shared x-axis labels
    step = max(1, n_bins // 20)          # avoid label crowding
    axes[2].set_xticks(x[::step])
    axes[2].set_xticklabels(bin_labels[::step], rotation=45, ha="right",
                             fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3, lw=0.6)
        ax.set_xlim(-0.5, n_bins - 0.5)

    fig.tight_layout()
    path = out_path or str(_out_dir() / "error_by_depth_bucket.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Bucket error figure saved → {path}")


def plot_depth_distributions(all_metrics: list, out_path: str = None) -> None:
    """
    Overlaid histogram of GT depth vs predicted depth across all frames,
    plus a per-frame KDE ribbon to show frame-to-frame variability.
    """
    pred_all = np.concatenate([m["_pred_flat"] for m in all_metrics])
    gt_all   = np.concatenate([m["_gt_flat"]   for m in all_metrics])

    # Clip to sensible display range
    depth_max = float(np.percentile(gt_all, 99))
    pred_all  = np.clip(pred_all, 0, depth_max)
    gt_clip   = np.clip(gt_all,   0, depth_max)

    bins = np.linspace(0, depth_max, 80)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # ---- Top: overlaid histograms (normalised to density) ----
    ax = axes[0]
    ax.hist(gt_clip,   bins=bins, density=True, alpha=0.55,
            color="steelblue", label="GT depth (RealSense)", edgecolor="none")
    ax.hist(pred_all,  bins=bins, density=True, alpha=0.55,
            color="tomato",    label="Predicted metric depth", edgecolor="none")

    # Mark means
    for val, col, lbl in [
        (float(np.mean(gt_clip)),  "steelblue", f"GT mean {np.mean(gt_clip):.2f} m"),
        (float(np.mean(pred_all)), "tomato",    f"Pred mean {np.mean(pred_all):.2f} m"),
    ]:
        ax.axvline(val, color=col, lw=1.8, ls="--", label=lbl)

    ax.set_xlabel("Depth (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Depth Distribution: GT vs Predicted  (all frames pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, lw=0.6)

    # ---- Bottom: per-frame histogram ribbon ----
    ax2 = axes[1]
    bin_c = (bins[:-1] + bins[1:]) / 2
    gt_curves, pred_curves = [], []
    for m in all_metrics:
        g = np.clip(m["_gt_flat"],   0, depth_max)
        p = np.clip(m["_pred_flat"], 0, depth_max)
        gt_curves.append(  np.histogram(g, bins=bins, density=True)[0])
        pred_curves.append(np.histogram(p, bins=bins, density=True)[0])

    gt_arr   = np.array(gt_curves)
    pred_arr = np.array(pred_curves)

    for arr, color, label in [
        (gt_arr,   "steelblue", "GT"),
        (pred_arr, "tomato",    "Predicted"),
    ]:
        med  = np.median(arr, axis=0)
        p10  = np.percentile(arr, 10, axis=0)
        p90  = np.percentile(arr, 90, axis=0)
        ax2.plot(bin_c, med, color=color, lw=1.8, label=f"{label} median")
        ax2.fill_between(bin_c, p10, p90, color=color, alpha=0.20,
                         label=f"{label} 10–90th pct")

    ax2.set_xlabel("Depth (m)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Per-frame Variability (median ± 10–90th percentile ribbon)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, lw=0.6)

    fig.tight_layout()
    path = out_path or str(_out_dir() / "depth_distributions.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Depth distribution figure saved → {path}")


def plot_signed_error_distribution(all_metrics: list, out_path: str = None) -> None:
    """
    Distribution of signed depth error (pred − GT) pooled across all frames.
    Shows bias (systematic over/under-prediction) and spread.
    Also breaks down signed error by 50 cm depth bucket.
    """
    pred_all = np.concatenate([m["_pred_flat"] for m in all_metrics])
    gt_all   = np.concatenate([m["_gt_flat"]   for m in all_metrics])
    error    = pred_all - gt_all          # signed: + = over-prediction (too far)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ---- Top: global signed error histogram ----
    ax = axes[0]
    clip_e = float(np.percentile(np.abs(error), 98))
    bins_e = np.linspace(-clip_e, clip_e, 100)

    ax.hist(error, bins=bins_e, density=True, color="mediumpurple",
            alpha=0.75, edgecolor="none", label="signed error")

    mean_e   = float(np.mean(error))
    median_e = float(np.median(error))
    std_e    = float(np.std(error))

    ax.axvline(0,        color="black",      lw=1.5, ls="-",  label="zero")
    ax.axvline(mean_e,   color="crimson",    lw=1.8, ls="--",
               label=f"mean  {mean_e:+.3f} m")
    ax.axvline(median_e, color="darkorange", lw=1.8, ls=":",
               label=f"median {median_e:+.3f} m")

    # ±1 std shading
    ax.axvspan(mean_e - std_e, mean_e + std_e, alpha=0.12, color="crimson",
               label=f"±1 std ({std_e:.3f} m)")

    ax.set_xlabel("Signed error: pred − GT  (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Signed Depth Error Distribution  (all frames pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, lw=0.6)

    # ---- Bottom: median signed error per 50 cm depth bucket ----
    ax2 = axes[1]
    depth_max = float(np.percentile(gt_all, 99))
    edges     = np.arange(0, depth_max + 0.5, 0.5)
    n_bins    = len(edges) - 1
    bin_c     = (edges[:-1] + edges[1:]) / 2

    med_err  = np.full(n_bins, np.nan)
    p25_err  = np.full(n_bins, np.nan)
    p75_err  = np.full(n_bins, np.nan)
    p10_err  = np.full(n_bins, np.nan)
    p90_err  = np.full(n_bins, np.nan)
    mean_err = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (gt_all >= edges[i]) & (gt_all < edges[i + 1])
        if mask.sum() < 5:
            continue
        e = error[mask]
        med_err[i]  = float(np.median(e))
        p25_err[i]  = float(np.percentile(e, 25))
        p75_err[i]  = float(np.percentile(e, 75))
        p10_err[i]  = float(np.percentile(e, 10))
        p90_err[i]  = float(np.percentile(e, 90))
        mean_err[i] = float(np.mean(e))

    valid = ~np.isnan(med_err)
    ax2.axhline(0, color="black", lw=1.2, ls="-")
    ax2.fill_between(bin_c[valid], p10_err[valid], p90_err[valid],
                     color="mediumpurple", alpha=0.18, label="10–90th pct")
    ax2.fill_between(bin_c[valid], p25_err[valid], p75_err[valid],
                     color="mediumpurple", alpha=0.35, label="25–75th pct (IQR)")
    ax2.plot(bin_c[valid], med_err[valid],  color="mediumpurple", lw=2.0,
             label="median signed error")
    ax2.plot(bin_c[valid], mean_err[valid], color="crimson", lw=1.5, ls="--",
             label="mean signed error")

    ax2.set_xlabel("GT depth bucket (m)", fontsize=11)
    ax2.set_ylabel("Signed error: pred − GT  (m)", fontsize=11)
    ax2.set_title("Signed Error by Depth Bucket  (50 cm bins)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, lw=0.6)
    # Colour positive/negative regions
    ax2.axhspan(0,  ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 5,
                alpha=0.04, color="tomato",    label="_over-predict")
    ax2.axhspan(ax2.get_ylim()[0] if ax2.get_ylim()[0] < 0 else -5, 0,
                alpha=0.04, color="steelblue", label="_under-predict")

    fig.tight_layout()
    path = out_path or str(_out_dir() / "signed_error_distribution.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Signed error distribution figure saved → {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline metric depth pipeline (DAv2 disparity + RealSense GT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frame",        type=int,   default=None,  metavar="N",
                        help="Process only frame_NNNNN. Default: all frames.")
    parser.add_argument("--no-viz",       action="store_true",
                        help="Skip saving visualisation panels and profile plots.")
    parser.add_argument("--max-gt-depth", type=float, default=None,  metavar="M",
                        help="Cap RealSense GT at M m before fitting/evaluation. "
                             "Output goes to output_<M>m/ (e.g. output_3m/).")
    parser.add_argument("--anchor-row",   type=int,   default=None,  metavar="R",
                        help="Fix scale-fit anchor to row R. "
                             "Omit to auto-select the row with the most GT depth spread.")
    parser.add_argument("--data-dir",     type=str,   default=None,  metavar="DIR",
                        help="Root dir containing rgb/, depth/, rel_depth/. "
                             "Defaults to the current working directory.")
    args = parser.parse_args()

    global MAX_GT_DEPTH, ANCHOR_ROW, RGB_DIR, DEPTH_DIR, REL_DIR
    MAX_GT_DEPTH = args.max_gt_depth
    ANCHOR_ROW   = args.anchor_row          # None → auto-select per frame

    if args.data_dir is not None:
        root = Path(args.data_dir)
        RGB_DIR   = root / "rgb"
        DEPTH_DIR = root / "depth"
        REL_DIR   = root / "rel_depth"

    print(f"[config] data_dir     : {RGB_DIR.parent}")
    print(f"[config] anchor_row   : {ANCHOR_ROW if ANCHOR_ROW is not None else 'auto'}")
    print(f"[config] max_gt_depth : {MAX_GT_DEPTH if MAX_GT_DEPTH is not None else 'none'}")
    print(f"[config] output_dir   : {_out_dir()}")

    stems = ([f"frame_{args.frame:05d}"] if args.frame is not None
             else sorted(p.stem for p in DEPTH_DIR.glob("frame_*.npy")))

    all_metrics = []
    for stem in stems:
        if not (REL_DIR / f"{stem}.npy").exists():
            print(f"[skip] {stem}: no relative depth")
            continue
        print(f"\n--- {stem} ---")
        m = run_frame(stem, save_viz=not args.no_viz)
        all_metrics.append(m)
        print(f"  anchor_row={m['anchor_row']}  s={m['scale_s']:.4f}  t={m['shift_t']:.4f}  inliers={m['n_inliers']}  "
              f"[in-range] AbsRel={m['AbsRel']:.4f}  MAE={m['MAE']:.4f}m  RMSE={m['RMSE']:.4f}m  "
              f"[full]  AbsRel={m['AbsRel_full']:.4f}  MAE={m['MAE_full']:.4f}m")

    if len(all_metrics) > 1:
        print("\n===== Summary =====")
        for key, label in [
            ("AbsRel", "AbsRel (in-range)"),
            ("MAE",    "MAE m  (in-range)"),
            ("RMSE",   "RMSE m (in-range)"),
            ("AbsRel_full", "AbsRel (full frame)"),
            ("MAE_full",    "MAE m  (full frame)"),
            ("scale_s", "scale s"),
            ("shift_t", "shift t"),
        ]:
            vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
            if vals:
                print(f"  {label:25s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                      f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    if not args.no_viz:
        print(f"\nVisualizations saved to {_out_dir()}/")

    if len(all_metrics) > 1 and not args.no_viz:
        _out_dir().mkdir(exist_ok=True)
        plot_bucket_errors(all_metrics, bucket_size=0.5)
        plot_depth_distributions(all_metrics)
        plot_signed_error_distribution(all_metrics)


if __name__ == "__main__":
    main()

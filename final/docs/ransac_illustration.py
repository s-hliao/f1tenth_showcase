"""
ransac_illustration.py

Generates a 3-panel figure comparing:
  1. Naive least-squares scale fit vs RANSAC on anchor-row samples (with injected outliers)
  2. Resulting metric depth maps side-by-side (no-scale / LS / RANSAC / GT)
  3. Absolute error map and per-method error bar chart

Uses real anchor-row data from testing_data/frame_00000.npy with synthetic outliers
injected to simulate real-world LiDAR noise (specular reflections, glass, dynamic objects).

Output: docs/ransac_vs_no_ransac.png
"""

import sys
import os
sys.path.insert(0, os.path.expanduser("~/Depth-Anything-V2"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / "testing_data"
OUTDIR  = BASE / "docs"
OUTDIR.mkdir(exist_ok=True)

FRAME       = "frame_00000"
ANCHOR_ROW  = 400
MAX_DEPTH   = 10.0
MAX_GT_DEPTH = 3.0         # RealSense unreliable beyond this; cap GT here
OUTLIER_FRAC = 0.25        # 25 % of anchor samples are "bad"
RNG = np.random.default_rng(7)

# ── Load real data ──────────────────────────────────────────────────────────
d_disp_full = np.load(DATA / "rel_depth" / f"{FRAME}.npy").astype(np.float32)
d_gt_full   = np.load(DATA / "depth"     / f"{FRAME}.npy").astype(np.float32)
rgb_bgr     = cv2.imread(str(DATA / "rgb" / f"{FRAME}.png"))
if rgb_bgr is None:
    rgb_bgr = np.zeros((480, 848, 3), dtype=np.uint8)
rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

anchor_disp = d_disp_full[ANCHOR_ROW, :]
anchor_gt   = d_gt_full  [ANCHOR_ROW, :]
# Cap GT at MAX_GT_DEPTH — RealSense is unreliable beyond 3 m
anchor_gt   = anchor_gt.copy()
anchor_gt[anchor_gt > MAX_GT_DEPTH] = 0.0
valid       = (anchor_disp > 0) & (anchor_gt > 0)

ad = anchor_disp[valid]
ag = anchor_gt[valid]
n  = len(ad)

# ── Inject realistic outliers ───────────────────────────────────────────────
# Simulate bad LiDAR returns: glass reflections (far), specular surfaces (near)
outlier_idx = RNG.choice(n, int(n * OUTLIER_FRAC), replace=False)
ag_noisy = ag.copy()
half = len(outlier_idx) // 2
ag_noisy[outlier_idx[:half]] = RNG.uniform(ag.max() * 1.5, MAX_DEPTH, half)   # far outliers
ag_noisy[outlier_idx[half:]] = RNG.uniform(0.1, ag.min() * 0.5,
                                            len(outlier_idx) - half)          # near outliers
is_outlier = np.zeros(n, dtype=bool)
is_outlier[outlier_idx] = True

# ── Fit helpers ─────────────────────────────────────────────────────────────
def fit_ls(disp, gt):
    A = np.stack([disp, np.ones(len(disp))], axis=1)
    st, _, _, _ = np.linalg.lstsq(A, gt, rcond=None)
    return float(st[0]), float(st[1])

def fit_ransac(disp, gt, n_iters=2000, inlier_tol=0.30):
    s_ls, t_ls = fit_ls(disp, gt)
    pred = np.clip(s_ls * disp + t_ls, 0, MAX_DEPTH)
    rel  = np.abs(pred - gt) / (gt + 1e-6)
    mask = rel < inlier_tol
    if mask.sum() >= 2:
        s, t = fit_ls(disp[mask], gt[mask])
    else:
        s, t = s_ls, t_ls
    # refine with RANSAC random sampling
    rng2 = np.random.default_rng(42)
    best_s, best_t, best_n = s, t, int(mask.sum())
    for _ in range(n_iters):
        i, j = rng2.choice(len(disp), 2, replace=False)
        A2 = np.array([[disp[i], 1], [disp[j], 1]], dtype=float)
        if abs(np.linalg.det(A2)) < 1e-8:
            continue
        st2 = np.linalg.solve(A2, [gt[i], gt[j]])
        sc, tc = float(st2[0]), float(st2[1])
        pred2 = np.clip(sc * disp + tc, 0, MAX_DEPTH)
        m2 = (np.abs(pred2 - gt) / (gt + 1e-6)) < inlier_tol
        if m2.sum() > best_n:
            best_n = int(m2.sum())
            s_r, t_r = fit_ls(disp[m2], gt[m2])
            pred_r = np.clip(s_r * disp + t_r, 0, MAX_DEPTH)
            mr = (np.abs(pred_r - gt) / (gt + 1e-6)) < inlier_tol
            best_s, best_t = s_r, t_r
            best_n = int(mr.sum())
    inlier_mask = (np.abs(np.clip(best_s * disp + best_t, 0, MAX_DEPTH) - gt) /
                   (gt + 1e-6)) < inlier_tol
    return best_s, best_t, inlier_mask

# Fit on noisy (with outliers) data
s_ls,    t_ls    = fit_ls(ad, ag_noisy)
s_ran,   t_ran, inlier_mask = fit_ransac(ad, ag_noisy)
# "ground truth" fit on clean data (no outliers)
s_clean, t_clean = fit_ls(ad, ag)

print(f"LS    fit: s={s_ls:.4f}  t={t_ls:.4f}")
print(f"RANSAC:    s={s_ran:.4f}  t={t_ran:.4f}")
print(f"Clean ref: s={s_clean:.4f}  t={t_clean:.4f}")
print(f"RANSAC inliers: {inlier_mask.sum()}/{n} ({100*inlier_mask.mean():.1f}%)")

# ── Build depth maps ────────────────────────────────────────────────────────
def apply(s, t, disp):
    return np.clip(s * disp + t, 0, MAX_DEPTH).astype(np.float32)

d_ls     = apply(s_ls,    t_ls,    d_disp_full)
d_ransac = apply(s_ran,   t_ran,   d_disp_full)
d_clean  = apply(s_clean, t_clean, d_disp_full)   # best possible (clean LS)

# ── Metrics helper ──────────────────────────────────────────────────────────
def metrics(pred, gt, mask=None):
    v = (gt > 0) & (gt <= MAX_GT_DEPTH) & (pred > 0)
    if mask is not None:
        v &= mask
    if v.sum() == 0:
        return dict(AbsRel=np.nan, MAE=np.nan, RMSE=np.nan)
    p, g = pred[v], gt[v]
    return dict(
        AbsRel=float(np.mean(np.abs(p - g) / g)),
        MAE=float(np.mean(np.abs(p - g))),
        RMSE=float(np.sqrt(np.mean((p - g) ** 2))),
    )

m_ls     = metrics(d_ls,     d_gt_full)
m_ransac = metrics(d_ransac, d_gt_full)
m_clean  = metrics(d_clean,  d_gt_full)

# ── Colorize helper ─────────────────────────────────────────────────────────
def colorize(depth, vmax):
    d = np.clip(depth / (vmax + 1e-6), 0, 1)
    c = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[depth <= 0] = 0
    return c

vmax = float(np.percentile(d_gt_full[d_gt_full > 0], 95))

c_gt     = colorize(d_gt_full, vmax)
c_ls     = colorize(d_ls,      vmax)
c_ransac = colorize(d_ransac,  vmax)
c_clean  = colorize(d_clean,   vmax)

# Absolute error maps (clipped for display)
err_ls     = np.abs(d_ls     - d_gt_full); err_ls    [d_gt_full <= 0] = 0
err_ransac = np.abs(d_ransac - d_gt_full); err_ransac[d_gt_full <= 0] = 0
e_vmax = float(np.percentile(err_ls[d_gt_full > 0], 95))

def colorize_err(e, ev):
    d = np.clip(e / (ev + 1e-6), 0, 1)
    c = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[e <= 0] = 0
    return c

ce_ls     = colorize_err(err_ls,     e_vmax)
ce_ransac = colorize_err(err_ransac, e_vmax)

# ═══════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
fig.suptitle(
    f"Why RANSAC Matters for Monocular Depth Scale Recovery  "
    f"(GT evaluated ≤ {MAX_GT_DEPTH} m)",
    fontsize=17, fontweight="bold", color="white", y=0.98,
)

gs_top = gridspec.GridSpec(1, 2, figure=fig, top=0.91, bottom=0.56,
                           left=0.04, right=0.96, wspace=0.06)
gs_mid = gridspec.GridSpec(2, 4, figure=fig, top=0.52, bottom=0.22,
                           left=0.04, right=0.96, wspace=0.03, hspace=0.03)
gs_bot = gridspec.GridSpec(1, 3, figure=fig, top=0.18, bottom=0.04,
                           left=0.10, right=0.90, wspace=0.35)

text_kw  = dict(color="white")
title_kw = dict(color="#aaddff", fontsize=11, fontweight="bold", pad=5)

# ── Panel A: Scatter + fit lines ────────────────────────────────────────────
ax_scatter = fig.add_subplot(gs_top[0])
ax_scatter.set_facecolor("#161b22")

inlier_v = ~is_outlier
ax_scatter.scatter(ad[inlier_v], ag_noisy[inlier_v],
                   c="#4fc3f7", s=8, alpha=0.6, label="Inlier anchor samples", zorder=3)
ax_scatter.scatter(ad[is_outlier], ag_noisy[is_outlier],
                   c="#ff5252", s=18, alpha=0.85, marker="x", linewidths=1.5,
                   label=f"Outliers ({int(OUTLIER_FRAC*100)}% injected)", zorder=4)

# Fit lines over disparity range
d_range = np.linspace(ad.min(), ad.max(), 200)
ax_scatter.plot(d_range, s_ls  * d_range + t_ls,
                color="#ff8a65", lw=2.2, ls="--",
                label=f"Naïve LS   s={s_ls:.3f} t={t_ls:.3f}", zorder=5)
ax_scatter.plot(d_range, s_ran * d_range + t_ran,
                color="#69f0ae", lw=2.5,
                label=f"RANSAC     s={s_ran:.3f} t={t_ran:.3f}", zorder=6)
ax_scatter.plot(d_range, s_clean * d_range + t_clean,
                color="#e0e0e0", lw=1.5, ls=":",
                label=f"Clean ref  s={s_clean:.3f} t={t_clean:.3f}", zorder=7)

ax_scatter.set_xlabel("DAv2 disparity (a.u.)", **text_kw, fontsize=10)
ax_scatter.set_ylabel("GT metric depth (m)", **text_kw, fontsize=10)
ax_scatter.set_title("A  —  Scale Estimation: Anchor Row Samples", **title_kw)
ax_scatter.tick_params(colors="white")
ax_scatter.spines[:].set_color("#444")
legend = ax_scatter.legend(fontsize=9, facecolor="#1e2530", labelcolor="white",
                           edgecolor="#555", framealpha=0.85)
ax_scatter.set_xlim(ad.min() - 0.1, ad.max() + 0.1)
ax_scatter.set_ylim(-0.2, MAX_DEPTH * 0.65)

# annotation arrow showing how LS is dragged by outliers
mid_d = float(np.percentile(ad, 75))
ax_scatter.annotate(
    "Outliers drag\nLS fit upward",
    xy=(mid_d, s_ls * mid_d + t_ls),
    xytext=(mid_d + 0.3, s_ls * mid_d + t_ls + 1.2),
    arrowprops=dict(arrowstyle="->", color="#ff8a65", lw=1.5),
    color="#ff8a65", fontsize=8.5,
)
ax_scatter.annotate(
    "RANSAC ignores\noutliers → correct scale",
    xy=(mid_d - 0.6, s_ran * (mid_d - 0.6) + t_ran),
    xytext=(mid_d - 1.8, s_ran * (mid_d - 0.6) + t_ran + 1.0),
    arrowprops=dict(arrowstyle="->", color="#69f0ae", lw=1.5),
    color="#69f0ae", fontsize=8.5,
)

# ── Panel B: Anchor row depth profile ──────────────────────────────────────
ax_profile = fig.add_subplot(gs_top[1])
ax_profile.set_facecolor("#161b22")

cols = np.where(valid)[0]
ax_profile.plot(cols, ag,                color="#e0e0e0", lw=1.8,
                label="GT (RealSense)", zorder=5)
ax_profile.plot(cols, np.clip(s_ls  * ad + t_ls,  0, MAX_DEPTH),
                color="#ff8a65", lw=1.5, ls="--",
                label=f"Naïve LS  MAE={m_ls['MAE']:.3f}m", zorder=4)
ax_profile.plot(cols, np.clip(s_ran * ad + t_ran, 0, MAX_DEPTH),
                color="#69f0ae", lw=1.8,
                label=f"RANSAC    MAE={m_ransac['MAE']:.3f}m", zorder=6)

# Mark inlier/outlier positions
out_cols = cols[is_outlier]
ax_profile.scatter(out_cols, ag_noisy[is_outlier],
                   c="#ff5252", s=20, marker="x", lw=1.5,
                   label="Injected outliers", zorder=7)

ax_profile.set_xlabel("Image column (px)", **text_kw, fontsize=10)
ax_profile.set_ylabel("Depth (m)", **text_kw, fontsize=10)
ax_profile.set_title(f"B  —  Anchor Row {ANCHOR_ROW} Depth Profile", **title_kw)
ax_profile.tick_params(colors="white")
ax_profile.spines[:].set_color("#444")
ax_profile.set_ylim(bottom=0)
ax_profile.legend(fontsize=9, facecolor="#1e2530", labelcolor="white",
                  edgecolor="#555", framealpha=0.85)

# ── Panel C (row 1): Depth maps ─────────────────────────────────────────────
depth_imgs   = [rgb_rgb, c_gt, c_ls, c_ransac]
depth_labels = ["RGB input", "GT Depth\n(RealSense)",
                f"Naïve LS depth\nAbsRel={m_ls['AbsRel']:.3f}",
                f"RANSAC depth\nAbsRel={m_ransac['AbsRel']:.3f}"]
border_colors = ["#888", "#e0e0e0", "#ff8a65", "#69f0ae"]

for col_i, (img, lbl, bc) in enumerate(zip(depth_imgs, depth_labels, border_colors)):
    ax = fig.add_subplot(gs_mid[0, col_i])
    ax.imshow(img)
    ax.set_title(lbl, color=bc, fontsize=9, fontweight="bold", pad=3)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(bc)
        spine.set_linewidth(2)
    # anchor row marker
    ax.axhline(ANCHOR_ROW, color=bc, lw=0.8, alpha=0.6)

# ── Panel C (row 2): Error maps ─────────────────────────────────────────────
# Dummy (blank) for RGB and GT columns
blank = np.zeros_like(c_gt)

err_imgs   = [blank, blank, ce_ls, ce_ransac]
err_labels = ["", "",
              f"LS abs error\nMAE={m_ls['MAE']:.3f}m  RMSE={m_ls['RMSE']:.3f}m",
              f"RANSAC abs error\nMAE={m_ransac['MAE']:.3f}m  RMSE={m_ransac['RMSE']:.3f}m"]
err_borders = ["#0d1117", "#0d1117", "#ff8a65", "#69f0ae"]

for col_i, (img, lbl, bc) in enumerate(zip(err_imgs, err_labels, err_borders)):
    ax = fig.add_subplot(gs_mid[1, col_i])
    if lbl:
        ax.imshow(img)
        ax.set_title(lbl, color=bc, fontsize=9, fontweight="bold", pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2)
    else:
        ax.set_facecolor("#0d1117")
    ax.axis("off")

# ── Panel D: Error bar chart ────────────────────────────────────────────────
methods = ["Naïve LS\n(with outliers)", "RANSAC\n(with outliers)", "Clean LS\n(no outliers,\nreference)"]
colors  = ["#ff8a65", "#69f0ae", "#90a4ae"]

for bar_i, (metric_key, metric_label, ax_idx) in enumerate([
    ("AbsRel", "AbsRel  (↓ better)", 0),
    ("MAE",    "MAE (m)  (↓ better)", 1),
    ("RMSE",   "RMSE (m) (↓ better)", 2),
]):
    vals = [m_ls[metric_key], m_ransac[metric_key], m_clean[metric_key]]
    ax = fig.add_subplot(gs_bot[ax_idx])
    ax.set_facecolor("#161b22")
    bars = ax.bar(methods, vals, color=colors, width=0.5, edgecolor="#333")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=10,
                fontweight="bold")
    ax.set_title(f"D{bar_i+1}  —  {metric_label}", **title_kw)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.set_ylim(0, max(vals) * 1.25)
    # highlight improvement
    improvement = (vals[0] - vals[1]) / vals[0] * 100
    if bar_i == 0:
        ax.text(0.5, 0.92, f"RANSAC improves by {improvement:.1f}%",
                transform=ax.transAxes, ha="center", color="#ffcc02",
                fontsize=9, fontweight="bold")

# ── Colorbar legend for depth maps ─────────────────────────────────────────
cbar_ax = fig.add_axes([0.965, 0.22, 0.012, 0.30])
grad = np.linspace(1, 0, 256).reshape(256, 1)
cbar_ax.imshow(grad, aspect="auto", cmap="inferno")
cbar_ax.set_xticks([])
cbar_ax.set_yticks([0, 255])
cbar_ax.set_yticklabels([f"{vmax:.1f}m", "0m"], color="white", fontsize=8)
cbar_ax.yaxis.set_label_position("right")
cbar_ax.yaxis.tick_right()

out_path = OUTDIR / f"ransac_vs_no_ransac_{MAX_GT_DEPTH:g}m.png"
fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved → {out_path}")

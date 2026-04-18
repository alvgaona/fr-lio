"""Post-LC map correction via source-pose-tagged points.

Demonstrates the foundation idea from `loop_closure_paper.md`:
- Each map point carries an index of the keyframe that inserted it
- After pose graph optimization corrects pose k by Δ_k, transform every
  point with source_idx = k by Δ_k
- The map ends up consistent with the corrected trajectory

Pipeline:
  1. Run IESKF with per-point Σ + sliding cube (gets drifted poses + tagged map)
  2. Detect LCs (by GT proximity)
  3. Pose graph optimize (fixed-weight LC for simplicity; we already showed
     edge weighting doesn't change the conclusion qualitatively)
  4. Apply omap.correct_map(original_poses, corrected_poses)
  5. Evaluate map quality before vs after correction:
     - Mean and median distance from each map point to nearest GT wall
     - Visual: scatter the corrected vs original map

Run:
    SIM_ENV=cube pixi run python -u sim_imu_trajectory.py
    SIM_ENV=cube SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 SIM_PERPOINT_COV=1 \\
      pixi run python -u sim_map_correction.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import PLANES
from sim_iekf_3d import rodrigues_exp
from sim_square_corridor_no_lc import (
    run_ieskf_no_lc, CUBE_LEN, BIAS_WALK_POS, BIAS_WALK_ROT,
)
from sim_square_corridor_fixed_lc import (
    detect_loop_closures, measure_loop_closure, build_odom_edges,
    pose_graph_optimize, compute_ate,
    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING,
    LC_NOISE_POS, LC_NOISE_ROT,
    FIXED_ODOM_POS_STD, FIXED_ODOM_ROT_STD,
    FIXED_LC_POS_STD, FIXED_LC_ROT_STD,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
ENV = os.environ.get("SIM_ENV", "default")


def map_to_gt_distance(points):
    """For each point, distance to the closest GT wall plane (perpendicular).
    PLANES are (point_on_plane, normal, bounds_uvuv). We use the perpendicular
    distance, ignoring bounds for the distance metric (closest plane wins).
    """
    if not points:
        return np.zeros(0)
    pts = np.asarray(points)
    n_planes = len(PLANES)
    dists = np.full((len(pts), n_planes), np.inf)
    for j, (p_on_plane, normal, _bounds) in enumerate(PLANES):
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        d_offset = float(n @ np.asarray(p_on_plane, dtype=float))
        signed = pts @ n - d_offset
        dists[:, j] = np.abs(signed)
    return dists.min(axis=1)


print(f"Loading sim data for env={ENV}...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

print(f"\nRunning IESKF (cube_len={CUBE_LEN:.1f} m, per-point Σ on)...")
np.random.seed(123)
(est_poses, est_times, gt_poses, _, _, _, _,
 _, _, _, edge_covs, omap) = run_ieskf_no_lc(
    imu_data, lidar_data, use_perpoint_cov=True)
n = len(est_poses)
print(f"  {n} keyframes; map has {len(omap)} points")
print(f"  source_idx range: [{min(omap.source_idx)}, {max(omap.source_idx)}]")

ate_no_lc = compute_ate(est_poses, gt_poses)
print(f"  No-LC ATE RMSE: {np.sqrt(np.mean(ate_no_lc**2)):.4f} m")

# Snapshot original map points BEFORE correction so we can compare
original_map_points = [p.copy() for p in omap.points]
original_dists = map_to_gt_distance(original_map_points)
print(f"\nMap-to-GT-walls distance BEFORE correction:")
print(f"  mean={original_dists.mean()*1000:.1f} mm  "
      f"median={np.median(original_dists)*1000:.1f} mm  "
      f"p95={np.percentile(original_dists, 95)*1000:.1f} mm")

print(f"\nDetecting loop closures...")
closures = detect_loop_closures(gt_poses, est_times,
                                LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
print(f"  {len(closures)} LCs found")

rng = np.random.default_rng(99)
lc_factors = []
for (i, j, _) in closures:
    R_rel, t_rel = measure_loop_closure(gt_poses, i, j,
                                        LC_NOISE_POS, LC_NOISE_ROT, rng)
    lc_cov = np.diag([
        FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2,
        FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2,
    ])
    lc_factors.append((i, j, R_rel, t_rel, lc_cov))

fixed_odom_cov = np.diag([
    FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2,
    FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2,
])
odom_edges = build_odom_edges(est_poses, fixed_odom_cov)

print("\nPose graph optimization...")
corrected_poses = pose_graph_optimize(est_poses, odom_edges, lc_factors)
ate_lc = compute_ate(corrected_poses, gt_poses)
print(f"  Corrected ATE RMSE: {np.sqrt(np.mean(ate_lc**2)):.4f} m")

print("\nApplying map correction (per-source-pose Δ transform)...")
n_corrected = omap.correct_map(est_poses, corrected_poses)
print(f"  {n_corrected} / {len(omap)} points corrected")

corrected_dists = map_to_gt_distance(omap.points)
print(f"\nMap-to-GT-walls distance AFTER correction:")
print(f"  mean={corrected_dists.mean()*1000:.1f} mm  "
      f"median={np.median(corrected_dists)*1000:.1f} mm  "
      f"p95={np.percentile(corrected_dists, 95)*1000:.1f} mm")

improvement = ((original_dists.mean() - corrected_dists.mean()) /
               original_dists.mean() * 100)
print(f"\nMap quality improvement (mean dist): {improvement:+.1f}%")

print("\nSaving plot...")
gt_p = np.array([p for _, p in gt_poses])
est_p = np.array([p for _, p in est_poses])
opt_p = np.array([p for _, p in corrected_poses])
orig_pts = np.asarray(original_map_points)
corr_pts = np.asarray(omap.points)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.scatter(orig_pts[:, 0], orig_pts[:, 1], s=0.3, alpha=0.4, c="r",
           label="map (drifted, pre-LC)")
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="GT trajectory")
ax.plot(est_p[:, 0], est_p[:, 1], "r-", lw=1.0, alpha=0.8, label="IESKF estimate")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Map BEFORE LC correction")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(corr_pts[:, 0], corr_pts[:, 1], s=0.3, alpha=0.4, c="b",
           label="map (corrected)")
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="GT trajectory")
ax.plot(opt_p[:, 0], opt_p[:, 1], "b-", lw=1.0, alpha=0.8, label="LC-optimized")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Map AFTER LC correction")
ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
bins = np.linspace(0, max(np.percentile(original_dists, 99),
                          np.percentile(corrected_dists, 99)), 60)
ax.hist(original_dists * 1000, bins=bins * 1000, alpha=0.55, color="r",
        edgecolor="black", label=f"before  mean={original_dists.mean()*1000:.0f}mm")
ax.hist(corrected_dists * 1000, bins=bins * 1000, alpha=0.55, color="b",
        edgecolor="black", label=f"after  mean={corrected_dists.mean()*1000:.0f}mm")
ax.set_xlabel("Distance to nearest GT wall (mm)")
ax.set_ylabel("# map points")
ax.set_title("Map-to-GT-walls distance distribution")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Overlay: source-pose-colored map (pre-correction) — visualizes how points
# from different keyframes ended up at different drifted positions.
ax = axes[1, 0]
src = np.asarray(omap.source_idx)
sc = ax.scatter(orig_pts[:, 0], orig_pts[:, 1], s=0.5,
                c=src, cmap="viridis", alpha=0.6)
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=0.7, alpha=0.5)
plt.colorbar(sc, ax=ax, label="source keyframe index")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Map points colored by source keyframe (pre-correction)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
sc = ax.scatter(corr_pts[:, 0], corr_pts[:, 1], s=0.5,
                c=src, cmap="viridis", alpha=0.6)
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=0.7, alpha=0.5)
plt.colorbar(sc, ax=ax, label="source keyframe index")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Map points colored by source keyframe (post-correction)")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

ax = axes[1, 2]
t_arr = np.array(est_times)
ax.plot(t_arr, ate_no_lc, "r-", lw=1.0, label=f"no LC  RMSE={np.sqrt(np.mean(ate_no_lc**2)):.3f}m")
ax.plot(t_arr, ate_lc, "b-", lw=1.0, label=f"LC opt  RMSE={np.sqrt(np.mean(ate_lc**2)):.3f}m")
for (_, j, _) in closures:
    ax.axvline(est_times[j], color="g", ls=":", lw=0.5, alpha=0.4)
ax.set_xlabel("t (s)"); ax.set_ylabel("position error (m)")
ax.set_title("Trajectory ATE pre/post LC")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/sim_map_correction_{ENV}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

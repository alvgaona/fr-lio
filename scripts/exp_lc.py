"""Experiment 4: loop closure + self-correcting maps.

Bucket #4 of the thesis experiment plan. Builds on the bucket #3 filter
(per-point Σ + S2S CRLB drift accumulator, plus the sliding cube + global
map architecture used in `sim_square_corridor_no_lc.run_ieskf_no_lc`).

Pipeline:
  1. Run IESKF with use_perpoint_cov=True and enable_global_map=True. The
     filter produces drifted keyframe poses, a working map (sliding-cube
     pruned, simulating ikd-Tree behaviour), and a shadow / global map that
     absorbs evicted points and remains correctable.
  2. Detect loop closures using ground-truth proximity (Python sim
     equivalent of the C++ GICP-based detector).
  3. Pose graph optimize with fixed-weight LC edges.
  4. Apply `correct_map(orig_poses, corrected_poses)` to BOTH the working
     map and the global map. Each map point uses its `source_idx` to look
     up the per-keyframe Δ.
  5. Save CSVs and plots comparing trajectory ATE, map-to-GT-wall distance,
     and pose graph quality before/after correction.

Tagging: outputs live under sim_data/exp_lc_{env}_*.{png,csv}.

Env knobs:
  SIM_ENV          : cube | corridor | room_corridor | corridor_grid | square_corridor
  SIM_CUBE_LEN     : working-map sliding-cube half-extent (m), default 4.0
  SIM_LC_DIST      : LC distance threshold in GT space (m), default 1.0
  SIM_LC_MIN_GAP   : min temporal gap between LC i and j (s), default 10.0
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import ENVIRONMENT, PLANES
from sim_square_corridor_no_lc import run_ieskf_no_lc, CUBE_LEN
from sim_square_corridor_fixed_lc import (
    detect_loop_closures, measure_loop_closure, build_odom_edges,
    pose_graph_optimize, compute_ate,
    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING,
    LC_NOISE_POS, LC_NOISE_ROT,
    FIXED_ODOM_POS_STD, FIXED_ODOM_ROT_STD,
    FIXED_LC_POS_STD, FIXED_LC_ROT_STD,
)

LC_WEIGHTS = os.environ.get("LC_WEIGHTS", "fixed").lower()
assert LC_WEIGHTS in {"fixed", "crlb"}, "LC_WEIGHTS must be fixed|crlb"

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)
TAG = f"exp_lc_{LC_WEIGHTS}_{ENVIRONMENT}"
CSV_DIR = os.path.join(OUT_DIR, f"{TAG}_csv")
os.makedirs(CSV_DIR, exist_ok=True)

# Variance floors when LC_WEIGHTS=crlb so the edge covariance does not
# collapse to zero (which would over-trust the LC measurement). Floors are
# the same magnitudes used by the fixed-weight config so a single LC step
# between very-confident keyframes still has a sensible noise level.
CRLB_LC_POS_FLOOR_VAR = FIXED_LC_POS_STD ** 2
CRLB_LC_ROT_FLOOR_VAR = FIXED_LC_ROT_STD ** 2


def map_to_gt_wall_distance(points):
    """Perpendicular distance from each point to the nearest GT plane.
    Ignores plane bounds (point-to-infinite-plane), so it's a measure of
    how well-aligned the map's geometry is with the room walls / blocks.
    """
    if len(points) == 0:
        return np.zeros(0)
    pts = np.asarray(points, dtype=float)
    dists = np.full((len(pts), len(PLANES)), np.inf)
    for j, (p_on_plane, normal, _) in enumerate(PLANES):
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        offset = float(n @ np.asarray(p_on_plane, dtype=float))
        dists[:, j] = np.abs(pts @ n - offset)
    return dists.min(axis=1)


def write_trajectory_csv(path, est_times, est_poses, corrected_poses, gt_poses):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t",
                    "est_x", "est_y", "est_z",
                    "corr_x", "corr_y", "corr_z",
                    "gt_x", "gt_y", "gt_z"])
        for i, t in enumerate(est_times):
            ep = est_poses[i][1]
            cp = corrected_poses[i][1]
            gp = gt_poses[i][1]
            w.writerow([t, ep[0], ep[1], ep[2], cp[0], cp[1], cp[2],
                        gp[0], gp[1], gp[2]])


def write_ate_csv(path, est_times, ate_no_lc, ate_lc):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ate_no_lc", "ate_lc"])
        for i, t in enumerate(est_times):
            w.writerow([t, ate_no_lc[i], ate_lc[i]])


def write_lc_edges_csv(path, closures, est_times):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kf_i", "kf_j", "t_i", "t_j", "gt_dist"])
        for (i, j, d) in closures:
            w.writerow([i, j, est_times[i], est_times[j], d])


def write_map_xy_csv(path, points_before, points_after, source_idx,
                     max_points=4000):
    """Save downsampled (x, y, source_idx) of map points before/after correction.
    Writes a single CSV per snapshot with columns: x, y, src.
    """
    import numpy as np
    pts_b = np.asarray(points_before)
    pts_a = np.asarray(points_after)
    src = np.asarray(source_idx)
    n = min(len(pts_b), len(pts_a), len(src))
    if n == 0:
        return
    step = max(1, n // max_points)
    idx = list(range(0, n, step))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_before", "y_before", "x_after", "y_after", "src"])
        for i in idx:
            w.writerow([pts_b[i, 0], pts_b[i, 1],
                        pts_a[i, 0], pts_a[i, 1], src[i]])


def write_mapdist_csv(path, before, after, label):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"dist_before_{label}", f"dist_after_{label}"])
        n = max(len(before), len(after))
        for i in range(n):
            row = [
                before[i] if i < len(before) else "",
                after[i] if i < len(after) else "",
            ]
            w.writerow(row)


def main():
    print(f"[exp_lc] ENV={ENVIRONMENT}  CUBE_LEN={CUBE_LEN:.2f}  "
          f"LC_DIST={LC_DIST_THRESH:.2f} m  LC_MIN_GAP={LC_MIN_TIME_GAP:.1f} s")
    print(f"Loading sim data from {OUT_DIR}...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

    print("\nRunning IESKF (per-point Σ, S2S CRLB, sliding cube + global map)...")
    np.random.seed(123)
    (est_poses, est_times, gt_poses, _, _, _, _,
     _, _, _, edge_covs, omap, gmap) = run_ieskf_no_lc(
        imu_data, lidar_data, use_perpoint_cov=True, enable_global_map=True)

    # Cumulative P_drift snapshot at each keyframe. edge_covs is a list of
    # 6x6 per-edge covariances in (pos, rot) block order — same convention as
    # the C++ `LCKeyframe::p_drift_snapshot`. None entries (no s2s update on
    # that step) contribute zero.
    p_drift_cumulative = []
    cum = np.zeros((6, 6))
    for ec in edge_covs:
        if ec is not None:
            cum = cum + ec
        p_drift_cumulative.append(cum.copy())
    n = len(est_poses)
    print(f"  {n} keyframes")
    print(f"  Working map: {len(omap)} points  Global map: {len(gmap)} points")

    ate_no_lc = compute_ate(est_poses, gt_poses)
    rmse_no_lc = float(np.sqrt(np.mean(ate_no_lc ** 2)))
    print(f"  No-LC ATE RMSE: {rmse_no_lc:.4f} m  "
          f"max={ate_no_lc.max():.4f} m  final={ate_no_lc[-1]:.4f} m")

    original_working_points = [p.copy() for p in omap.points]
    original_global_points = [p.copy() for p in gmap.points]
    before_working_dists = map_to_gt_wall_distance(original_working_points)
    before_global_dists = map_to_gt_wall_distance(original_global_points)
    print(f"  Map→GT distance BEFORE:")
    print(f"    working: mean={before_working_dists.mean() * 1000:.1f} mm  "
          f"median={np.median(before_working_dists) * 1000:.1f} mm  "
          f"p95={np.percentile(before_working_dists, 95) * 1000:.1f} mm")
    print(f"    global : mean={before_global_dists.mean() * 1000:.1f} mm  "
          f"median={np.median(before_global_dists) * 1000:.1f} mm  "
          f"p95={np.percentile(before_global_dists, 95) * 1000:.1f} mm")

    print(f"\nDetecting loop closures (GT proximity, dist≤{LC_DIST_THRESH} m, "
          f"gap≥{LC_MIN_TIME_GAP} s)...")
    closures = detect_loop_closures(gt_poses, est_times,
                                    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
    print(f"  {len(closures)} LCs detected")
    for (i, j, d) in closures[:8]:
        print(f"    kf {i:>4} (t={est_times[i]:6.2f}s) ↔ "
              f"kf {j:>4} (t={est_times[j]:6.2f}s)  GT-dist={d:.3f} m")
    if len(closures) > 8:
        print(f"    … and {len(closures) - 8} more")

    if len(closures) == 0:
        print("\nNo LCs detected — skipping pose graph + correction. "
              "Consider a longer trajectory or revisiting environment.")
        return

    rng = np.random.default_rng(99)
    lc_factors = []
    fixed_lc_cov = np.diag([
        FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2,
        FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2,
    ])
    floor_pgo = np.diag([CRLB_LC_ROT_FLOOR_VAR] * 3 + [CRLB_LC_POS_FLOOR_VAR] * 3)
    lc_cov_traces = []
    for (i, j, _) in closures:
        R_rel, t_rel = measure_loop_closure(gt_poses, i, j,
                                            LC_NOISE_POS, LC_NOISE_ROT, rng)
        if LC_WEIGHTS == "crlb":
            i_, j_ = sorted((i, j))
            # P_drift_cumulative entries are in (pos, rot) block order;
            # PGO factor cov is in (rot, pos) block order — swap.
            dP = p_drift_cumulative[j_] - p_drift_cumulative[i_]
            dP_pgo = np.zeros((6, 6))
            dP_pgo[0:3, 0:3] = dP[3:6, 3:6]   # rot block
            dP_pgo[3:6, 3:6] = dP[0:3, 0:3]   # pos block
            dP_pgo[0:3, 3:6] = dP[3:6, 0:3]   # cross blocks (PGO order)
            dP_pgo[3:6, 0:3] = dP[0:3, 3:6]
            # Symmetric + PSD safety + floor.
            dP_pgo = 0.5 * (dP_pgo + dP_pgo.T)
            eigvals, eigvecs = np.linalg.eigh(dP_pgo)
            eigvals = np.maximum(eigvals, 0.0)
            dP_pgo = eigvecs @ np.diag(eigvals) @ eigvecs.T
            lc_cov = dP_pgo + floor_pgo
        else:
            lc_cov = fixed_lc_cov
        lc_cov_traces.append(float(np.trace(lc_cov)))
        lc_factors.append((i, j, R_rel, t_rel, lc_cov))
    if LC_WEIGHTS == "crlb" and lc_cov_traces:
        print(f"  LC edge cov trace: min={min(lc_cov_traces):.4e}  "
              f"median={np.median(lc_cov_traces):.4e}  "
              f"max={max(lc_cov_traces):.4e}  (vs fixed={np.trace(fixed_lc_cov):.4e})")
    fixed_odom_cov = np.diag([
        FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2,
        FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2,
    ])
    odom_edges = build_odom_edges(est_poses, fixed_odom_cov)

    print("\nPose graph optimization (fixed-weight LC)...")
    corrected_poses = pose_graph_optimize(est_poses, odom_edges, lc_factors)
    ate_lc = compute_ate(corrected_poses, gt_poses)
    rmse_lc = float(np.sqrt(np.mean(ate_lc ** 2)))
    print(f"  Corrected ATE RMSE: {rmse_lc:.4f} m  "
          f"max={ate_lc.max():.4f} m  final={ate_lc[-1]:.4f} m")

    print("\nApplying per-source-pose Δ map correction to BOTH maps...")
    n_corr_w = omap.correct_map(est_poses, corrected_poses)
    n_corr_g = gmap.correct_map(est_poses, corrected_poses)
    print(f"  Working map: {n_corr_w} / {len(omap)} points corrected")
    print(f"  Global  map: {n_corr_g} / {len(gmap)} points corrected")

    after_working_dists = map_to_gt_wall_distance(omap.points)
    after_global_dists = map_to_gt_wall_distance(gmap.points)
    print(f"  Map→GT distance AFTER:")
    print(f"    working: mean={after_working_dists.mean() * 1000:.1f} mm  "
          f"median={np.median(after_working_dists) * 1000:.1f} mm  "
          f"p95={np.percentile(after_working_dists, 95) * 1000:.1f} mm")
    print(f"    global : mean={after_global_dists.mean() * 1000:.1f} mm  "
          f"median={np.median(after_global_dists) * 1000:.1f} mm  "
          f"p95={np.percentile(after_global_dists, 95) * 1000:.1f} mm")

    imp_traj = (1.0 - rmse_lc / max(rmse_no_lc, 1e-9)) * 100
    imp_w = (1.0 - after_working_dists.mean() / max(before_working_dists.mean(), 1e-9)) * 100
    imp_g = (1.0 - after_global_dists.mean() / max(before_global_dists.mean(), 1e-9)) * 100
    print(f"\nImprovement summary:")
    print(f"  Trajectory ATE RMSE  : {imp_traj:+.1f}%  ({rmse_no_lc:.4f} → {rmse_lc:.4f} m)")
    print(f"  Working map mean dist: {imp_w:+.1f}%")
    print(f"  Global  map mean dist: {imp_g:+.1f}%")

    write_trajectory_csv(os.path.join(CSV_DIR, "trajectory.csv"),
                         est_times, est_poses, corrected_poses, gt_poses)
    write_ate_csv(os.path.join(CSV_DIR, "ate.csv"), est_times, ate_no_lc, ate_lc)
    write_lc_edges_csv(os.path.join(CSV_DIR, "lc_edges.csv"), closures, est_times)
    write_mapdist_csv(os.path.join(CSV_DIR, "mapdist_working.csv"),
                      before_working_dists, after_working_dists, "working")
    write_mapdist_csv(os.path.join(CSV_DIR, "mapdist_global.csv"),
                      before_global_dists, after_global_dists, "global")
    write_map_xy_csv(os.path.join(CSV_DIR, "map_global_xy.csv"),
                     original_global_points, gmap.points, gmap.source_idx)
    write_map_xy_csv(os.path.join(CSV_DIR, "map_working_xy.csv"),
                     original_working_points, omap.points, omap.source_idx)
    print(f"\nSaved CSVs to {CSV_DIR}")

    gt_p = np.array([p for _, p in gt_poses])
    est_p = np.array([p for _, p in est_poses])
    opt_p = np.array([p for _, p in corrected_poses])
    orig_global = np.asarray(original_global_points) if original_global_points else np.zeros((0, 3))
    corr_global = np.asarray(gmap.points) if gmap.points else np.zeros((0, 3))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    if len(orig_global):
        ax.scatter(orig_global[:, 0], orig_global[:, 1], s=0.3, alpha=0.4, c="r",
                   label=f"global map drifted (N={len(orig_global)})")
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.4, label="GT")
    ax.plot(est_p[:, 0], est_p[:, 1], "r-", lw=1.0, alpha=0.85, label="IESKF estimate")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"BEFORE LC correction — {ENVIRONMENT}")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(corr_global):
        ax.scatter(corr_global[:, 0], corr_global[:, 1], s=0.3, alpha=0.4, c="b",
                   label=f"global map corrected (N={len(corr_global)})")
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.4, label="GT")
    ax.plot(opt_p[:, 0], opt_p[:, 1], "b-", lw=1.0, alpha=0.85, label="LC-optimized")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"AFTER LC correction — {ENVIRONMENT}")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    upper = max(np.percentile(before_global_dists, 99) if len(before_global_dists) else 0.1,
                np.percentile(after_global_dists, 99) if len(after_global_dists) else 0.1)
    bins = np.linspace(0, upper, 60) * 1000
    if len(before_global_dists):
        ax.hist(before_global_dists * 1000, bins=bins, alpha=0.55, color="r",
                edgecolor="black",
                label=f"before mean={before_global_dists.mean() * 1000:.0f}mm")
    if len(after_global_dists):
        ax.hist(after_global_dists * 1000, bins=bins, alpha=0.55, color="b",
                edgecolor="black",
                label=f"after  mean={after_global_dists.mean() * 1000:.0f}mm")
    ax.set_xlabel("Distance to nearest GT wall (mm)"); ax.set_ylabel("# map points")
    ax.set_title("Global map: distance to GT walls")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(orig_global):
        src = np.asarray(gmap.source_idx)
        sc = ax.scatter(orig_global[:, 0], orig_global[:, 1], s=0.5,
                        c=src, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, ax=ax, label="source keyframe")
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=0.7, alpha=0.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Global map colored by source keyframe (pre)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(corr_global):
        sc = ax.scatter(corr_global[:, 0], corr_global[:, 1], s=0.5,
                        c=src, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, ax=ax, label="source keyframe")
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=0.7, alpha=0.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Global map colored by source keyframe (post)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    t_arr = np.array(est_times)
    ax.plot(t_arr, ate_no_lc, "r-", lw=1.0,
            label=f"no LC  RMSE={rmse_no_lc:.3f} m")
    ax.plot(t_arr, ate_lc, "b-", lw=1.0,
            label=f"LC opt RMSE={rmse_lc:.3f} m")
    for (_, j, _) in closures:
        ax.axvline(est_times[j], color="g", ls=":", lw=0.5, alpha=0.4)
    ax.set_xlabel("t (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("Trajectory ATE — before vs after LC")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{TAG}_overview.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

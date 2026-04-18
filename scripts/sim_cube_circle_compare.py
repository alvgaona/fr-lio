"""Cube-room + circle-trajectory: compare LC configurations.

Same 7 configs as sim_square_corridor_compare.py, but on the default cube room
(10x10x10 m with obstacles) and a circle trajectory. Geometry is roughly
isotropic from the interior — useful for testing whether CRLB magnitude
calibration alone closes the gap to fixed weights.

Run:
    SIM_ENV=cube SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_cube_circle_compare.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import trajectory_at, DURATION
from sim_environment_3d import OBSTACLES, ROOM_X, ROOM_Y
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
from sim_square_corridor_compare import build_crlb_edges

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)


def plot_walls_cube(ax):
    ax.plot([0, ROOM_X, ROOM_X, 0, 0], [0, 0, ROOM_Y, ROOM_Y, 0],
            color="#444444", lw=1.3)
    for obs in OBSTACLES:
        cx, cy, _ = obs["center"]
        sx, sy, _ = obs["size"]
        ax.plot([cx-sx/2, cx+sx/2, cx+sx/2, cx-sx/2, cx-sx/2],
                [cy-sy/2, cy-sy/2, cy+sy/2, cy+sy/2, cy-sy/2],
                color="#cc6600", lw=0.8, alpha=0.8)


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans, "
      f"duration={DURATION:.1f} s")

print(f"\nRunning IESKF once (cube_len={CUBE_LEN:.1f} m)...")
np.random.seed(123)
(est_poses, est_times, gt_poses, _, _, _, _,
 _, _, _, edge_covs_per_step) = run_ieskf_no_lc(imu_data, lidar_data,
                                                  use_perpoint_cov=False)
n = len(est_poses)
n_valid = sum(1 for c in edge_covs_per_step if c is not None)
print(f"  {n} keyframe poses, {n_valid} valid CRLB edges")

print(f"\nRunning IESKF with per-point covariance (for Config I)...")
np.random.seed(123)
(est_poses_pp, est_times_pp, gt_poses_pp, _, _, _, _,
 _, _, _, edge_covs_pp) = run_ieskf_no_lc(imu_data, lidar_data,
                                            use_perpoint_cov=True)
print(f"  {len(est_poses_pp)} keyframe poses (per-point)")

ate_no_lc = compute_ate(est_poses, gt_poses)
rmse_no_lc = float(np.sqrt(np.mean(ate_no_lc ** 2)))
print(f"  No-LC ATE: RMSE={rmse_no_lc:.4f} m  mean={ate_no_lc.mean():.4f} m  "
      f"max={ate_no_lc.max():.4f} m  final={ate_no_lc[-1]:.4f} m")

print(f"\nDetecting loop closures...")
closures = detect_loop_closures(gt_poses, est_times,
                                LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
print(f"  Found {len(closures)} loop closures")
for (i, j, d) in closures[:8]:
    print(f"    pose {i:>4} (t={est_times[i]:6.2f}s) ↔ pose {j:>4} "
          f"(t={est_times[j]:6.2f}s)  GT-dist={d:.3f} m")
if len(closures) > 8:
    print(f"    ... and {len(closures)-8} more")

rng = np.random.default_rng(99)
lc_factors_template = [(i, j, *measure_loop_closure(gt_poses, i, j,
                                                     LC_NOISE_POS, LC_NOISE_ROT, rng))
                       for (i, j, _) in closures]

lc_cov_fixed = np.diag([
    FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2,
    FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2,
])
fixed_odom_cov = np.diag([
    FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2,
    FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2,
])
fixed_pos_trace = float(np.trace(fixed_odom_cov[3:6, 3:6]))
fixed_rot_trace = float(np.trace(fixed_odom_cov[0:3, 0:3]))

valid_traces = [np.trace(c) for c in edge_covs_per_step if c is not None]
fallback_cov = (np.eye(6) * (np.mean(valid_traces) / 6.0)
                if valid_traces else fixed_odom_cov)

odom_edges_fixed = build_odom_edges(est_poses, fixed_odom_cov)
common_kw = dict(strip_bias_walk=True,
                 bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
                 dt_per_step=0.1)
odom_edges_crlb = build_crlb_edges(edge_covs_per_step, fallback_cov, **common_kw)
odom_edges_crlb_bw = build_crlb_edges(edge_covs_per_step, fallback_cov,
                                       strip_bias_walk=False)
odom_edges_crlb_shape = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    rescale_target={"pos": fixed_pos_trace, "rot": fixed_rot_trace}, **common_kw)
odom_edges_crlb_bw_x10 = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    extra_bias_walk_pos=BIAS_WALK_POS * 10,
    extra_bias_walk_rot=BIAS_WALK_ROT * 10, **common_kw)
odom_edges_crlb_bw_x100 = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    extra_bias_walk_pos=BIAS_WALK_POS * 100,
    extra_bias_walk_rot=BIAS_WALK_ROT * 100, **common_kw)
odom_edges_crlb_imu = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    add_imu_floor=True, **common_kw)
odom_edges_crlb_imu_pos = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    add_imu_floor_pos_only=True, **common_kw)


def edge_summary(label, edges):
    pos = [np.trace(e[2][3:6, 3:6]) for e in edges]
    rot = [np.trace(e[2][0:3, 0:3]) for e in edges]
    print(f"  {label:<30}  pos median={np.median(pos):.3e} m^2  "
          f"rot median={np.median(rot):.3e} rad^2  "
          f"pos max/min={np.max(pos)/max(np.min(pos), 1e-15):.1f}")


print("\nPer-edge covariance stats:")
edge_summary("Fixed", odom_edges_fixed)
edge_summary("CRLB (no bias-walk)", odom_edges_crlb)
edge_summary("CRLB + bias-walk", odom_edges_crlb_bw)
edge_summary("CRLB shape, fixed scale (E)", odom_edges_crlb_shape)
edge_summary("CRLB + 10x bias-walk (F1)", odom_edges_crlb_bw_x10)
edge_summary("CRLB + 100x bias-walk (F2)", odom_edges_crlb_bw_x100)
edge_summary("CRLB + IMU process floor (G)", odom_edges_crlb_imu)
edge_summary("CRLB + IMU floor pos only (H)", odom_edges_crlb_imu_pos)

lc_factors_fixed = [(i, j, Rr, tr, lc_cov_fixed)
                    for (i, j, Rr, tr) in lc_factors_template]


def run_and_eval(label, odom):
    print(f"\n[{label}]  optimizing...")
    poses = pose_graph_optimize(est_poses, odom, lc_factors_fixed)
    ate = compute_ate(poses, gt_poses)
    rmse = float(np.sqrt(np.mean(ate ** 2)))
    print(f"  ATE: RMSE={rmse:.4f} m  mean={ate.mean():.4f} m  "
          f"max={ate.max():.4f} m  final={ate[-1]:.4f} m")
    return poses, ate


configs = [("A: No LC", est_poses, ate_no_lc)]
for label, odom in [
    ("B: Fixed-weight LC", odom_edges_fixed),
    ("C: CRLB LC", odom_edges_crlb),
    ("D: CRLB+bias-walk LC", odom_edges_crlb_bw),
    ("E: CRLB shape (fixed scale)", odom_edges_crlb_shape),
    ("F1: CRLB + 10x bias-walk", odom_edges_crlb_bw_x10),
    ("F2: CRLB + 100x bias-walk", odom_edges_crlb_bw_x100),
    ("G: CRLB + IMU floor", odom_edges_crlb_imu),
    ("H: CRLB + IMU floor (pos)", odom_edges_crlb_imu_pos),
]:
    poses, ate = run_and_eval(label, odom)
    configs.append((label, poses, ate))

# Config I and I+G: per-point IESKF baseline
odom_edges_I = build_odom_edges(est_poses_pp, fixed_odom_cov)
odom_edges_I_G = build_crlb_edges(
    edge_covs_pp, fallback_cov,
    add_imu_floor=True, **common_kw)

print(f"\n[I: per-point IESKF + fixed LC]  optimizing...")
poses_I = pose_graph_optimize(est_poses_pp, odom_edges_I, lc_factors_fixed)
ate_I = compute_ate(poses_I, gt_poses_pp)
rmse_I = float(np.sqrt(np.mean(ate_I ** 2)))
print(f"  ATE: RMSE={rmse_I:.4f} m  mean={ate_I.mean():.4f} m  "
      f"max={ate_I.max():.4f} m  final={ate_I[-1]:.4f} m")
configs.append(("I: per-point IESKF + fixed LC", poses_I, ate_I))

print(f"\n[I+G: per-point IESKF + IMU-floor LC]  optimizing...")
poses_IG = pose_graph_optimize(est_poses_pp, odom_edges_I_G, lc_factors_fixed)
ate_IG = compute_ate(poses_IG, gt_poses_pp)
rmse_IG = float(np.sqrt(np.mean(ate_IG ** 2)))
print(f"  ATE: RMSE={rmse_IG:.4f} m  mean={ate_IG.mean():.4f} m  "
      f"max={ate_IG.max():.4f} m  final={ate_IG[-1]:.4f} m")
configs.append(("I+G: per-point IESKF + IMU LC", poses_IG, ate_IG))

print("\n" + "=" * 70)
print("SUMMARY (lower RMSE = better trajectory quality)")
print("=" * 70)
print(f"  {'Config':<32} {'RMSE':>10} {'mean':>10} {'max':>10} {'final':>11}")
for label, _, ate in configs:
    r = float(np.sqrt(np.mean(ate ** 2)))
    print(f"  {label:<32} {r:>10.4f} {ate.mean():>10.4f} {ate.max():>10.4f} {ate[-1]:>11.4f}")

base = float(np.sqrt(np.mean(ate_no_lc ** 2)))
print(f"\nRMSE-ATE improvement vs no-LC:")
for label, _, ate in configs[1:]:
    r = float(np.sqrt(np.mean(ate ** 2)))
    pct = (1 - r / base) * 100
    print(f"  {label:<32} {pct:+.1f}%")

t_arr = np.array(est_times)
gt_p = np.array([p for _, p in gt_poses])

fig = plt.figure(figsize=(20, 14))
colors = ["r", "m", "b", "g", "c", "#aa6600", "#660066", "#0099aa",
          "#cc0099", "#ff8800", "#008844"]

ax = plt.subplot(3, 3, 1)
plot_walls_cube(ax)
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="GT")
for (label, poses, _), c in zip(configs, colors):
    p = np.array([pp for _, pp in poses])
    ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.0, alpha=0.85, label=label)
ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=8)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"Trajectories (cube room, {len(closures)} LC)")
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 3, 2)
for (label, _, ate), c in zip(configs, colors):
    ax.plot(t_arr, ate, "-", color=c, lw=1.1, alpha=0.85, label=label)
for (_, j, _) in closures:
    ax.axvline(est_times[j], color="g", ls=":", lw=0.5, alpha=0.4)
ax.set_xlabel("t (s)"); ax.set_ylabel("position error (m)")
ax.set_title("ATE vs time")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 3, 3)
labels = [c[0].split(":")[0] for c in configs]
mean_vals = [c[2].mean() for c in configs]
max_vals = [c[2].max() for c in configs]
final_vals = [c[2][-1] for c in configs]
x = np.arange(len(labels)); w = 0.25
ax.bar(x - w, mean_vals, w, label="mean", color="#cc4444")
ax.bar(x, max_vals, w, label="max", color="#4444cc")
ax.bar(x + w, final_vals, w, label="final", color="#44aa44")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("position error (m)"); ax.set_title("ATE summary")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

n_cfg = len(configs)
n_rows = 4
n_cols = 4
fig.set_size_inches(20, 16)
for idx, (label, poses, _) in enumerate(configs):
    if 5 + idx > n_rows * n_cols:
        break
    ax = plt.subplot(n_rows, n_cols, 5 + idx)
    plot_walls_cube(ax)
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.2, label="GT")
    p = np.array([pp for _, pp in poses])
    ax.plot(p[:, 0], p[:, 1], "-", color=colors[idx], lw=1.0, alpha=0.9)
    ax.set_aspect("equal"); ax.set_title(label, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/sim_cube_circle_compare.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")

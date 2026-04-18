"""Room-corridor compare: tests CRLB vs fixed in HETEROGENEOUS geometry.

Same 7 configs as sim_square_corridor_compare.py, but on room_corridor where
the trajectory oscillates between an information-rich room and a
geometrically-degenerate corridor. Hypothesis: CRLB anisotropy *should*
help here because edge covariances differ between segments.

Run:
    SIM_ENV=room_corridor SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_room_corridor_compare.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import trajectory_at, DURATION
from sim_environment_3d import OBSTACLES
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
from sim_square_corridor_compare import (
    rescale_to_trace, build_crlb_edges,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)


def plot_walls_room_corridor(ax):
    """Top-down outline of room_corridor environment."""
    ax.plot([0, 10, 10, 0, 0], [0, 0, 10, 10, 0], color="#444444", lw=1.3)
    ax.plot([10, 30], [3.5, 3.5], color="#444444", lw=1.3)
    ax.plot([10, 30], [6.5, 6.5], color="#444444", lw=1.3)
    ax.plot([30, 30], [3.5, 6.5], color="#444444", lw=1.3)
    ax.plot([10, 10], [0, 3.5], color="#444444", lw=1.3)
    ax.plot([10, 10], [6.5, 10], color="#444444", lw=1.3)
    for obs in OBSTACLES:
        cx, cy, _ = obs["center"]
        sx, sy, _ = obs["size"]
        ax.plot([cx-sx/2, cx+sx/2, cx+sx/2, cx-sx/2, cx-sx/2],
                [cy-sy/2, cy-sy/2, cy+sy/2, cy+sy/2, cy-sy/2],
                color="#cc6600", lw=0.8, alpha=0.7)


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans, "
      f"duration={DURATION:.1f} s")
print(f"  bias-walk: pos={BIAS_WALK_POS} m^2/s, rot={BIAS_WALK_ROT} rad^2/s")

print(f"\nRunning IESKF once (cube_len={CUBE_LEN:.1f} m)...")
np.random.seed(123)
(est_poses, est_times, gt_poses, _, _, _, _,
 _, _, _, edge_covs_per_step, _, _, _) = run_ieskf_no_lc(imu_data, lidar_data)
n = len(est_poses)
n_valid = sum(1 for c in edge_covs_per_step if c is not None)
print(f"  {n} keyframe poses, {n_valid} valid CRLB edges")

ate_no_lc = compute_ate(est_poses, gt_poses)
print(f"  No-LC ATE: mean={ate_no_lc.mean():.4f} m, max={ate_no_lc.max():.4f} m, "
      f"final={ate_no_lc[-1]:.4f} m")

print(f"\nDetecting loop closures...")
closures = detect_loop_closures(gt_poses, est_times,
                                LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
print(f"  Found {len(closures)} loop closures")
for (i, j, d) in closures[:10]:
    print(f"    pose {i:>4} (t={est_times[i]:6.2f}s) ↔ pose {j:>4} "
          f"(t={est_times[j]:6.2f}s)  GT-dist={d:.3f} m")
if len(closures) > 10:
    print(f"    ... and {len(closures)-10} more")

rng = np.random.default_rng(99)
lc_factors_template = []
for (i, j, _) in closures:
    R_rel, t_rel = measure_loop_closure(gt_poses, i, j,
                                        LC_NOISE_POS, LC_NOISE_ROT, rng)
    lc_factors_template.append((i, j, R_rel, t_rel))

lc_cov_fixed = np.diag([
    FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2,
    FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2,
])
fixed_odom_cov = np.diag([
    FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2,
    FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2,
])

valid_traces = [np.trace(c) for c in edge_covs_per_step if c is not None]
fallback_cov = (np.eye(6) * (np.mean(valid_traces) / 6.0)
                if valid_traces else fixed_odom_cov)

odom_edges_fixed = build_odom_edges(est_poses, fixed_odom_cov)
odom_edges_crlb = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    strip_bias_walk=True,
    bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
    dt_per_step=1.0 / 10.0)
odom_edges_crlb_bw = build_crlb_edges(
    edge_covs_per_step, fallback_cov, strip_bias_walk=False)
fixed_pos_trace = float(np.trace(fixed_odom_cov[3:6, 3:6]))
fixed_rot_trace = float(np.trace(fixed_odom_cov[0:3, 0:3]))
odom_edges_crlb_shape = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    strip_bias_walk=True,
    bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
    dt_per_step=1.0 / 10.0,
    rescale_target={"pos": fixed_pos_trace, "rot": fixed_rot_trace})
odom_edges_crlb_bw_x10 = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    strip_bias_walk=True,
    bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
    dt_per_step=1.0 / 10.0,
    extra_bias_walk_pos=BIAS_WALK_POS * 10,
    extra_bias_walk_rot=BIAS_WALK_ROT * 10)
odom_edges_crlb_bw_x100 = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    strip_bias_walk=True,
    bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
    dt_per_step=1.0 / 10.0,
    extra_bias_walk_pos=BIAS_WALK_POS * 100,
    extra_bias_walk_rot=BIAS_WALK_ROT * 100)


def edge_summary(label, edges):
    pos_traces = [np.trace(e[2][3:6, 3:6]) for e in edges]
    rot_traces = [np.trace(e[2][0:3, 0:3]) for e in edges]
    print(f"  {label:<30}  pos median trace={np.median(pos_traces):.3e} "
          f"m^2  rot median trace={np.median(rot_traces):.3e} rad^2  "
          f"pos max/min ratio={np.max(pos_traces)/max(np.min(pos_traces), 1e-15):.1f}")


print("\nPer-edge covariance stats (max/min ratio shows env-adaptivity):")
edge_summary("Fixed", odom_edges_fixed)
edge_summary("CRLB (no bias-walk)", odom_edges_crlb)
edge_summary("CRLB + bias-walk", odom_edges_crlb_bw)
edge_summary("CRLB shape, fixed scale (E)", odom_edges_crlb_shape)
edge_summary("CRLB + 10x bias-walk (F1)", odom_edges_crlb_bw_x10)
edge_summary("CRLB + 100x bias-walk (F2)", odom_edges_crlb_bw_x100)

lc_factors_fixed = [(i, j, Rr, tr, lc_cov_fixed)
                    for (i, j, Rr, tr) in lc_factors_template]


def run_and_eval(label, odom, lc_factors):
    print(f"\n[{label}]  optimizing...")
    poses = pose_graph_optimize(est_poses, odom, lc_factors)
    ate = compute_ate(poses, gt_poses)
    print(f"  ATE: mean={ate.mean():.4f} m  max={ate.max():.4f} m  "
          f"final={ate[-1]:.4f} m")
    return poses, ate


configs = [("A: No LC", est_poses, ate_no_lc)]
for (label, odom) in [
    ("B: Fixed-weight LC", odom_edges_fixed),
    ("C: CRLB LC", odom_edges_crlb),
    ("D: CRLB+bias-walk LC", odom_edges_crlb_bw),
    ("E: CRLB shape (fixed scale)", odom_edges_crlb_shape),
    ("F1: CRLB + 10x bias-walk", odom_edges_crlb_bw_x10),
    ("F2: CRLB + 100x bias-walk", odom_edges_crlb_bw_x100),
]:
    poses, ate = run_and_eval(label, odom, lc_factors_fixed)
    configs.append((label, poses, ate))

print("\n" + "=" * 70)
print("SUMMARY (lower mean ATE = better trajectory quality)")
print("=" * 70)
print(f"  {'Config':<32} {'mean ATE':>10} {'max ATE':>10} {'final ATE':>11}")
for label, _, ate in configs:
    print(f"  {label:<32} {ate.mean():>10.4f} {ate.max():>10.4f} {ate[-1]:>11.4f}")

base = ate_no_lc.mean()
print(f"\nMean-ATE improvement vs no-LC:")
for label, _, ate in configs[1:]:
    pct = (1 - ate.mean() / base) * 100
    print(f"  {label:<32} {pct:+.1f}%")

t_arr = np.array(est_times)
gt_p = np.array([p for _, p in gt_poses])

fig = plt.figure(figsize=(20, 14))
colors = ["r", "m", "b", "g", "c", "#aa6600", "#660066"]

ax = plt.subplot(3, 3, 1)
plot_walls_room_corridor(ax)
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="GT")
for (label, poses, _), c in zip(configs, colors):
    p = np.array([pp for _, pp in poses])
    ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.0, alpha=0.85, label=label)
ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=8)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Trajectories (room+corridor, {len(closures)} LC)")
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper right")
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 3, 2)
for (label, _, ate), c in zip(configs, colors):
    ax.plot(t_arr, ate, "-", color=c, lw=1.1, alpha=0.85, label=label)
for (_, j, _) in closures:
    ax.axvline(est_times[j], color="g", ls=":", lw=0.5, alpha=0.4)
ax.set_xlabel("t (s)")
ax.set_ylabel("position error (m)")
ax.set_title("ATE vs time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = plt.subplot(3, 3, 3)
labels = [c[0].split(":")[0] for c in configs]
mean_vals = [c[2].mean() for c in configs]
max_vals = [c[2].max() for c in configs]
final_vals = [c[2][-1] for c in configs]
x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, mean_vals, w, label="mean", color="#cc4444")
ax.bar(x, max_vals, w, label="max", color="#4444cc")
ax.bar(x + w, final_vals, w, label="final", color="#44aa44")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("position error (m)")
ax.set_title("ATE summary")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

for idx, (label, poses, _) in enumerate(configs):
    ax = plt.subplot(3, 4, 5 + idx)
    plot_walls_room_corridor(ax)
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.2, label="GT")
    p = np.array([pp for _, pp in poses])
    ax.plot(p[:, 0], p[:, 1], "-", color=colors[idx], lw=1.0, alpha=0.9, label=label)
    ax.set_aspect("equal")
    ax.set_title(label, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/sim_room_corridor_compare.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")

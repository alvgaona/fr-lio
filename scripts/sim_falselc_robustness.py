"""False-LC robustness sweep: tests how each edge-weighting strategy degrades
when a fraction of loop closures are wrong (place-recognition false positives).

Compares three configs at increasing false-LC rates:
  B   - Fixed-weight LC (LIO-SAM style)
  F2  - CRLB + 100x bias-walk (loose CRLB)
  G   - CRLB + IMU process noise floor (principled CRLB)

Hypothesis: fixed-weight is catastrophically corrupted by false LCs because it
weights LC ~30x more than odometry. CRLB-weighted variants (whose chain noise
is comparable to LC noise) are more graceful.

Run on the env whose data is currently in sim_data/:
    SIM_ENV=square_corridor pixi run python -u sim_imu_trajectory.py
    SIM_ENV=square_corridor SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_falselc_robustness.py

Output: sim_data/sim_falselc_robustness_<env>.png + table to stdout.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ENV = os.environ.get("SIM_ENV", "default")

FALSE_FRACS = [0.0, 0.05, 0.10, 0.20]
N_TRIALS_PER_FRAC = int(os.environ.get("SIM_N_TRIALS", "3"))


print(f"Loading sim data for env={ENV}...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

print(f"\nRunning IESKF (cube_len={CUBE_LEN:.1f}, bias_walk={BIAS_WALK_POS})...")
np.random.seed(123)
(est_poses, est_times, gt_poses, _, _, _, _,
 _, _, _, edge_covs_per_step, _) = run_ieskf_no_lc(imu_data, lidar_data)
n = len(est_poses)
print(f"  {n} keyframes")

ate_no_lc = compute_ate(est_poses, gt_poses)
rmse_no_lc = float(np.sqrt(np.mean(ate_no_lc ** 2)))
print(f"  Baseline no-LC ATE: RMSE={rmse_no_lc:.4f} m  "
      f"mean={ate_no_lc.mean():.4f} m  max={ate_no_lc.max():.4f} m")

print(f"\nDetecting LC candidates...")
closures = detect_loop_closures(gt_poses, est_times,
                                LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
print(f"  {len(closures)} candidate loop closures")

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
common_kw = dict(strip_bias_walk=True,
                 bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
                 dt_per_step=0.1)

odom_edges_fixed = build_odom_edges(est_poses, fixed_odom_cov)
odom_edges_F2 = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    extra_bias_walk_pos=BIAS_WALK_POS * 100,
    extra_bias_walk_rot=BIAS_WALK_ROT * 100, **common_kw)
odom_edges_G = build_crlb_edges(
    edge_covs_per_step, fallback_cov,
    add_imu_floor=True, **common_kw)

CONFIGS = [
    ("B: Fixed-weight", odom_edges_fixed),
    ("F2: CRLB + 100x bias-walk", odom_edges_F2),
    ("G: CRLB + IMU floor", odom_edges_G),
]

results = {label: {f: [] for f in FALSE_FRACS} for label, _ in CONFIGS}

for false_p in FALSE_FRACS:
    print(f"\n=== False-LC fraction: {false_p*100:.0f}% ===")
    for trial in range(N_TRIALS_PER_FRAC):
        rng = np.random.default_rng(99 + trial * 7)
        lc_factors = []
        n_false = 0
        for (i, j, _) in closures:
            if false_p > 0 and rng.random() < false_p:
                # mark this LC as false: use a random wrong target
                n_attempts = 0
                while True:
                    wrong_j = int(rng.integers(0, n))
                    if abs(wrong_j - j) >= 20 and abs(wrong_j - i) >= 20:
                        break
                    n_attempts += 1
                    if n_attempts > 30:
                        wrong_j = j
                        break
                Rj_use, pj_use = gt_poses[wrong_j]
                n_false += 1
            else:
                Rj_use, pj_use = gt_poses[j]
            Ri, pi = gt_poses[i]
            R_rel = Ri.T @ Rj_use
            t_rel = Ri.T @ (pj_use - pi)
            t_rel = t_rel + rng.normal(0, LC_NOISE_POS, 3)
            R_rel = R_rel @ rodrigues_exp(rng.normal(0, LC_NOISE_ROT, 3))
            lc_factors.append((i, j, R_rel, t_rel, lc_cov_fixed))

        for label, odom_edges in CONFIGS:
            poses = pose_graph_optimize(est_poses, odom_edges, lc_factors)
            ate = compute_ate(poses, gt_poses)
            rmse = float(np.sqrt(np.mean(ate ** 2)))
            results[label][false_p].append(rmse)
            print(f"  trial {trial+1}/{N_TRIALS_PER_FRAC}  "
                  f"{label:<28}  RMSE={rmse:.4f} m  "
                  f"(false LCs: {n_false}/{len(closures)})")

print("\n" + "=" * 80)
print(f"SUMMARY: RMSE ATE (over {N_TRIALS_PER_FRAC} trials) by false-LC fraction")
print("=" * 80)
header = f"  {'Config':<30} " + " ".join(f"{int(f*100):>4}%" for f in FALSE_FRACS)
print(header)
for label, _ in CONFIGS:
    vals = [np.mean(results[label][f]) for f in FALSE_FRACS]
    row = f"  {label:<30} " + " ".join(f"{v:>5.3f}" for v in vals)
    print(row)

print(f"\nWith no-LC baseline RMSE = {rmse_no_lc:.4f} m for reference.")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {"B: Fixed-weight": "r",
          "F2: CRLB + 100x bias-walk": "#aa6600",
          "G: CRLB + IMU floor": "#0099aa"}

ax = axes[0]
for label, _ in CONFIGS:
    means = [np.mean(results[label][f]) for f in FALSE_FRACS]
    stds = [np.std(results[label][f]) for f in FALSE_FRACS]
    ax.errorbar([f*100 for f in FALSE_FRACS], means, yerr=stds,
                marker="o", lw=1.5, capsize=4, label=label,
                color=colors.get(label, "k"))
ax.axhline(rmse_no_lc, color="gray", ls="--", lw=1, label="no-LC baseline")
ax.set_xlabel("False-LC fraction (%)")
ax.set_ylabel(f"RMSE ATE over {N_TRIALS_PER_FRAC} trials (m)")
ax.set_title(f"[{ENV}] LC robustness to false data association")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for label, _ in CONFIGS:
    means = [np.mean(results[label][f]) for f in FALSE_FRACS]
    base = means[0]
    rel = [m / base if base > 1e-9 else 1.0 for m in means]
    ax.plot([f*100 for f in FALSE_FRACS], rel, marker="o", lw=1.5,
            label=label, color=colors.get(label, "k"))
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_xlabel("False-LC fraction (%)")
ax.set_ylabel("ATE relative to clean (0% false)")
ax.set_title("Degradation factor (1.0 = no degradation)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/sim_falselc_robustness_{ENV}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")

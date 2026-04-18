"""Per-point covariance diagnosis: does GICP-style point covariance improve
no-LC IESKF accuracy vs the scalar LIDAR_POINT_VAR?

Runs the IESKF twice on the same data:
- Baseline: scalar LIDAR_POINT_VAR = 0.001 (current default)
- Per-point: R_i = sigma_range^2 + n_i^T Σ_p_i n_i (GICP-style)

Reports RMSE ATE for both, plus the distribution of effective per-match
sigma so we know whether the per-point variance varies meaningfully across
edges (anisotropy story for the paper).

Run:
    SIM_ENV=cube pixi run python -u sim_imu_trajectory.py
    SIM_ENV=cube SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      pixi run python -u sim_perpoint_diagnosis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_iekf_3d import (
    LIDAR_POINT_VAR, SIGMA_RANGE_PERPOINT,
    State, predict, OnlineMap, find_correspondences,
    iekf_update, DT_IMU, NN_K,
)
from sim_imu_trajectory import (
    INITIAL_GYR_BIAS, INITIAL_ACC_BIAS, trajectory_at,
)
from sim_square_corridor_no_lc import (
    run_ieskf_no_lc, prune_to_cube,
    CUBE_LEN, CUBE_REPRUNE_TRIGGER,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
ENV = os.environ.get("SIM_ENV", "default")


def rmse(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def position_error(est_poses, gt_poses):
    return np.array([
        float(np.linalg.norm(p - pg)) for (_, p), (_, pg) in zip(est_poses, gt_poses)
    ])


def collect_perpoint_sigma_distribution(imu_data, lidar_data):
    """Re-run IESKF with per-point cov ON, but also collect the distribution
    of n^T Σ_p n across all matches over all scans, to characterize anisotropy.
    """
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = State(R0, p0.copy(), v0.copy(),
                   INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2
    omap = OnlineMap()
    cube_center = state.p.copy()
    nTSnT_all = []

    li = 0
    for sample in imu_data:
        t = sample["t"]
        state, P = predict(state, P, sample["gyro"], sample["acc"], DT_IMU)
        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                if len(omap) >= NN_K:
                    matches = find_correspondences(state, scan["points_body"], omap)
                    for m in matches:
                        n_w, sigma_p = m[1], m[3]
                        nTSnT_all.append(float(n_w @ sigma_p @ n_w))
                    state, P = iekf_update(state, P, scan["points_body"], omap,
                                            use_fej=False, use_perpoint_cov=True)
                wp = (state.R @ scan["points_body"].T).T + state.p
                omap.add_points(wp)
                if np.max(np.abs(state.p - cube_center)) > CUBE_REPRUNE_TRIGGER:
                    cube_center = state.p.copy()
                    prune_to_cube(omap, cube_center, CUBE_LEN / 2.0)
            li += 1
    return np.array(nTSnT_all)


print(f"Loading sim data for env={ENV}...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

print(f"\n[1/2] IESKF baseline (LIDAR_POINT_VAR={LIDAR_POINT_VAR})...")
np.random.seed(123)
(est_b, t_b, gt_b, _, _, _, _, _, _, _, _, _, _) = run_ieskf_no_lc(imu_data, lidar_data,
                                                              use_perpoint_cov=False)
err_b = position_error(est_b, gt_b)
print(f"  RMSE ATE = {rmse(err_b):.4f} m  mean = {err_b.mean():.4f} m  "
      f"max = {err_b.max():.4f} m")

print(f"\n[2/2] IESKF with per-point Σ (sigma_range={SIGMA_RANGE_PERPOINT})...")
np.random.seed(123)
(est_p, t_p, gt_p, _, _, _, _, _, _, _, _, _, _) = run_ieskf_no_lc(imu_data, lidar_data,
                                                              use_perpoint_cov=True)
err_p = position_error(est_p, gt_p)
print(f"  RMSE ATE = {rmse(err_p):.4f} m  mean = {err_p.mean():.4f} m  "
      f"max = {err_p.max():.4f} m")

improvement = (rmse(err_b) - rmse(err_p)) / rmse(err_b) * 100
print(f"\nRMSE improvement (per-point vs scalar): {improvement:+.1f}%")

print("\nCollecting per-point Σ anisotropy distribution...")
np.random.seed(123)
nTSn = collect_perpoint_sigma_distribution(imu_data, lidar_data)
print(f"  {len(nTSn)} match samples")
print(f"  n^T Σ n per match (m^2): "
      f"median={np.median(nTSn):.3e}  "
      f"p5={np.percentile(nTSn, 5):.3e}  "
      f"p95={np.percentile(nTSn, 95):.3e}  "
      f"p99={np.percentile(nTSn, 99):.3e}")
print(f"  Effective per-point sigma (sqrt(sigma_range^2 + n^T Σ n)) median: "
      f"{np.sqrt(SIGMA_RANGE_PERPOINT**2 + np.median(nTSn))*1000:.2f} mm")
print(f"  Compare to scalar LIDAR_POINT_VAR sigma: "
      f"{np.sqrt(LIDAR_POINT_VAR)*1000:.2f} mm")

t_arr = np.array(t_b)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(t_arr, err_b, "r-", lw=1.0, label=f"baseline (LIDAR_POINT_VAR), RMSE={rmse(err_b):.3f}")
ax.plot(t_arr, err_p, "b-", lw=1.0,
         label=f"per-point Σ, RMSE={rmse(err_p):.3f}")
ax.set_xlabel("t (s)")
ax.set_ylabel("position error (m)")
ax.set_title(f"[{ENV}] No-LC ATE: scalar vs per-point R")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
gt_xy = np.array([p for _, p in gt_b])
b_xy = np.array([p for _, p in est_b])
p_xy = np.array([p for _, p in est_p])
ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", lw=1.5, label="GT")
ax.plot(b_xy[:, 0], b_xy[:, 1], "r-", lw=0.9, alpha=0.8, label="baseline")
ax.plot(p_xy[:, 0], p_xy[:, 1], "b-", lw=0.9, alpha=0.8, label="per-point Σ")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Trajectory comparison")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
nTSn_clip = np.clip(nTSn, 1e-10, 1e-1)
ax.hist(np.log10(nTSn_clip), bins=80, color="#4444aa", alpha=0.7, edgecolor="black")
ax.axvline(np.log10(LIDAR_POINT_VAR), color="r", ls="--", lw=2,
            label=f"scalar LIDAR_POINT_VAR (={LIDAR_POINT_VAR})")
ax.axvline(np.log10(SIGMA_RANGE_PERPOINT**2), color="g", ls="--", lw=2,
            label=f"sigma_range^2 (={SIGMA_RANGE_PERPOINT**2:.1e})")
ax.set_xlabel("log10(n^T Σ_p n)  [m^2]")
ax.set_ylabel("# matches")
ax.set_title("Per-point Σ anisotropy distribution")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/sim_perpoint_diagnosis_{ENV}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")

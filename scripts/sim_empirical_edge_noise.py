"""Empirical per-edge noise vs scan-to-scan CRLB.

Diagnoses whether the scan-to-scan CRLB (which is what we feed to the pose graph
as edge covariance) actually matches the true per-edge relative-pose noise.

For each consecutive keyframe pair (k, k+1):
- Empirical noise = (IESKF relative pose) - (GT relative pose), in world frame
- CRLB cov for that edge (already in world frame)
- Mahalanobis distance: should be ~6 (chi^2 with 6 DOF) if CRLB is calibrated

If mean Mahalanobis >> 6 -> CRLB under-estimates per-edge noise (expected, because
CRLB is the registration noise floor and ignores IMU integration between updates).
If mean ~ 6 -> CRLB is calibrated; gap to fixed-weight LC is from elsewhere.

Run (after generating data for the desired env):
    SIM_ENV=square_corridor pixi run python sim_imu_trajectory.py
    SIM_ENV=square_corridor SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_empirical_edge_noise.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_iekf_3d import rodrigues_log
from sim_square_corridor_no_lc import run_ieskf_no_lc, CUBE_LEN, BIAS_WALK_POS, BIAS_WALK_ROT

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
ENV = os.environ.get("SIM_ENV", "default")


def per_edge_world_error(R_a, p_a, R_b, p_b, R_a_gt, p_a_gt, R_b_gt, p_b_gt):
    """Return (eps_trans_world, eps_rot_world) — error of estimate relative pose
    (a -> b) compared to GT relative pose (a -> b), expressed in world frame.

    Convention chosen to match Adj-rotated CRLB: world-frame perturbation around
    pose a, [trans; rot] order to match P_rel layout (translation first then rot).
    """
    # Estimate relative pose in body of a, then world-frame:
    t_rel_est_world = p_b - p_a
    R_rel_est_world = R_b @ R_a.T

    # GT relative pose in world frame:
    t_rel_gt_world = p_b_gt - p_a_gt
    R_rel_gt_world = R_b_gt @ R_a_gt.T

    eps_trans = t_rel_est_world - t_rel_gt_world
    # Right-multiplied error: R_rel_est = Exp(eps_rot) · R_rel_gt
    eps_rot = rodrigues_log(R_rel_est_world @ R_rel_gt_world.T)
    return eps_trans, eps_rot


print(f"Loading sim data for env={ENV}...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")
print(f"  cube_len={CUBE_LEN}, bias_walk pos={BIAS_WALK_POS}, rot={BIAS_WALK_ROT}")

print("\nRunning IESKF...")
np.random.seed(123)
(est_poses, est_times, gt_poses, _, _, _, _,
 _, _, _, edge_covs, _) = run_ieskf_no_lc(imu_data, lidar_data)
n = len(est_poses)
print(f"  {n} keyframes, {sum(1 for c in edge_covs if c is not None)} valid edges")

print("\nComputing per-edge empirical noise vs CRLB...")

eps_trans_all = []
eps_rot_all = []
mahalanobis_all = []
mahalanobis_pos = []
mahalanobis_rot = []
crlb_pos_trace = []
crlb_rot_trace = []
emp_pos_norm = []
emp_rot_norm = []

for k in range(n - 1):
    cov = edge_covs[k + 1]
    if cov is None:
        continue
    R_a, p_a = est_poses[k]
    R_b, p_b = est_poses[k + 1]
    R_a_gt, p_a_gt = gt_poses[k]
    R_b_gt, p_b_gt = gt_poses[k + 1]

    eps_t, eps_r = per_edge_world_error(R_a, p_a, R_b, p_b,
                                        R_a_gt, p_a_gt, R_b_gt, p_b_gt)
    eps_trans_all.append(eps_t)
    eps_rot_all.append(eps_r)

    # CRLB stored as [trans; rot] (per compute_scan_to_scan_covariance)
    eps = np.concatenate([eps_t, eps_r])
    cov_reg = cov + np.eye(6) * 1e-12
    try:
        m_full = float(eps @ np.linalg.solve(cov_reg, eps))
    except np.linalg.LinAlgError:
        continue
    mahalanobis_all.append(m_full)

    cov_pos = cov[0:3, 0:3] + np.eye(3) * 1e-12
    cov_rot = cov[3:6, 3:6] + np.eye(3) * 1e-12
    try:
        m_pos = float(eps_t @ np.linalg.solve(cov_pos, eps_t))
        m_rot = float(eps_r @ np.linalg.solve(cov_rot, eps_r))
    except np.linalg.LinAlgError:
        continue
    mahalanobis_pos.append(m_pos)
    mahalanobis_rot.append(m_rot)

    crlb_pos_trace.append(float(np.trace(cov[0:3, 0:3])))
    crlb_rot_trace.append(float(np.trace(cov[3:6, 3:6])))
    emp_pos_norm.append(float(np.dot(eps_t, eps_t)))
    emp_rot_norm.append(float(np.dot(eps_r, eps_r)))

mahalanobis_all = np.array(mahalanobis_all)
mahalanobis_pos = np.array(mahalanobis_pos)
mahalanobis_rot = np.array(mahalanobis_rot)
crlb_pos_trace = np.array(crlb_pos_trace)
crlb_rot_trace = np.array(crlb_rot_trace)
emp_pos_norm = np.array(emp_pos_norm)
emp_rot_norm = np.array(emp_rot_norm)

print(f"\nValid edges analyzed: {len(mahalanobis_all)}")

print("\n=== Mahalanobis (expected mean if CRLB is calibrated) ===")
print(f"  Full edge (6 DOF, target ~6):  mean={mahalanobis_all.mean():>10.2f}  "
      f"median={np.median(mahalanobis_all):>10.2f}  "
      f"p95={np.percentile(mahalanobis_all, 95):>10.2f}")
print(f"  Position only (3 DOF, ~3):     mean={mahalanobis_pos.mean():>10.2f}  "
      f"median={np.median(mahalanobis_pos):>10.2f}  "
      f"p95={np.percentile(mahalanobis_pos, 95):>10.2f}")
print(f"  Rotation only (3 DOF, ~3):     mean={mahalanobis_rot.mean():>10.2f}  "
      f"median={np.median(mahalanobis_rot):>10.2f}  "
      f"p95={np.percentile(mahalanobis_rot, 95):>10.2f}")

ratio_pos = np.sqrt(emp_pos_norm / np.maximum(crlb_pos_trace, 1e-15))
ratio_rot = np.sqrt(emp_rot_norm / np.maximum(crlb_rot_trace, 1e-15))
print(f"\n=== Empirical / CRLB sigma ratio (per edge, target ~1 if calibrated) ===")
print(f"  Pos:  median={np.median(ratio_pos):>8.2f}  p95={np.percentile(ratio_pos, 95):>8.2f}")
print(f"  Rot:  median={np.median(ratio_rot):>8.2f}  p95={np.percentile(ratio_rot, 95):>8.2f}")

print("\n=== Empirical per-edge noise stats ===")
print(f"  Pos sigma (RMS over edges, m):  {np.sqrt(emp_pos_norm.mean() / 3):.4e}")
print(f"  Rot sigma (RMS over edges, rad):{np.sqrt(emp_rot_norm.mean() / 3):.4e}")
print(f"  CRLB pos sigma (RMS):           {np.sqrt(crlb_pos_trace.mean() / 3):.4e}")
print(f"  CRLB rot sigma (RMS):           {np.sqrt(crlb_rot_trace.mean() / 3):.4e}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
bins = np.linspace(0, max(np.percentile(mahalanobis_all, 99), 30), 60)
ax.hist(mahalanobis_all, bins=bins, color="#cc4444", alpha=0.7, edgecolor="black")
ax.axvline(6, color="g", ls="--", lw=2, label="expected (chi^2_6)")
ax.axvline(np.median(mahalanobis_all), color="b", ls="-", lw=1.5, label=f"median={np.median(mahalanobis_all):.1f}")
ax.set_xlabel("Mahalanobis (full 6 DOF)")
ax.set_ylabel("# edges")
ax.set_title(f"[{ENV}] Per-edge Mahalanobis: empirical vs CRLB")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
bins = np.linspace(0, max(np.percentile(mahalanobis_pos, 99), 20), 60)
ax.hist(mahalanobis_pos, bins=bins, color="#4444cc", alpha=0.7, edgecolor="black")
ax.axvline(3, color="g", ls="--", lw=2, label="expected (chi^2_3)")
ax.axvline(np.median(mahalanobis_pos), color="r", ls="-", lw=1.5,
           label=f"median={np.median(mahalanobis_pos):.1f}")
ax.set_xlabel("Mahalanobis (position only)")
ax.set_ylabel("# edges")
ax.set_title("Position-only edge consistency")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
bins = np.linspace(0, max(np.percentile(mahalanobis_rot, 99), 20), 60)
ax.hist(mahalanobis_rot, bins=bins, color="#44cc44", alpha=0.7, edgecolor="black")
ax.axvline(3, color="g", ls="--", lw=2, label="expected (chi^2_3)")
ax.axvline(np.median(mahalanobis_rot), color="r", ls="-", lw=1.5,
           label=f"median={np.median(mahalanobis_rot):.1f}")
ax.set_xlabel("Mahalanobis (rotation only)")
ax.set_ylabel("# edges")
ax.set_title("Rotation-only edge consistency")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.scatter(np.sqrt(crlb_pos_trace), np.sqrt(emp_pos_norm), s=4, alpha=0.4, color="#cc4444")
mx = max(np.sqrt(crlb_pos_trace).max(), np.sqrt(emp_pos_norm).max())
ax.plot([0, mx], [0, mx], "k--", lw=1, label="y=x (calibrated)")
ax.set_xlabel("CRLB sqrt(trace) per edge (m)")
ax.set_ylabel("Empirical |dt| per edge (m)")
ax.set_title("Position: empirical vs CRLB per edge")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(np.sqrt(crlb_rot_trace), np.sqrt(emp_rot_norm), s=4, alpha=0.4, color="#44cc44")
mx = max(np.sqrt(crlb_rot_trace).max(), np.sqrt(emp_rot_norm).max())
ax.plot([0, mx], [0, mx], "k--", lw=1, label="y=x (calibrated)")
ax.set_xlabel("CRLB sqrt(trace) per edge (rad)")
ax.set_ylabel("Empirical |dtheta| per edge (rad)")
ax.set_title("Rotation: empirical vs CRLB per edge")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
edge_idx = np.arange(len(ratio_pos))
ax.semilogy(edge_idx, ratio_pos, ".", ms=2, alpha=0.4, label="pos ratio", color="#cc4444")
ax.semilogy(edge_idx, ratio_rot, ".", ms=2, alpha=0.4, label="rot ratio", color="#44cc44")
ax.axhline(1.0, color="g", ls="--", lw=1.2, label="calibrated (ratio=1)")
ax.set_xlabel("edge index")
ax.set_ylabel("empirical / CRLB sigma ratio (log)")
ax.set_title("Per-edge under-estimation factor")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
out = f"{OUT_DIR}/sim_edge_noise_diagnosis_{ENV}.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved {out}")

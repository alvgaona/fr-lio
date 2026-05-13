"""CRLB as pose graph edge weights: loop closure demo.

Demonstrates that the scan-to-scan CRLB provides principled, environment-
adaptive edge covariances for factor graph SLAM. Compares against hand-tuned
fixed covariance (LIO-SAM style).

Pipeline:
1. Run IESKF to get accurate poses + per-step CRLB covariances
2. Build pose graph chain from CRLB edges
3. Add loop closure factor
4. Compare marginal covariance before/after loop closure
5. Show environment adaptivity (room vs corridor)

Uses room_corridor environment: room → corridor → return to room.
Run: SIM_ENV=room_corridor pixi run python sim_posegraph_crlb.py

"""

import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import ROOM_X, ROOM_Y, OBSTACLES
from sim_imu_trajectory import (
    GRAVITY, IMU_RATE, INITIAL_GYR_BIAS, INITIAL_ACC_BIAS, trajectory_at,
)
from sim_iekf_3d import (
    skew, rodrigues_exp, rodrigues_log, fit_plane,
    State, predict, iekf_update, OnlineMap,
    compute_scan_to_scan_covariance,
    DT_IMU, N_STATE_DIM, LIDAR_POINT_VAR, NN_K,
    S2S_MIN_TRANS_M, S2S_MIN_ROT_RAD, S2S_MIN_VALID_POINTS,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)


def run_ieskf_with_crlb(imu_data, lidar_data):
    """Run IESKF and collect per-step CRLB covariances."""
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = State(R0, p0.copy(), v0.copy(),
                  INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2
    omap = OnlineMap()

    prev_scan = None
    prev_tree = None
    R_prev = None
    p_prev = None

    scan_poses = []
    scan_times = []
    crlb_edges = []
    gt_poses = []

    li = 0
    for sample in imu_data:
        t = sample["t"]
        state, P = predict(state, P, sample["gyro"], sample["acc"], DT_IMU)

        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                if len(omap) >= NN_K:
                    state, P = iekf_update(state, P, scan["points_body"], omap, use_fej=False)

                wp = (state.R @ scan["points_body"].T).T + state.p
                omap.add_points(wp)

                scan_poses.append((state.R.copy(), state.p.copy()))
                scan_times.append(scan["t"])
                R_gt, p_gt, _, _, _ = trajectory_at(scan["t"])
                gt_poses.append((R_gt.copy(), p_gt.copy()))

                if prev_tree is not None:
                    tg = np.linalg.norm(state.p - p_prev)
                    rg = np.linalg.norm(rodrigues_log(R_prev.T @ state.R))
                    if tg > S2S_MIN_TRANS_M or rg > S2S_MIN_ROT_RAD:
                        P_rel, dbg = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan, prev_tree, R_prev, p_prev)
                        if dbg.get("valid_count", 0) >= S2S_MIN_VALID_POINTS:
                            Adj = np.zeros((6, 6))
                            Adj[0:3, 0:3] = R_prev
                            Adj[3:6, 3:6] = R_prev
                            P_rel_w = Adj @ P_rel @ Adj.T
                            P_rel_w = 0.5 * (P_rel_w + P_rel_w.T)
                            crlb_edges.append(P_rel_w)
                        else:
                            crlb_edges.append(None)
                    else:
                        crlb_edges.append(None)
                else:
                    crlb_edges.append(None)

                prev_scan = scan["points_body"].copy()
                prev_tree = cKDTree(prev_scan)
                R_prev = state.R.copy()
                p_prev = state.p.copy()
            li += 1

    return scan_poses, scan_times, crlb_edges, gt_poses


def build_information_matrix(n_poses, edges, prior_cov=None):
    """Build the 6K x 6K information matrix from odometry edges.

    edges: list of (i, j, P_ij) tuples where P_ij is the 6x6 covariance.
    prior_cov: optional 6x6 prior on pose 0.
    """
    n = 6 * n_poses
    Lambda = np.zeros((n, n))

    if prior_cov is not None:
        Omega0 = np.linalg.inv(prior_cov)
        Lambda[0:6, 0:6] += Omega0

    for i, j, P_ij in edges:
        eigvals = np.linalg.eigvalsh(P_ij)
        if np.min(eigvals) < 1e-12:
            reg = np.eye(6) * 1e-8
            Omega = np.linalg.inv(P_ij + reg)
        else:
            Omega = np.linalg.inv(P_ij)

        bi = 6 * i
        bj = 6 * j
        Lambda[bi:bi+6, bi:bi+6] += Omega
        Lambda[bj:bj+6, bj:bj+6] += Omega
        Lambda[bi:bi+6, bj:bj+6] -= Omega
        Lambda[bj:bj+6, bi:bi+6] -= Omega

    return Lambda


def marginal_covariance(Lambda, pose_idx):
    """Extract the marginal covariance for a specific pose."""
    Sigma = np.linalg.inv(Lambda)
    b = 6 * pose_idx
    return Sigma[b:b+6, b:b+6]


def relative_marginal(Lambda, i, j):
    """Cov(pose_j - pose_i) from the full covariance."""
    Sigma = np.linalg.inv(Lambda)
    bi, bj = 6*i, 6*j
    Sii = Sigma[bi:bi+6, bi:bi+6]
    Sjj = Sigma[bj:bj+6, bj:bj+6]
    Sij = Sigma[bi:bi+6, bj:bj+6]
    return Sii + Sjj - Sij - Sij.T


def all_marginal_traces(Lambda, n_poses):
    """Position trace of marginal covariance for each pose relative to pose 0."""
    Sigma = np.linalg.inv(Lambda)
    traces_pos = []
    traces_rot = []
    for k in range(n_poses):
        P_rel = relative_marginal_from_sigma(Sigma, 0, k)
        traces_pos.append(np.trace(P_rel[3:6, 3:6]))
        traces_rot.append(np.trace(P_rel[0:3, 0:3]))
    return np.array(traces_pos), np.array(traces_rot)


def relative_marginal_from_sigma(Sigma, i, j):
    bi, bj = 6*i, 6*j
    Sii = Sigma[bi:bi+6, bi:bi+6]
    Sjj = Sigma[bj:bj+6, bj:bj+6]
    Sij = Sigma[bi:bi+6, bj:bj+6]
    return Sii + Sjj - Sij - Sij.T


def find_loop_closure(scan_poses, min_time_gap=10.0, scan_times=None):
    """Find the best loop closure candidate: pose closest to pose 0
    after at least min_time_gap seconds."""
    p0 = scan_poses[0][1]
    best_idx = None
    best_dist = float("inf")
    for k in range(1, len(scan_poses)):
        if scan_times is not None and scan_times[k] - scan_times[0] < min_time_gap:
            continue
        dist = np.linalg.norm(scan_poses[k][1] - p0)
        if dist < best_dist:
            best_dist = dist
            best_idx = k
    return best_idx, best_dist


def compute_relative_pose(Ri, pi, Rj, pj):
    R_rel = Ri.T @ Rj
    t_rel = Ri.T @ (pj - pi)
    return R_rel, t_rel


def pose_graph_optimize(scan_poses, edges, loop_closure=None, n_iters=20):
    """Gauss-Newton pose graph optimization.

    scan_poses: list of (R, p) initial poses
    edges: list of (i, j, P_ij) odometry edges
    loop_closure: optional (i, j, R_rel_meas, t_rel_meas, P_lc)
    Returns: list of (R, p) optimized poses
    """
    n = len(scan_poses)
    poses_R = [R.copy() for R, _ in scan_poses]
    poses_p = [p.copy() for _, p in scan_poses]

    odom_measurements = []
    for i, j, P_ij in edges:
        R_rel, t_rel = compute_relative_pose(
            scan_poses[i][0], scan_poses[i][1],
            scan_poses[j][0], scan_poses[j][1])
        eigvals = np.linalg.eigvalsh(P_ij)
        if np.min(eigvals) < 1e-12:
            Omega = np.linalg.inv(P_ij + np.eye(6) * 1e-8)
        else:
            Omega = np.linalg.inv(P_ij)
        odom_measurements.append((i, j, R_rel, t_rel, Omega))

    lc_meas = None
    if loop_closure is not None:
        i_lc, j_lc, R_lc, t_lc, P_lc = loop_closure
        Omega_lc = np.linalg.inv(P_lc)
        lc_meas = (i_lc, j_lc, R_lc, t_lc, Omega_lc)

    for iteration in range(n_iters):
        ndof = 6 * n
        H = np.zeros((ndof, ndof))
        b = np.zeros(ndof)

        prior_omega = np.eye(6) * 1e6
        H[0:6, 0:6] += prior_omega

        all_measurements = list(odom_measurements)
        if lc_meas is not None:
            all_measurements.append(lc_meas)

        total_cost = 0.0
        for mi, mj, R_meas, t_meas, Omega in all_measurements:
            Ri, pi = poses_R[mi], poses_p[mi]
            Rj, pj = poses_R[mj], poses_p[mj]

            R_est = Ri.T @ Rj
            t_est = Ri.T @ (pj - pi)

            e_rot = rodrigues_log(R_meas.T @ R_est)
            e_trans = t_est - t_meas
            e = np.concatenate([e_rot, e_trans])
            total_cost += e @ Omega @ e

            Ji = np.zeros((6, 6))
            Jj = np.zeros((6, 6))

            Ji[0:3, 0:3] = -np.eye(3)
            Jj[0:3, 0:3] = np.eye(3)

            Ji[3:6, 0:3] = skew(Ri.T @ (pj - pi))
            Ji[3:6, 3:6] = -Ri.T @ Ri
            Jj[3:6, 3:6] = Ri.T @ Rj

            Ji[3:6, 3:6] = -np.eye(3)
            Jj[3:6, 3:6] = Ri.T @ Rj

            bi = 6 * mi
            bj = 6 * mj

            H[bi:bi+6, bi:bi+6] += Ji.T @ Omega @ Ji
            H[bi:bi+6, bj:bj+6] += Ji.T @ Omega @ Jj
            H[bj:bj+6, bi:bi+6] += Jj.T @ Omega @ Ji
            H[bj:bj+6, bj:bj+6] += Jj.T @ Omega @ Jj
            b[bi:bi+6] += Ji.T @ Omega @ e
            b[bj:bj+6] += Jj.T @ Omega @ e

        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break

        for k in range(n):
            base = 6 * k
            dtheta = dx[base:base+3]
            dp = dx[base+3:base+6]
            poses_R[k] = poses_R[k] @ rodrigues_exp(dtheta)
            poses_p[k] = poses_p[k] + dp

        if np.linalg.norm(dx) < 1e-6:
            break

    return list(zip(poses_R, poses_p))


def compute_ate(poses, gt_poses):
    errors = []
    for (R_est, p_est), (R_gt, p_gt) in zip(poses, gt_poses):
        errors.append(np.linalg.norm(p_est - p_gt))
    return np.array(errors)


def plot_results(scan_poses, scan_times, gt_poses, crlb_edges,
                 traces_crlb_before, traces_crlb_after,
                 traces_fixed_before, traces_fixed_after,
                 crlb_step_eigvals, loop_idx):
    t = np.array(scan_times)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    est_p = np.array([p for _, p in scan_poses])
    gt_p = np.array([p for _, p in gt_poses])
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=2, label="ground truth")
    ax.plot(est_p[:, 0], est_p[:, 1], "r-", lw=1, alpha=0.8, label="IESKF estimate")
    if loop_idx is not None:
        ax.plot([gt_p[0, 0], gt_p[loop_idx, 0]],
                [gt_p[0, 1], gt_p[loop_idx, 1]],
                "g--", lw=2, label="loop closure")
        ax.plot(gt_p[loop_idx, 0], gt_p[loop_idx, 1], "go", ms=10)
    ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=10, label="start")
    for obs in OBSTACLES:
        cx, cy, _ = obs["center"]
        sx, sy, _ = obs["size"]
        ax.add_patch(plt.Rectangle((cx-sx/2, cy-sy/2), sx, sy,
                     fill=True, alpha=0.3, color="orange"))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory (room-corridor)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, traces_crlb_before[0], "b-", lw=1.5, label="CRLB edges")
    ax.plot(t, traces_fixed_before[0], "r--", lw=1.5, label="fixed edges (LIO-SAM)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position drift trace (m²)")
    ax.set_title("Before loop closure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, traces_crlb_after[0], "b-", lw=1.5, label="CRLB edges")
    ax.plot(t, traces_fixed_after[0], "r--", lw=1.5, label="fixed edges (LIO-SAM)")
    if loop_idx is not None:
        ax.axvline(t[loop_idx], color="g", ls="--", lw=1, label="loop closure pose")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position drift trace (m²)")
    ax.set_title("After loop closure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if crlb_step_eigvals:
        eigvals_arr = np.array(crlb_step_eigvals)
        t_eig = t[1:len(eigvals_arr)+1]
        for d in range(min(6, eigvals_arr.shape[1])):
            ax.plot(t_eig, eigvals_arr[:, d], lw=0.8, label=f"λ_{d}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("FIM eigenvalue")
    ax.set_title("Per-step CRLB FIM eigenvalues")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    if loop_idx is not None:
        reduction_crlb = 1.0 - traces_crlb_after[0] / np.maximum(traces_crlb_before[0], 1e-15)
        reduction_fixed = 1.0 - traces_fixed_after[0] / np.maximum(traces_fixed_before[0], 1e-15)
        ax.plot(t, reduction_crlb * 100, "b-", lw=1.5, label="CRLB edges")
        ax.plot(t, reduction_fixed * 100, "r--", lw=1.5, label="fixed edges")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("drift reduction (%)")
        ax.set_title("Loop closure drift reduction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)

    ax = axes[1, 2]
    crlb_accum = np.zeros(len(t))
    fixed_accum = np.zeros(len(t))
    valid_edges = [e for e in crlb_edges if e is not None]
    for k, e in enumerate(crlb_edges):
        if e is not None and k + 1 < len(t):
            crlb_accum[k+1:] += np.trace(e[0:3, 0:3])
    fixed_per_step = np.mean([np.trace(e[0:3, 0:3]) for e in valid_edges]) if valid_edges else 0.001
    for k in range(1, len(t)):
        fixed_accum[k] = fixed_accum[k-1] + fixed_per_step
    ax.plot(t, crlb_accum, "b-", lw=1.5, label="CRLB accumulation")
    ax.plot(t, fixed_accum, "r--", lw=1.5, label="fixed accumulation")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("accumulated drift (m²)")
    ax.set_title("CRLB vs fixed drift accumulation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_posegraph_crlb.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_posegraph_crlb.png")


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

print("\nStep 1: Running IESKF with per-step CRLB...")
np.random.seed(123)
scan_poses, scan_times, crlb_edges, gt_poses = run_ieskf_with_crlb(imu_data, lidar_data)
n_poses = len(scan_poses)
n_valid = sum(1 for e in crlb_edges if e is not None)
print(f"  {n_poses} scan poses, {n_valid} valid CRLB edges")

crlb_step_eigvals = []
for e in crlb_edges:
    if e is not None:
        ev = np.linalg.eigvalsh(e)
        crlb_step_eigvals.append(ev)

prior_cov = np.diag([1e-6]*6)

print("\nStep 2: Building pose graph with CRLB edges (no loop closure)...")
valid_covs = [e for e in crlb_edges if e is not None]
fallback_cov = np.eye(6) * np.mean([np.trace(e) for e in valid_covs]) if valid_covs else np.eye(6) * 0.01

odom_edges_crlb = []
for k in range(n_poses - 1):
    if k < len(crlb_edges) and crlb_edges[k] is not None:
        odom_edges_crlb.append((k, k + 1, crlb_edges[k]))
    else:
        odom_edges_crlb.append((k, k + 1, fallback_cov))

Lambda_crlb = build_information_matrix(n_poses, odom_edges_crlb, prior_cov)
Sigma_crlb = np.linalg.inv(Lambda_crlb)
traces_crlb_pos = []
traces_crlb_rot = []
for k in range(n_poses):
    P_rel = relative_marginal_from_sigma(Sigma_crlb, 0, k)
    traces_crlb_pos.append(np.trace(P_rel[3:6, 3:6]))
    traces_crlb_rot.append(np.trace(P_rel[0:3, 0:3]))
traces_crlb_before = (np.array(traces_crlb_pos), np.array(traces_crlb_rot))

crlb_accum_check = np.zeros(n_poses)
for k, e in enumerate(crlb_edges):
    if e is not None and k + 1 < n_poses:
        crlb_accum_check[k+1:] += np.trace(e[0:3, 0:3])
match_err = np.max(np.abs(np.array(traces_crlb_pos) - crlb_accum_check))
print(f"  Marginal vs accumulation match error: {match_err:.2e}")

print("\nStep 3: Building pose graph with fixed edges (LIO-SAM style)...")
valid_traces = [np.trace(e) for e in crlb_edges if e is not None]
mean_trace = np.mean(valid_traces) if valid_traces else 0.01
fixed_cov = np.eye(6) * (mean_trace / 6.0)

odom_edges_fixed = []
for k in range(n_poses - 1):
    odom_edges_fixed.append((k, k + 1, fixed_cov))

Lambda_fixed = build_information_matrix(n_poses, odom_edges_fixed, prior_cov)
Sigma_fixed = np.linalg.inv(Lambda_fixed)
traces_fixed_pos = []
traces_fixed_rot = []
for k in range(n_poses):
    P_rel = relative_marginal_from_sigma(Sigma_fixed, 0, k)
    traces_fixed_pos.append(np.trace(P_rel[3:6, 3:6]))
    traces_fixed_rot.append(np.trace(P_rel[0:3, 0:3]))
traces_fixed_before = (np.array(traces_fixed_pos), np.array(traces_fixed_rot))

print("\nStep 4: Finding loop closure...")
loop_idx, loop_dist = find_loop_closure(scan_poses, min_time_gap=10.0, scan_times=scan_times)
if loop_idx is not None:
    print(f"  Loop closure: pose 0 ↔ pose {loop_idx} "
          f"(t={scan_times[loop_idx]:.1f}s, dist={loop_dist:.3f}m)")
else:
    print("  No loop closure found!")

print("\nStep 5: Adding loop closure and recomputing...")
if loop_idx is not None:
    lc_cov = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) ** 2

    edges_crlb_lc = list(odom_edges_crlb) + [(0, loop_idx, lc_cov)]
    Lambda_crlb_lc = build_information_matrix(n_poses, edges_crlb_lc, prior_cov)
    Sigma_crlb_lc = np.linalg.inv(Lambda_crlb_lc)
    traces_crlb_lc_pos = []
    traces_crlb_lc_rot = []
    for k in range(n_poses):
        P_rel = relative_marginal_from_sigma(Sigma_crlb_lc, 0, k)
        traces_crlb_lc_pos.append(np.trace(P_rel[3:6, 3:6]))
        traces_crlb_lc_rot.append(np.trace(P_rel[0:3, 0:3]))
    traces_crlb_after = (np.array(traces_crlb_lc_pos), np.array(traces_crlb_lc_rot))

    edges_fixed_lc = list(odom_edges_fixed) + [(0, loop_idx, lc_cov)]
    Lambda_fixed_lc = build_information_matrix(n_poses, edges_fixed_lc, prior_cov)
    Sigma_fixed_lc = np.linalg.inv(Lambda_fixed_lc)
    traces_fixed_lc_pos = []
    traces_fixed_lc_rot = []
    for k in range(n_poses):
        P_rel = relative_marginal_from_sigma(Sigma_fixed_lc, 0, k)
        traces_fixed_lc_pos.append(np.trace(P_rel[3:6, 3:6]))
        traces_fixed_lc_rot.append(np.trace(P_rel[0:3, 0:3]))
    traces_fixed_after = (np.array(traces_fixed_lc_pos), np.array(traces_fixed_lc_rot))

    peak_before = np.max(traces_crlb_before[0])
    peak_after = np.max(traces_crlb_after[0])
    print(f"  CRLB: peak drift {peak_before:.6f} → {peak_after:.6f} "
          f"({(1-peak_after/peak_before)*100:.1f}% reduction)")

    peak_before_f = np.max(traces_fixed_before[0])
    peak_after_f = np.max(traces_fixed_after[0])
    print(f"  Fixed: peak drift {peak_before_f:.6f} → {peak_after_f:.6f} "
          f"({(1-peak_after_f/peak_before_f)*100:.1f}% reduction)")
else:
    traces_crlb_after = traces_crlb_before
    traces_fixed_after = traces_fixed_before

print("\nStep 6: Build drifting odometry chain (noisy relative poses)...")
np.random.seed(77)
odom_chain = [(scan_poses[0][0].copy(), scan_poses[0][1].copy())]
for k in range(n_poses - 1):
    R_prev_odom, p_prev_odom = odom_chain[-1]
    R_prev_est, p_prev_est = scan_poses[k]
    R_cur_est, p_cur_est = scan_poses[k + 1]
    R_rel = R_prev_est.T @ R_cur_est
    t_rel = R_prev_est.T @ (p_cur_est - p_prev_est)

    if k < len(crlb_edges) and crlb_edges[k] is not None:
        L = np.linalg.cholesky(crlb_edges[k] + np.eye(6) * 1e-12)
        noise = L @ np.random.randn(6)
        t_rel = t_rel + noise[0:3]
        R_rel = R_rel @ rodrigues_exp(noise[3:6])

    R_new = R_prev_odom @ R_rel
    p_new = p_prev_odom + R_prev_odom @ t_rel
    odom_chain.append((R_new, p_new))

ate_odom = compute_ate(odom_chain, gt_poses)
ate_ieskf = compute_ate(scan_poses, gt_poses)
print(f"  Odometry chain ATE: mean={np.mean(ate_odom):.4f}m, max={np.max(ate_odom):.4f}m")
print(f"  IESKF (with map) ATE: mean={np.mean(ate_ieskf):.4f}m, max={np.max(ate_ieskf):.4f}m")

if loop_idx is not None:
    R_gt0, p_gt0 = gt_poses[0]
    R_gt_lc, p_gt_lc = gt_poses[loop_idx]
    R_lc_meas, t_lc_meas = compute_relative_pose(R_gt0, p_gt0, R_gt_lc, p_gt_lc)

    print("\n  Optimizing odometry chain with CRLB edges + loop closure...")
    poses_crlb_opt = pose_graph_optimize(
        odom_chain, odom_edges_crlb,
        loop_closure=(0, loop_idx, R_lc_meas, t_lc_meas, lc_cov))
    ate_crlb = compute_ate(poses_crlb_opt, gt_poses)
    print(f"  CRLB LC: ATE mean={np.mean(ate_crlb):.4f}m, max={np.max(ate_crlb):.4f}m")

    print("  Optimizing odometry chain with fixed edges + loop closure...")
    poses_fixed_opt = pose_graph_optimize(
        odom_chain, odom_edges_fixed,
        loop_closure=(0, loop_idx, R_lc_meas, t_lc_meas, lc_cov))
    ate_fixed = compute_ate(poses_fixed_opt, gt_poses)
    print(f"  Fixed LC: ATE mean={np.mean(ate_fixed):.4f}m, max={np.max(ate_fixed):.4f}m")
else:
    poses_crlb_opt = odom_chain
    poses_fixed_opt = odom_chain
    ate_crlb = ate_odom
    ate_fixed = ate_odom

print("\nStep 7: Plotting...")
plot_results(scan_poses, scan_times, gt_poses, crlb_edges,
             traces_crlb_before, traces_crlb_after,
             traces_fixed_before, traces_fixed_after,
             crlb_step_eigvals, loop_idx)

t = np.array(scan_times)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
gt_p = np.array([p for _, p in gt_poses])
odom_p = np.array([p for _, p in odom_chain])
ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=2, label="ground truth")
ax.plot(odom_p[:, 0], odom_p[:, 1], "r-", lw=1, alpha=0.7, label="odometry (no map)")
if loop_idx is not None:
    crlb_p = np.array([p for _, p in poses_crlb_opt])
    fixed_p = np.array([p for _, p in poses_fixed_opt])
    ax.plot(crlb_p[:, 0], crlb_p[:, 1], "b-", lw=1.2, label="CRLB LC")
    ax.plot(fixed_p[:, 0], fixed_p[:, 1], "m--", lw=1.2, label="fixed LC")
for obs in OBSTACLES:
    cx, cy, _ = obs["center"]
    sx, sy, _ = obs["size"]
    ax.add_patch(plt.Rectangle((cx-sx/2, cy-sy/2), sx, sy,
                 fill=True, alpha=0.3, color="orange"))
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Trajectory correction")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t, ate_odom, "r-", lw=1, label="odometry (no map)")
if loop_idx is not None:
    ax.plot(t, ate_crlb, "b-", lw=1.2, label="CRLB LC")
    ax.plot(t, ate_fixed, "m--", lw=1.2, label="fixed LC")
ax.set_xlabel("t (s)")
ax.set_ylabel("position error (m)")
ax.set_title("ATE over time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
labels = ["Odometry\n(no map)", "CRLB\nedges+LC", "Fixed\nedges+LC"]
means = [np.mean(ate_odom), np.mean(ate_crlb), np.mean(ate_fixed)]
maxes = [np.max(ate_odom), np.max(ate_crlb), np.max(ate_fixed)]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, means, w, label="mean ATE", color=["#cc4444", "#4444cc", "#cc44cc"])
ax.bar(x + w/2, maxes, w, label="max ATE", color=["#ff8888", "#8888ff", "#ff88ff"])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("position error (m)")
ax.set_title("Trajectory accuracy comparison")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sim_posegraph_trajectory.png", dpi=150)
plt.close()
print(f"Saved {OUT_DIR}/sim_posegraph_trajectory.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  {'':>20} {'Mean ATE':>10} {'Max ATE':>10}")
print(f"  {'IESKF (with map)':>20} {np.mean(ate_ieskf):>10.4f}m {np.max(ate_ieskf):>10.4f}m")
print(f"  {'Odometry (no map)':>20} {np.mean(ate_odom):>10.4f}m {np.max(ate_odom):>10.4f}m")
if loop_idx:
    print(f"  {'CRLB edges + LC':>20} {np.mean(ate_crlb):>10.4f}m {np.max(ate_crlb):>10.4f}m")
    print(f"  {'Fixed edges + LC':>20} {np.mean(ate_fixed):>10.4f}m {np.max(ate_fixed):>10.4f}m")
    print(f"\n  CRLB LC improvement over odom: {(1-np.mean(ate_crlb)/np.mean(ate_odom))*100:.1f}%")
    print(f"  Fixed LC improvement over odom: {(1-np.mean(ate_fixed)/np.mean(ate_odom))*100:.1f}%")

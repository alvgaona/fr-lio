"""Square corridor multi-lap with FIXED-WEIGHT loop closure (LIO-SAM style).

Baseline for the loop closure paper. Runs the IESKF (no LC) to get drifted
keyframes, detects loop closures by ground-truth proximity (proxy for a Scan
Context / descriptor-based detector), and runs Gauss-Newton pose graph
optimization with a constant hand-tuned covariance for both odometry and LC
edges. Compares ATE against the no-LC baseline.

Run:
    SIM_ENV=square_corridor SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_square_corridor_fixed_lc.py

Assumes sim_imu_trajectory.py has already been run with SIM_ENV=square_corridor.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import (
    trajectory_at, SQ_PERIM, SQ_SPEED, SQ_N_LAPS,
)
from sim_iekf_3d import skew, rodrigues_exp, rodrigues_log
from sim_square_corridor_no_lc import (
    run_ieskf_no_lc, compute_nees, position_error, yaw_error,
    plot_walls, OUTER, INNER_LO, INNER_HI, CUBE_LEN,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)

LC_DIST_THRESH = float(os.environ.get("SIM_LC_DIST", "1.0"))
LC_MIN_TIME_GAP = float(os.environ.get("SIM_LC_MIN_GAP", "10.0"))
LC_MIN_SPACING = float(os.environ.get("SIM_LC_MIN_SPACING", "5.0"))
LC_NOISE_POS = float(os.environ.get("SIM_LC_NOISE_POS", "0.05"))
LC_NOISE_ROT = float(os.environ.get("SIM_LC_NOISE_ROT", "0.005"))

FIXED_ODOM_POS_STD = float(os.environ.get("SIM_FIXED_ODOM_POS_STD", "0.05"))
FIXED_ODOM_ROT_STD = float(os.environ.get("SIM_FIXED_ODOM_ROT_STD", "0.005"))
FIXED_LC_POS_STD = float(os.environ.get("SIM_FIXED_LC_POS_STD", "0.05"))
FIXED_LC_ROT_STD = float(os.environ.get("SIM_FIXED_LC_ROT_STD", "0.005"))


def relative_pose(Ri, pi, Rj, pj):
    R_rel = Ri.T @ Rj
    t_rel = Ri.T @ (pj - pi)
    return R_rel, t_rel


def build_odom_edges(est_poses, fixed_cov):
    edges = []
    for k in range(len(est_poses) - 1):
        edges.append((k, k + 1, fixed_cov))
    return edges


def detect_loop_closures(gt_poses, est_times, dist_thresh,
                         min_time_gap, min_spacing):
    """Return list of (i, j) pairs where j > i, GT positions within dist_thresh,
    and time gap > min_time_gap. Suppress closures that are too close in
    time to a previously emitted closure.
    """
    n = len(gt_poses)
    closures = []
    last_emit_t = -np.inf
    for j in range(1, n):
        for i in range(j):
            if est_times[j] - est_times[i] < min_time_gap:
                continue
            d = np.linalg.norm(gt_poses[j][1] - gt_poses[i][1])
            if d > dist_thresh:
                continue
            if est_times[j] - last_emit_t < min_spacing:
                continue
            closures.append((i, j, d))
            last_emit_t = est_times[j]
            break
    return closures


def measure_loop_closure(gt_poses, i, j, noise_pos, noise_rot, rng,
                         false_p=0.0, false_min_idx_gap=20):
    """Simulate a successful ICP between scans i and j using GT relative
    pose plus small noise. Returns (R_rel, t_rel).

    When false_p > 0, with that probability replace the target pose j with a
    random pose at least `false_min_idx_gap` indices away — simulates a false
    data association from scan-context-style place recognition.
    """
    if false_p > 0 and rng.random() < false_p:
        n = len(gt_poses)
        for _ in range(20):
            wrong_j = int(rng.integers(0, n))
            if abs(wrong_j - j) >= false_min_idx_gap and abs(wrong_j - i) >= false_min_idx_gap:
                break
        Rj, pj = gt_poses[wrong_j]
    else:
        Rj, pj = gt_poses[j]
    Ri, pi = gt_poses[i]
    R_rel, t_rel = relative_pose(Ri, pi, Rj, pj)
    t_rel = t_rel + rng.normal(0, noise_pos, 3)
    R_rel = R_rel @ rodrigues_exp(rng.normal(0, noise_rot, 3))
    return R_rel, t_rel


def pose_graph_optimize(init_poses, odom_edges, lc_factors, n_iters=50,
                        prior_strength=1e6, lm_init=1e-3, lm_max=1e6, verbose=False):
    """Levenberg-Marquardt optimization on SE(3) pose graph.

    Right-perturbation convention: state update is
      R_k <- R_k @ Exp(dθ_k)
      p_k <- p_k + R_k @ dp_k    (dp_k is in body frame of pose k)

    init_poses: list of (R, p) initial estimates
    odom_edges: list of (i, j, P_ij). Measurement = relative pose computed
        from init_poses (so initial residual is zero).
    lc_factors: list of (i, j, R_meas, t_meas, P_ij).
    """
    n = len(init_poses)
    poses_R = [R.copy() for R, _ in init_poses]
    poses_p = [p.copy() for _, p in init_poses]

    odom_meas = []
    for i, j, P_ij in odom_edges:
        Ri, pi = init_poses[i]
        Rj, pj = init_poses[j]
        R_rel, t_rel = relative_pose(Ri, pi, Rj, pj)
        Omega = np.linalg.inv(P_ij + np.eye(6) * 1e-12)
        odom_meas.append((i, j, R_rel, t_rel, Omega))

    lc_meas = []
    for (i, j, R_rel, t_rel, P_ij) in lc_factors:
        Omega = np.linalg.inv(P_ij + np.eye(6) * 1e-12)
        lc_meas.append((i, j, R_rel, t_rel, Omega))

    def total_cost_at(Rs, ps):
        c = 0.0
        for batches in (odom_meas, lc_meas):
            for mi, mj, R_meas, t_meas, Omega in batches:
                Ri, pi = Rs[mi], ps[mi]
                Rj, pj = Rs[mj], ps[mj]
                R_est = Ri.T @ Rj
                t_est = Ri.T @ (pj - pi)
                e_rot = rodrigues_log(R_meas.T @ R_est)
                e_trans = t_est - t_meas
                e = np.concatenate([e_rot, e_trans])
                c += float(e @ Omega @ e)
        return c

    lm_lambda = lm_init
    last_cost = total_cost_at(poses_R, poses_p)

    for it in range(n_iters):
        ndof = 6 * n
        H = np.zeros((ndof, ndof))
        b = np.zeros(ndof)
        H[0:6, 0:6] += np.eye(6) * prior_strength

        for batches in (odom_meas, lc_meas):
            for mi, mj, R_meas, t_meas, Omega in batches:
                Ri, pi = poses_R[mi], poses_p[mi]
                Rj, pj = poses_R[mj], poses_p[mj]

                R_est = Ri.T @ Rj
                t_est = Ri.T @ (pj - pi)

                e_rot = rodrigues_log(R_meas.T @ R_est)
                e_trans = t_est - t_meas
                e = np.concatenate([e_rot, e_trans])

                Ji = np.zeros((6, 6))
                Jj = np.zeros((6, 6))
                Ji[0:3, 0:3] = -np.eye(3)
                Jj[0:3, 0:3] = np.eye(3)
                Ji[3:6, 0:3] = skew(Ri.T @ (pj - pi))
                Ji[3:6, 3:6] = -np.eye(3)
                Jj[3:6, 3:6] = Ri.T @ Rj

                bi, bj = 6 * mi, 6 * mj
                H[bi:bi+6, bi:bi+6] += Ji.T @ Omega @ Ji
                H[bi:bi+6, bj:bj+6] += Ji.T @ Omega @ Jj
                H[bj:bj+6, bi:bi+6] += Jj.T @ Omega @ Ji
                H[bj:bj+6, bj:bj+6] += Jj.T @ Omega @ Jj
                b[bi:bi+6] += Ji.T @ Omega @ e
                b[bj:bj+6] += Jj.T @ Omega @ e

        accepted = False
        while not accepted and lm_lambda < lm_max:
            try:
                H_lm = H + lm_lambda * np.diag(np.diag(H) + 1e-12)
                dx = np.linalg.solve(H_lm, -b)
            except np.linalg.LinAlgError:
                lm_lambda *= 10
                continue

            new_R = [R.copy() for R in poses_R]
            new_p = [p.copy() for p in poses_p]
            for k in range(n):
                base = 6 * k
                dtheta = dx[base:base+3]
                dp = dx[base+3:base+6]
                new_R[k] = new_R[k] @ rodrigues_exp(dtheta)
                new_p[k] = new_p[k] + new_R[k] @ dp

            new_cost = total_cost_at(new_R, new_p)
            if np.isfinite(new_cost) and new_cost < last_cost:
                poses_R = new_R
                poses_p = new_p
                lm_lambda = max(lm_lambda / 3.0, 1e-9)
                if verbose:
                    print(f"    iter {it:3d}: cost {last_cost:.4e} -> {new_cost:.4e}, "
                          f"|dx|={np.linalg.norm(dx):.3e}, λ={lm_lambda:.2e}")
                last_cost = new_cost
                accepted = True
                if np.linalg.norm(dx) < 1e-7:
                    return list(zip(poses_R, poses_p))
            else:
                lm_lambda *= 10
        if not accepted:
            if verbose:
                print(f"    iter {it:3d}: LM failed (λ={lm_lambda:.2e}), stopping")
            break

    return list(zip(poses_R, poses_p))


def compute_ate(poses, gt_poses):
    return np.array([
        np.linalg.norm(p - pg) for (_, p), (_, pg) in zip(poses, gt_poses)
    ])


if __name__ == "__main__":
    print("Loading simulated data...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

    print(f"\nRunning IESKF (cube_len={CUBE_LEN:.1f} m)...")
    np.random.seed(123)
    (est_poses, est_times, gt_poses, _, _, _, _,
     _, _, _, _, _, _) = run_ieskf_no_lc(imu_data, lidar_data)
    n = len(est_poses)
    print(f"  {n} keyframe poses")

    ate_no_lc = compute_ate(est_poses, gt_poses)
    print(f"  ATE no-LC: mean={ate_no_lc.mean():.4f} m, max={ate_no_lc.max():.4f} m, "
          f"final={ate_no_lc[-1]:.4f} m")

    print(f"\nDetecting loop closures (dist≤{LC_DIST_THRESH} m, "
          f"gap≥{LC_MIN_TIME_GAP} s, spacing≥{LC_MIN_SPACING} s)...")
    closures = detect_loop_closures(gt_poses, est_times,
                                    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
    print(f"  Found {len(closures)} loop closures:")
    for (i, j, d) in closures:
        print(f"    pose {i:>4} (t={est_times[i]:6.2f}s) ↔ pose {j:>4} "
              f"(t={est_times[j]:6.2f}s)  GT-dist={d:.3f} m")

    print("\nBuilding fixed-weight pose graph...")
    fixed_odom_cov = np.diag([
        FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2, FIXED_ODOM_ROT_STD ** 2,
        FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2, FIXED_ODOM_POS_STD ** 2,
    ])
    fixed_lc_cov = np.diag([
        FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2, FIXED_LC_ROT_STD ** 2,
        FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2, FIXED_LC_POS_STD ** 2,
    ])
    print(f"  fixed odom σ: pos={FIXED_ODOM_POS_STD} m, rot={FIXED_ODOM_ROT_STD} rad")
    print(f"  fixed LC   σ: pos={FIXED_LC_POS_STD} m,   rot={FIXED_LC_ROT_STD} rad")

    odom_edges = build_odom_edges(est_poses, fixed_odom_cov)

    rng = np.random.default_rng(99)
    lc_factors = []
    for (i, j, _) in closures:
        R_rel, t_rel = measure_loop_closure(gt_poses, i, j,
                                            LC_NOISE_POS, LC_NOISE_ROT, rng)
        lc_factors.append((i, j, R_rel, t_rel, fixed_lc_cov))

    print("\nOptimizing pose graph (Levenberg-Marquardt)...")
    opt_poses = pose_graph_optimize(est_poses, odom_edges, lc_factors, verbose=True)

    ate_lc = compute_ate(opt_poses, gt_poses)
    print(f"  ATE fixed-LC: mean={ate_lc.mean():.4f} m, "
          f"max={ate_lc.max():.4f} m, final={ate_lc[-1]:.4f} m")

    improvement_mean = (1.0 - ate_lc.mean() / ate_no_lc.mean()) * 100
    improvement_max = (1.0 - ate_lc.max() / ate_no_lc.max()) * 100
    improvement_final = (1.0 - ate_lc[-1] / ate_no_lc[-1]) * 100
    print(f"\nImprovement vs no-LC:")
    print(f"  mean ATE:  {improvement_mean:+.1f}%")
    print(f"  max ATE:   {improvement_max:+.1f}%")
    print(f"  final ATE: {improvement_final:+.1f}%")

    t_arr = np.array(est_times)
    gt_p = np.array([p for _, p in gt_poses])
    est_p = np.array([p for _, p in est_poses])
    opt_p = np.array([p for _, p in opt_poses])

    lap_time = SQ_PERIM / SQ_SPEED
    lap_idx_boundaries = [k * lap_time for k in range(1, SQ_N_LAPS + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    plot_walls(ax)
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="ground truth")
    ax.plot(est_p[:, 0], est_p[:, 1], "r-", lw=1.0, alpha=0.7, label="no-LC IESKF")
    ax.plot(opt_p[:, 0], opt_p[:, 1], "b-", lw=1.2, alpha=0.9, label="fixed-LC pose graph")
    for (i, j, _) in closures:
        ax.plot([gt_p[i, 0], gt_p[j, 0]], [gt_p[i, 1], gt_p[j, 1]],
                "g--", lw=0.8, alpha=0.7)
    ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=8, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory ({SQ_N_LAPS} laps, {len(closures)} LC)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_arr, ate_no_lc, "r-", lw=1.2, label="no LC")
    ax.plot(t_arr, ate_lc, "b-", lw=1.2, label="fixed-weight LC")
    for (_, j, _) in closures:
        ax.axvline(est_times[j], color="g", ls=":", lw=0.7, alpha=0.6)
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title("ATE vs time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ["no LC", "fixed-LC"]
    mean_vals = [ate_no_lc.mean(), ate_lc.mean()]
    max_vals = [ate_no_lc.max(), ate_lc.max()]
    final_vals = [ate_no_lc[-1], ate_lc[-1]]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, mean_vals, w, label="mean", color=["#cc4444", "#4444cc"])
    ax.bar(x, max_vals, w, label="max", color=["#ff8888", "#8888ff"])
    ax.bar(x + w, final_vals, w, label="final", color=["#bb6666", "#6666bb"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("position error (m)")
    ax.set_title("ATE summary")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_square_corridor_fixed_lc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")

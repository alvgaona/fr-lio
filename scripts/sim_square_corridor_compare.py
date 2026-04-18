"""Square corridor multi-lap: compare loop closure configurations.

Runs the IESKF once (drifted multi-lap baseline with sliding cube), detects loop
closures by GT proximity, and runs the same Levenberg-Marquardt pose graph
optimizer with four different edge weighting schemes:

A. No LC                 (IESKF only)
B. Fixed-weight LC       (LIO-SAM style: constant odom + LC covariance)
C. CRLB-weighted LC      (per-step scan-to-scan CRLB, no bias-walk)
D. CRLB+bias-walk LC     (per-step CRLB + bias-walk floor; the calibrated cov)

Reports a unified ATE table and saves a 4-row trajectory + ATE comparison plot.

Run:
    SIM_ENV=square_corridor SIM_CUBE_LEN=4.0 SIM_PERSIST=0 \\
      SIM_BIAS_WALK_POS=6e-4 SIM_BIAS_WALK_ROT=6e-5 \\
      pixi run python -u sim_square_corridor_compare.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import (
    trajectory_at, SQ_PERIM, SQ_SPEED, SQ_N_LAPS,
)
from sim_iekf_3d import rodrigues_exp
from sim_square_corridor_no_lc import (
    run_ieskf_no_lc, plot_walls, OUTER, INNER_LO, INNER_HI, CUBE_LEN,
    BIAS_WALK_POS, BIAS_WALK_ROT,
)
from sim_square_corridor_fixed_lc import (
    detect_loop_closures, measure_loop_closure, build_odom_edges,
    pose_graph_optimize, compute_ate, relative_pose,
    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING,
    LC_NOISE_POS, LC_NOISE_ROT,
    FIXED_ODOM_POS_STD, FIXED_ODOM_ROT_STD,
    FIXED_LC_POS_STD, FIXED_LC_ROT_STD,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)


def edge_cov_with_floor(edge_cov, fallback_cov):
    if edge_cov is None:
        return fallback_cov
    return edge_cov


def rescale_to_trace(cov, target_pos_trace, target_rot_trace):
    """Scale rotation and position blocks of cov so each block trace matches
    the target. Preserves anisotropy and rot-pos coupling within each block.
    """
    out = cov.copy()
    rot_trace = float(np.trace(out[0:3, 0:3]))
    pos_trace = float(np.trace(out[3:6, 3:6]))
    s_rot = (target_rot_trace / rot_trace) if rot_trace > 1e-15 else 1.0
    s_pos = (target_pos_trace / pos_trace) if pos_trace > 1e-15 else 1.0
    out[0:3, 0:3] *= s_rot
    out[3:6, 3:6] *= s_pos
    s_off = np.sqrt(s_rot * s_pos)
    out[0:3, 3:6] *= s_off
    out[3:6, 0:3] *= s_off
    return out


GYR_COV_CONT = 0.1
ACC_COV_CONT = 0.1


def imu_process_floor(dt_edge):
    """Per-edge process noise floor from IMU continuous-time noise model.

    Position: integrating acc white noise twice over dt_edge ->
        sigma_p^2 (per axis) = (1/3) * acc_cov * dt_edge^3
    Rotation: integrating gyro white noise once over dt_edge ->
        sigma_theta^2 (per axis) = gyr_cov * dt_edge

    Returns 6x6 cov in [pos; rot] order to match P_rel layout.
    """
    sigma_p2 = ACC_COV_CONT * (dt_edge ** 3) / 3.0
    sigma_r2 = GYR_COV_CONT * dt_edge
    floor = np.zeros((6, 6))
    floor[0:3, 0:3] = np.eye(3) * sigma_p2
    floor[3:6, 3:6] = np.eye(3) * sigma_r2
    return floor


def build_crlb_edges(edge_covs_per_step, fallback_cov, strip_bias_walk=False,
                     bias_walk_pos=0.0, bias_walk_rot=0.0,
                     dt_per_step=0.1, rescale_target=None,
                     extra_bias_walk_pos=0.0, extra_bias_walk_rot=0.0,
                     add_imu_floor=False, add_imu_floor_pos_only=False):
    """Build odometry edges from per-step CRLB covariances.

    edge_covs_per_step[k] is the world-frame relative covariance for the
    transition from keyframe k-1 to keyframe k (or None if registration
    was skipped).

    If strip_bias_walk=True, subtract the bias-walk floor that was added
    inside the runner — useful to isolate the pure CRLB contribution.
    """
    n = len(edge_covs_per_step)
    edges = []
    for k in range(n - 1):
        cov = edge_covs_per_step[k + 1]
        if cov is None:
            edges.append((k, k + 1, fallback_cov))
            continue
        cov_use = cov.copy()
        if strip_bias_walk:
            cov_use[0:3, 0:3] -= np.eye(3) * bias_walk_pos * dt_per_step
            cov_use[3:6, 3:6] -= np.eye(3) * bias_walk_rot * dt_per_step
        if rescale_target is not None:
            cov_use = rescale_to_trace(cov_use,
                                       rescale_target["pos"],
                                       rescale_target["rot"])
        if extra_bias_walk_pos > 0:
            cov_use[3:6, 3:6] += np.eye(3) * extra_bias_walk_pos * dt_per_step
        if extra_bias_walk_rot > 0:
            cov_use[0:3, 0:3] += np.eye(3) * extra_bias_walk_rot * dt_per_step
        if add_imu_floor:
            cov_use += imu_process_floor(dt_per_step)
        if add_imu_floor_pos_only:
            sigma_p2 = ACC_COV_CONT * (dt_per_step ** 3) / 3.0
            cov_use[0:3, 0:3] += np.eye(3) * sigma_p2
        eigvals = np.linalg.eigvalsh(cov_use)
        if np.min(eigvals) <= 0:
            cov_use = cov_use + np.eye(6) * (abs(np.min(eigvals)) + 1e-9)
        edges.append((k, k + 1, cov_use))
    return edges


if __name__ == "__main__":
    print("Loading simulated data...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")
    print(f"  bias-walk: pos={BIAS_WALK_POS} m^2/s, rot={BIAS_WALK_ROT} rad^2/s")

    print(f"\nRunning IESKF once (cube_len={CUBE_LEN:.1f} m)...")
    np.random.seed(123)
    (est_poses, est_times, gt_poses, _, _, _, _,
     _, _, _, edge_covs_per_step, _) = run_ieskf_no_lc(imu_data, lidar_data,
                                                      use_perpoint_cov=False)
    n = len(est_poses)
    print(f"  {n} keyframe poses, "
          f"{sum(1 for c in edge_covs_per_step if c is not None)} valid CRLB edges")

    print(f"\nRunning IESKF with per-point covariance (for Config I)...")
    np.random.seed(123)
    (est_poses_pp, est_times_pp, gt_poses_pp, _, _, _, _,
     _, _, _, edge_covs_pp, _) = run_ieskf_no_lc(imu_data, lidar_data,
                                                use_perpoint_cov=True)
    print(f"  {len(est_poses_pp)} keyframe poses (per-point)")

    ate_no_lc = compute_ate(est_poses, gt_poses)
    rmse_no_lc = float(np.sqrt(np.mean(ate_no_lc ** 2)))
    print(f"  No-LC ATE: RMSE={rmse_no_lc:.4f} m  mean={ate_no_lc.mean():.4f} m  "
          f"max={ate_no_lc.max():.4f} m  final={ate_no_lc[-1]:.4f} m")

    print(f"\nDetecting loop closures (dist≤{LC_DIST_THRESH} m, "
          f"gap≥{LC_MIN_TIME_GAP} s, spacing≥{LC_MIN_SPACING} s)...")
    closures = detect_loop_closures(gt_poses, est_times,
                                    LC_DIST_THRESH, LC_MIN_TIME_GAP, LC_MIN_SPACING)
    print(f"  Found {len(closures)} loop closures")

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
    odom_edges_crlb_imu = build_crlb_edges(
        edge_covs_per_step, fallback_cov,
        strip_bias_walk=True,
        bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
        dt_per_step=1.0 / 10.0,
        add_imu_floor=True)
    odom_edges_crlb_imu_pos = build_crlb_edges(
        edge_covs_per_step, fallback_cov,
        strip_bias_walk=True,
        bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
        dt_per_step=1.0 / 10.0,
        add_imu_floor_pos_only=True)


    def edge_summary(label, edges):
        pos_traces = [np.trace(e[2][3:6, 3:6]) for e in edges]
        rot_traces = [np.trace(e[2][0:3, 0:3]) for e in edges]
        print(f"  {label:<30}  pos median trace={np.median(pos_traces):.3e} "
              f"m^2  rot median trace={np.median(rot_traces):.3e} rad^2")


    print("\nPer-edge covariance stats (smaller = stiffer odom edge):")
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
    lc_factors_crlb = lc_factors_fixed
    lc_factors_crlb_bw = lc_factors_fixed


    def run_and_eval(label, odom, lc_factors):
        print(f"\n[{label}]  optimizing...")
        poses = pose_graph_optimize(est_poses, odom, lc_factors)
        ate = compute_ate(poses, gt_poses)
        rmse = float(np.sqrt(np.mean(ate ** 2)))
        print(f"  ATE: RMSE={rmse:.4f} m  mean={ate.mean():.4f} m  "
              f"max={ate.max():.4f} m  final={ate[-1]:.4f} m")
        return poses, ate


    configs = []
    configs.append(("A: No LC", est_poses, ate_no_lc))

    poses_B, ate_B = run_and_eval("B: Fixed-weight LC", odom_edges_fixed, lc_factors_fixed)
    configs.append(("B: Fixed-weight LC", poses_B, ate_B))

    poses_C, ate_C = run_and_eval("C: CRLB LC (no bias-walk)", odom_edges_crlb, lc_factors_crlb)
    configs.append(("C: CRLB LC", poses_C, ate_C))

    poses_D, ate_D = run_and_eval("D: CRLB + bias-walk LC", odom_edges_crlb_bw, lc_factors_crlb_bw)
    configs.append(("D: CRLB+bias-walk LC", poses_D, ate_D))

    poses_E, ate_E = run_and_eval("E: CRLB shape, fixed scale", odom_edges_crlb_shape, lc_factors_fixed)
    configs.append(("E: CRLB shape (fixed scale)", poses_E, ate_E))

    poses_F1, ate_F1 = run_and_eval("F1: CRLB + 10x bias-walk", odom_edges_crlb_bw_x10, lc_factors_fixed)
    configs.append(("F1: CRLB + 10x bias-walk", poses_F1, ate_F1))

    poses_F2, ate_F2 = run_and_eval("F2: CRLB + 100x bias-walk", odom_edges_crlb_bw_x100, lc_factors_fixed)
    configs.append(("F2: CRLB + 100x bias-walk", poses_F2, ate_F2))

    poses_G, ate_G = run_and_eval("G: CRLB + IMU process floor", odom_edges_crlb_imu, lc_factors_fixed)
    configs.append(("G: CRLB + IMU floor", poses_G, ate_G))

    poses_H, ate_H = run_and_eval("H: CRLB + IMU floor pos only", odom_edges_crlb_imu_pos, lc_factors_fixed)
    configs.append(("H: CRLB + IMU floor (pos)", poses_H, ate_H))

    odom_edges_I = build_odom_edges(est_poses_pp, fixed_odom_cov)
    odom_edges_I_G = build_crlb_edges(
        edge_covs_pp, fallback_cov,
        strip_bias_walk=True,
        bias_walk_pos=BIAS_WALK_POS, bias_walk_rot=BIAS_WALK_ROT,
        dt_per_step=1.0 / 10.0,
        add_imu_floor=True)

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
    print("SUMMARY (lower is better)")
    print("=" * 70)
    print(f"  {'Config':<28} {'RMSE':>10} {'mean':>10} {'max':>10} {'final':>11}")
    for label, _, ate in configs:
        r = float(np.sqrt(np.mean(ate ** 2)))
        print(f"  {label:<28} {r:>10.4f} {ate.mean():>10.4f} {ate.max():>10.4f} {ate[-1]:>11.4f}")

    base = float(np.sqrt(np.mean(ate_no_lc ** 2)))
    print(f"\nRMSE-ATE improvement vs no-LC:")
    for label, _, ate in configs[1:]:
        r = float(np.sqrt(np.mean(ate ** 2)))
        pct = (1 - r / base) * 100
        print(f"  {label:<28} {pct:+.1f}%")

    t_arr = np.array(est_times)
    gt_p = np.array([p for _, p in gt_poses])

    lap_time = SQ_PERIM / SQ_SPEED
    lap_idx = [k * lap_time for k in range(1, SQ_N_LAPS + 1)]

    fig = plt.figure(figsize=(20, 14))
    colors = ["r", "m", "b", "g", "c", "#aa6600", "#660066", "#0099aa",
              "#cc0099", "#ff8800", "#008844"]

    ax = plt.subplot(3, 3, 1)
    plot_walls(ax)
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="GT")
    for (label, poses, _), c in zip(configs, colors):
        p = np.array([pp for _, pp in poses])
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.0, alpha=0.85, label=label)
    ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectories ({SQ_N_LAPS} laps, {len(closures)} LC)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = plt.subplot(3, 3, 2)
    for (label, _, ate), c in zip(configs, colors):
        ax.plot(t_arr, ate, "-", color=c, lw=1.1, alpha=0.85, label=label)
    for (_, j, _) in closures:
        ax.axvline(est_times[j], color="g", ls=":", lw=0.5, alpha=0.4)
    for tb in lap_idx:
        ax.axvline(tb, color="gray", ls="--", lw=0.7, alpha=0.4)
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

    # 7 small per-config trajectory subplots in remaining grid (rows 2-3 = 6 slots)
    # We have 7 configs so use 4 per row x 2 rows = 8 slots, last empty
    n_rows = 4
    n_cols = 4
    fig.set_size_inches(20, 16)
    for idx, (label, poses, _) in enumerate(configs):
        if 5 + idx > n_rows * n_cols:
            break
        ax = plt.subplot(n_rows, n_cols, 5 + idx)
        plot_walls(ax)
        ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.2, label="GT")
        p = np.array([pp for _, pp in poses])
        ax.plot(p[:, 0], p[:, 1], "-", color=colors[idx], lw=1.0, alpha=0.9, label=label)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=8)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_square_corridor_compare.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")

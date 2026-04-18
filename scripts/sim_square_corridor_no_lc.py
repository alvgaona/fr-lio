"""Square corridor multi-lap drift baseline (no loop closure).

Runs the IESKF with scan-to-map updates over the square corridor environment
for several laps and visualizes how drift accumulates relative to ground truth.
This is the baseline for the loop closure paper's square_corridor experiment.

Run:
    SIM_ENV=square_corridor pixi run python sim_square_corridor_no_lc.py

Assumes sim_imu_trajectory.py has already been run with SIM_ENV=square_corridor
to produce sim_imu.npy and sim_lidar.npy.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import (
    INITIAL_GYR_BIAS, INITIAL_ACC_BIAS, trajectory_at,
    SQ_HALF, SQ_CORNER_R, SQ_CENTER, SQ_PERIM, SQ_SPEED, SQ_N_LAPS,
)
from collections import deque
from sim_iekf_3d import (
    State, predict, iekf_update, OnlineMap,
    DT_IMU, NN_K, MAP_VOXEL_SIZE, rodrigues_log,
    compute_scan_to_scan_covariance,
    S2S_MIN_TRANS_M, S2S_MIN_ROT_RAD, S2S_MIN_VALID_POINTS,
    S2S_PERSIST_ALPHA, S2S_PERSIST_NORMAL_TAU,
    S2S_PERSIST_DIST_EPS, S2S_PERSIST_HISTORY,
)
from scipy.spatial import cKDTree

USE_PERSIST = os.environ.get("SIM_PERSIST", "1") == "1"
BIAS_WALK_POS = float(os.environ.get("SIM_BIAS_WALK_POS", "0.0"))
BIAS_WALK_ROT = float(os.environ.get("SIM_BIAS_WALK_ROT", "0.0"))
USE_PERPOINT_COV = os.environ.get("SIM_PERPOINT_COV", "0") == "1"
# Option 1 (Schmidt-Kalman style): use filter's own P[bg], P[ba] to derive a
# per-edge cov floor. Captures bias-state uncertainty propagation into pose
# space without empirical tuning. Per-edge variance:
#   floor_pos_axis = sigma_ba_axis^2 * dt^2
#   floor_rot_axis = sigma_bg_axis^2 * dt^2
USE_FILTER_BIAS_FLOOR = os.environ.get("SIM_FILTER_BIAS_FLOOR", "0") == "1"

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)

OUTER = 10.0
INNER_LO = 2.0
INNER_HI = 8.0

CUBE_LEN = float(os.environ.get("SIM_CUBE_LEN", "4.0"))
CUBE_REPRUNE_TRIGGER = 0.4 * CUBE_LEN


def prune_to_cube(omap, center, half_len):
    if len(omap.points) == 0:
        return
    pts = np.array(omap.points)
    mask = np.all(np.abs(pts - center) <= half_len, axis=1)
    if mask.all():
        return
    src = omap.source_idx
    omap.points = [p for p in pts[mask]]
    omap.source_idx = [src[i] for i in range(len(src)) if mask[i]]
    omap.voxel_set = set(
        tuple(np.round(p / MAP_VOXEL_SIZE).astype(int)) for p in omap.points
    )
    omap._dirty = True


def run_ieskf_no_lc(imu_data, lidar_data, use_perpoint_cov=None):
    if use_perpoint_cov is None:
        use_perpoint_cov = USE_PERPOINT_COV
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = State(R0, p0.copy(), v0.copy(),
                  INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2
    omap = OnlineMap()
    cube_center = state.p.copy()
    map_size_history = []

    P_drift = np.zeros((6, 6))
    prev_scan_pts = None
    prev_tree = None
    R_prev_s2s = None
    p_prev_s2s = None
    plane_history = deque(maxlen=S2S_PERSIST_HISTORY)
    persist_history = []
    last_t = 0.0
    edge_covs = []
    pending_edge_cov = None

    est_poses = []
    est_times = []
    gt_poses = []
    cov_pos = []
    cov_rot = []
    cov_pos_inflated = []
    cov_rot_inflated = []
    imu_est_p = []
    imu_t = []

    li = 0
    for sample in imu_data:
        t = sample["t"]
        state, P = predict(state, P, sample["gyro"], sample["acc"], DT_IMU)
        imu_est_p.append(state.p.copy())
        imu_t.append(t)

        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                if len(omap) >= NN_K:
                    state, P = iekf_update(state, P, scan["points_body"], omap,
                                            use_fej=False,
                                            use_perpoint_cov=use_perpoint_cov)

                pending_edge_cov = None
                if prev_tree is not None:
                    tg = np.linalg.norm(state.p - p_prev_s2s)
                    rg = np.linalg.norm(rodrigues_log(R_prev_s2s.T @ state.R))
                    if tg > S2S_MIN_TRANS_M or rg > S2S_MIN_ROT_RAD:
                        P_rel, dbg = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan_pts, prev_tree, R_prev_s2s, p_prev_s2s,
                            use_perpoint_cov=use_perpoint_cov)
                        if dbg.get("valid_count", 0) >= S2S_MIN_VALID_POINTS:
                            curr_planes = dbg.get("planes_world", [])
                            persist_frac = 0.0
                            if USE_PERSIST and plane_history and curr_planes:
                                hits = 0
                                for pc in curr_planes:
                                    nc = pc[0:3]
                                    dc = pc[3]
                                    matched = False
                                    for old_set in plane_history:
                                        for pp in old_set:
                                            dot = float(nc @ pp[0:3])
                                            if abs(dot) < S2S_PERSIST_NORMAL_TAU:
                                                continue
                                            dp_signed = pp[3] if dot >= 0 else -pp[3]
                                            if abs(dc - dp_signed) < S2S_PERSIST_DIST_EPS:
                                                matched = True
                                                break
                                        if matched:
                                            break
                                    if matched:
                                        hits += 1
                                persist_frac = hits / len(curr_planes)
                            persist_scale = max(0.0, 1.0 - S2S_PERSIST_ALPHA * persist_frac)
                            Adj = np.zeros((6, 6))
                            Adj[0:3, 0:3] = R_prev_s2s
                            Adj[3:6, 3:6] = R_prev_s2s
                            P_rel_w = persist_scale * Adj @ P_rel @ Adj.T
                            P_drift += P_rel_w
                            pending_edge_cov = P_rel_w
                            persist_history.append(persist_frac)
                            if curr_planes:
                                plane_history.append(curr_planes)

                dt_step = scan["t"] - last_t
                bias_step = np.zeros((6, 6))
                if dt_step > 0:
                    if BIAS_WALK_POS > 0 or BIAS_WALK_ROT > 0:
                        bias_step[0:3, 0:3] += np.eye(3) * BIAS_WALK_POS * dt_step
                        bias_step[3:6, 3:6] += np.eye(3) * BIAS_WALK_ROT * dt_step
                    if USE_FILTER_BIAS_FLOOR:
                        # Filter state layout: [rot(0:3), pos(3:6), vel(6:9),
                        # bg(9:12), ba(12:15)]. Project bias-state uncertainty
                        # into pose space per edge:
                        #   sigma_pos_axis^2 = sigma_ba_axis^2 * dt^2
                        #   sigma_rot_axis^2 = sigma_bg_axis^2 * dt^2
                        sigma_ba2 = np.diag(P[12:15, 12:15])
                        sigma_bg2 = np.diag(P[9:12, 9:12])
                        bias_step[0:3, 0:3] += np.diag(sigma_ba2 * dt_step ** 2)
                        bias_step[3:6, 3:6] += np.diag(sigma_bg2 * dt_step ** 2)
                    if np.any(bias_step):
                        P_drift += bias_step
                if pending_edge_cov is not None:
                    edge_covs.append(pending_edge_cov + bias_step)
                else:
                    edge_covs.append(None if bias_step.sum() == 0 else bias_step.copy())
                last_t = scan["t"]

                wp = (state.R @ scan["points_body"].T).T + state.p
                # Source-pose index = current keyframe index (= len(est_poses)
                # before the append below). This tags every map point with the
                # keyframe whose pose was used at insertion time, enabling
                # post-LC map correction via `omap.correct_map(...)`.
                omap.add_points(wp, source_idx=len(est_poses))

                if np.max(np.abs(state.p - cube_center)) > CUBE_REPRUNE_TRIGGER:
                    cube_center = state.p.copy()
                    prune_to_cube(omap, cube_center, CUBE_LEN / 2.0)

                Adj_curT = np.zeros((6, 6))
                Adj_curT[0:3, 0:3] = state.R.T
                Adj_curT[3:6, 3:6] = state.R.T
                P_drift_body = Adj_curT @ P_drift @ Adj_curT.T
                P_pub = P.copy()
                P_pub[3:6, 3:6] += P_drift_body[0:3, 0:3]
                P_pub[0:3, 0:3] += P_drift_body[3:6, 3:6]

                est_poses.append((state.R.copy(), state.p.copy()))
                est_times.append(scan["t"])
                cov_rot.append(P[0:3, 0:3].copy())
                cov_pos.append(P[3:6, 3:6].copy())
                cov_rot_inflated.append(P_pub[0:3, 0:3].copy())
                cov_pos_inflated.append(P_pub[3:6, 3:6].copy())
                map_size_history.append(len(omap))
                R_gt, p_gt, _, _, _ = trajectory_at(scan["t"])
                gt_poses.append((R_gt.copy(), p_gt.copy()))

                prev_scan_pts = scan["points_body"].copy()
                prev_tree = cKDTree(prev_scan_pts)
                R_prev_s2s = state.R.copy()
                p_prev_s2s = state.p.copy()
            li += 1

    return (est_poses, est_times, gt_poses,
            cov_pos, cov_rot, cov_pos_inflated, cov_rot_inflated,
            np.array(imu_est_p), np.array(imu_t), np.array(map_size_history),
            edge_covs, omap)


def compute_nees(est_poses, gt_poses, cov_pos, cov_rot):
    nees_pos = np.zeros(len(est_poses))
    nees_rot = np.zeros(len(est_poses))
    for k, ((Re, pe), (Rg, pg)) in enumerate(zip(est_poses, gt_poses)):
        e_p = pe - pg
        e_th = rodrigues_log(Rg.T @ Re)
        nees_pos[k] = e_p @ np.linalg.solve(cov_pos[k], e_p)
        nees_rot[k] = e_th @ np.linalg.solve(cov_rot[k], e_th)
    return nees_pos, nees_rot


def position_error(est_poses, gt_poses):
    errs = []
    for (_, pe), (_, pg) in zip(est_poses, gt_poses):
        errs.append(np.linalg.norm(pe - pg))
    return np.array(errs)


def yaw_from_R(R):
    return np.arctan2(R[1, 0], R[0, 0])


def yaw_error(est_poses, gt_poses):
    errs = []
    for (Re, _), (Rg, _) in zip(est_poses, gt_poses):
        de = yaw_from_R(Re) - yaw_from_R(Rg)
        de = (de + np.pi) % (2 * np.pi) - np.pi
        errs.append(abs(de))
    return np.array(errs)


def plot_walls(ax):
    ax.plot([0, OUTER, OUTER, 0, 0], [0, 0, OUTER, OUTER, 0],
            color="#444444", lw=1.5)
    ax.plot([INNER_LO, INNER_HI, INNER_HI, INNER_LO, INNER_LO],
            [INNER_LO, INNER_LO, INNER_HI, INNER_HI, INNER_LO],
            color="#444444", lw=1.5)


if __name__ == "__main__":
    print("Loading simulated data...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")
    print(f"  perimeter={SQ_PERIM:.2f} m, speed={SQ_SPEED:.1f} m/s, "
          f"laps={SQ_N_LAPS}, total≈{SQ_PERIM*SQ_N_LAPS/SQ_SPEED:.1f} s")

    print(f"\nRunning IESKF (no loop closure, cube_len={CUBE_LEN:.1f} m)...")
    np.random.seed(123)
    (est_poses, est_times, gt_poses, cov_pos, cov_rot,
     cov_pos_inf, cov_rot_inf,
     imu_p, imu_t, map_sizes, _edge_covs, _omap) = run_ieskf_no_lc(imu_data, lidar_data)
    print(f"  {len(est_poses)} keyframe poses, "
          f"map size: max={map_sizes.max()}, final={map_sizes[-1]}")

    err_pos = position_error(est_poses, gt_poses)
    err_yaw = yaw_error(est_poses, gt_poses)
    nees_pos, nees_rot = compute_nees(est_poses, gt_poses, cov_pos, cov_rot)
    nees_pos_inf, nees_rot_inf = compute_nees(est_poses, gt_poses, cov_pos_inf, cov_rot_inf)
    print(f"  position error: mean={err_pos.mean():.4f} m, "
          f"max={err_pos.max():.4f} m, final={err_pos[-1]:.4f} m")
    print(f"  yaw error: mean={np.degrees(err_yaw.mean()):.3f} deg, "
          f"max={np.degrees(err_yaw.max()):.3f} deg, "
          f"final={np.degrees(err_yaw[-1]):.3f} deg")
    print(f"  NEES pos (filter only):    mean={nees_pos.mean():>12.2f} median={np.median(nees_pos):>12.2f}")
    print(f"  NEES pos (CRLB-inflated):  mean={nees_pos_inf.mean():>12.2f} median={np.median(nees_pos_inf):>12.2f}")
    print(f"  NEES rot (filter only):    mean={nees_rot.mean():>12.2f} median={np.median(nees_rot):>12.2f}")
    print(f"  NEES rot (CRLB-inflated):  mean={nees_rot_inf.mean():>12.2f} median={np.median(nees_rot_inf):>12.2f}")
    print("  (expected ~3 for both pos and rot; >>3 = overconfident)")
    filter_pos_std = np.sqrt(np.array([np.trace(c)/3 for c in cov_pos]))
    filter_rot_std = np.sqrt(np.array([np.trace(c)/3 for c in cov_rot]))
    inflated_pos_std = np.sqrt(np.array([np.trace(c)/3 for c in cov_pos_inf]))
    inflated_rot_std = np.sqrt(np.array([np.trace(c)/3 for c in cov_rot_inf]))

    t_arr = np.array(est_times)
    gt_p = np.array([p for _, p in gt_poses])
    est_p = np.array([p for _, p in est_poses])

    lap_time = SQ_PERIM / SQ_SPEED
    lap_idx_boundaries = [k * lap_time for k in range(1, SQ_N_LAPS + 1)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    plot_walls(ax)
    ax.plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=1.5, label="ground truth")
    ax.plot(est_p[:, 0], est_p[:, 1], "r-", lw=1.0, alpha=0.85, label="IESKF estimate")
    ax.plot(gt_p[0, 0], gt_p[0, 1], "ks", ms=8, label="start")
    ax.plot(est_p[-1, 0], est_p[-1, 1], "ro", ms=8, label="estimate end")
    ax.plot(gt_p[-1, 0], gt_p[-1, 1], "k^", ms=8, label="GT end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Trajectory ({SQ_N_LAPS} laps, no loop closure)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_arr, err_pos, "r-", lw=1.2)
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.6)
    for k in range(SQ_N_LAPS):
        ax.text(k * lap_time + lap_time / 2, ax.get_ylim()[1] * 0.95,
                f"lap {k+1}", ha="center", fontsize=8, color="gray")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title("Drift magnitude vs time")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    err_x = est_p[:, 0] - gt_p[:, 0]
    err_y = est_p[:, 1] - gt_p[:, 1]
    err_z = est_p[:, 2] - gt_p[:, 2]
    ax.plot(t_arr, err_x, label="x", lw=1.0)
    ax.plot(t_arr, err_y, label="y", lw=1.0)
    ax.plot(t_arr, err_z, label="z", lw=1.0)
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("component error (m)")
    ax.set_title("Per-axis drift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_arr, np.degrees(err_yaw), "b-", lw=1.0, label="|yaw err| (deg)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|yaw error| (deg)", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_arr, map_sizes, "g-", lw=1.0, alpha=0.7, label="map points")
    ax2.set_ylabel("map size (points)", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(f"Yaw drift + map size (cube_len={CUBE_LEN:.1f} m)")

    ax = axes[0, 2]
    ax.plot(t_arr, err_pos, "r-", lw=1.2, label="actual |error|")
    ax.plot(t_arr, 3 * filter_pos_std, "b--", lw=1.0, label="filter 3σ (raw)")
    ax.plot(t_arr, 3 * inflated_pos_std, color="#00aa00", lw=1.2,
            label="filter 3σ + CRLB drift")
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Consistency: actual error vs filter σ (raw vs CRLB-inflated)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.semilogy(t_arr, np.maximum(nees_pos, 1e-3), "r-", lw=1.0,
                label="NEES pos (raw filter)")
    ax.semilogy(t_arr, np.maximum(nees_pos_inf, 1e-3), color="#00aa00", lw=1.2,
                label="NEES pos (CRLB-inflated)")
    ax.semilogy(t_arr, np.maximum(nees_rot, 1e-3), "b-", lw=1.0, alpha=0.6,
                label="NEES rot (raw filter)")
    ax.semilogy(t_arr, np.maximum(nees_rot_inf, 1e-3), color="#cc44cc", lw=1.0,
                alpha=0.8, label="NEES rot (CRLB-inflated)")
    ax.axhline(3.0, color="k", ls="--", lw=1.2, label="expected (3 DOF)")
    ax.axhline(7.81, color="orange", ls=":", lw=1.0, label="χ²₃ 95% bound")
    for tb in lap_idx_boundaries:
        ax.axvline(tb, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("NEES (log scale)")
    ax.set_title("NEES: raw filter vs CRLB-inflated. Target ≈ 3")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/sim_square_corridor_no_lc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")

    print("\nPer-lap stats:")
    print(f"  {'lap':>4} {'mean drift (m)':>16} {'max drift (m)':>16} {'end drift (m)':>16}")
    for k in range(SQ_N_LAPS):
        t_lo = k * lap_time
        t_hi = (k + 1) * lap_time
        mask = (t_arr >= t_lo) & (t_arr < t_hi)
        if not mask.any():
            continue
        seg = err_pos[mask]
        print(f"  {k+1:>4} {seg.mean():>16.4f} {seg.max():>16.4f} {seg[-1]:>16.4f}")

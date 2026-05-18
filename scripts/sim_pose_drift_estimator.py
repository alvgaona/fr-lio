"""Script F: Tightly-coupled 3D pose + drift covariance with health-aware PCRB.

Built on top of the working sim_iekf_3d.py framework (15-DOF IESKF, online
voxel map, point-to-plane updates). Adds a parallel shadow recursion that
maintains P_drift = PCRB + accumulated divergence-gap term, where the
divergence gap responds to per-scan health alphas (convergence, innovation,
geometry).

Rationale: a pre-built fixed map is conceptually cleaner but the
contribution (PCRB + health-aware inflation) is independent of whether the
map is fixed or grown online. Sticking with the online map keeps the
baseline filter behavior identical to what laser_mapping.cpp does.

Failure injection: outlier-flood and sensor-blockage windows to verify the
health alphas catch failure modes and that P_drift grows in response.

Run: python sim_pose_drift_estimator.py
"""

import os
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_imu_trajectory import (
    GRAVITY, IMU_RATE, INITIAL_GYR_BIAS, INITIAL_ACC_BIAS, trajectory_at,
)
from sim_iekf_3d import (
    State, predict, rodrigues_exp, rodrigues_log, skew,
    OnlineMap, fit_plane,
    DT_IMU, N_STATE_DIM,
    LIDAR_POINT_VAR, SIGMA_RANGE_PERPOINT, MAX_PLANE_DIST,
    MAX_POINTS_PER_SCAN, MAX_IEKF_ITERS, CONVERGENCE_THRESH,
    NN_K, PLANE_FIT_THRESH, MAP_VOXEL_SIZE,
    initial_state_and_cov,
    find_correspondences, compute_residuals_and_H,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)

# Failure injection windows (seconds from start)
INJECT_OUTLIER_T = (8.0, 10.0)
INJECT_BLOCKAGE_T = (15.0, 17.0)

# Health-alpha tuning. All produce α ∈ [0, 1]: 1 = healthy, 0 = failed.
# Convergence: penalize when the last IESKF inner-step delta is large.
ALPHA_CONV_DX_MAX = 0.1
# Innovation: NIS / DOF ratio. ~1 is healthy; large = bad fit.
ALPHA_INNO_NIS_K = 5.0
# Geometric: smallest eigenvalue of measurement info on pose block.
# Calibrated empirically: typical healthy lambda_min is ~1e4 with 1000 points
# at LIDAR_POINT_VAR. Use ratio against a running median for adaptivity.
ALPHA_GEOM_REL = 0.1  # alpha=0 if lambda_min < 10% of recent median
ALPHA_BLEND = "min"


def inject_failures(scan, t):
    """Possibly corrupt the scan body points to simulate sensor failure."""
    points = scan["points_body"]
    in_failure = False
    if INJECT_OUTLIER_T[0] <= t <= INJECT_OUTLIER_T[1] and len(points) > 0:
        n_replace = int(0.5 * len(points))
        idx = np.random.choice(len(points), n_replace, replace=False)
        points = points.copy()
        points[idx] = np.random.uniform(-10, 10, size=(n_replace, 3))
        in_failure = True
    if INJECT_BLOCKAGE_T[0] <= t <= INJECT_BLOCKAGE_T[1]:
        if len(points) > 30:
            keep = max(15, int(0.03 * len(points)))
            idx = np.random.choice(len(points), keep, replace=False)
            points = points[idx]
            in_failure = True
    return points, in_failure


def iekf_update_with_health(state_prior, P_prior, points_body, online_map):
    """IESKF update augmented with per-scan health metrics."""
    matches = find_correspondences(state_prior, points_body, online_map)
    info = {
        "n_iter": 0, "final_dx_norm": np.inf,
        "NIS": np.nan, "lambda_min_meas": 0.0,
        "n_matches": len(matches),
        "meas_info_pose": None,
    }
    if len(matches) < 5:
        return state_prior, P_prior, info

    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]

    R_meas = LIDAR_POINT_VAR * np.eye(len(matches))
    inv_var = 1.0 / LIDAR_POINT_VAR
    state = state_prior.copy()
    H_final = K_final = h_final = S_final = None
    final_dx_norm = np.inf

    for j in range(MAX_IEKF_ITERS):
        h, H = compute_residuals_and_H(state, matches)
        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)
        dx_from_prior = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx_from_prior)
        state = state_prior.boxplus(dx_from_prior + delta)
        H_final, K_final, h_final, S_final = H, K, h, S
        final_dx_norm = float(np.linalg.norm(delta))
        info["n_iter"] = j + 1
        if final_dx_norm < CONVERGENCE_THRESH:
            break

    P_new = (np.eye(N_STATE_DIM) - K_final @ H_final) @ P_prior

    info["final_dx_norm"] = final_dx_norm
    info["NIS"] = float(h_final @ np.linalg.solve(S_final, h_final))
    info_meas = inv_var * (H_final.T @ H_final)
    info["meas_info_pose"] = info_meas[0:6, 0:6]
    eigvals = np.linalg.eigvalsh(info["meas_info_pose"])
    info["lambda_min_meas"] = float(eigvals[0])
    return state, P_new, info


def health_alphas(info, lambda_min_window):
    """Per-scan health α ∈ [0, 1]. 1 = healthy, 0 = failed."""
    if info["n_matches"] < 5:
        return 0.0, 0.0, 0.0, 0.0

    alpha_conv = max(0.0, min(1.0, 1.0 - info["final_dx_norm"] / ALPHA_CONV_DX_MAX))
    dof = max(1, info["n_matches"])
    nis_ratio = info["NIS"] / dof
    alpha_inno = max(0.0, min(1.0, 1.0 - max(0.0, nis_ratio - 1.0) / (ALPHA_INNO_NIS_K - 1.0)))
    # Adaptive geom: compare against rolling median of recent lambda_min.
    if len(lambda_min_window) > 5:
        med = float(np.median(lambda_min_window))
        alpha_geom = max(0.0, min(1.0, info["lambda_min_meas"] / max(ALPHA_GEOM_REL * med, 1e-9)))
    else:
        alpha_geom = 1.0  # warm-up: trust geometry

    if ALPHA_BLEND == "min":
        alpha = min(alpha_conv, alpha_inno, alpha_geom)
    else:
        alpha = alpha_conv * alpha_inno * alpha_geom
    return alpha, alpha_conv, alpha_inno, alpha_geom


def run():
    print("Loading simulated data ...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    # Restrict to first T_MAX seconds for faster iteration.
    T_MAX = 25.0
    imu_arr = imu_arr[imu_arr[:, 0] <= T_MAX]
    lidar_arr = [s for s in lidar_arr if s["t"] <= T_MAX]
    imu_data = [{"t": row[0], "gyro": row[1:4], "acc": row[4:7]} for row in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_data)} IMU samples, {len(lidar_data)} LiDAR scans (t ≤ {T_MAX}s)")

    np.random.seed(123)
    state, P = initial_state_and_cov()
    online_map = OnlineMap()

    # Shadow: PCRB on 6-DOF pose subspace (rot + trans) and divergence-gap
    # accumulator. Initialized from EKF prior on pose.
    J_pcrb = np.linalg.inv(P[0:6, 0:6])
    div_gap = np.zeros((6, 6))
    lambda_min_window = deque(maxlen=30)

    history = {
        "t": [], "p_est": [], "p_true": [], "rot_err": [], "err_body": [],
        "P_filter_pos_trace": [], "P_drift_pos_trace": [],
        "P_filter_rot_trace": [], "P_drift_rot_trace": [],
        "P_pcrb_pos_trace": [], "P_pcrb_rot_trace": [],
    }
    scan_history = {
        "scan_t": [], "alpha": [], "alpha_conv": [], "alpha_inno": [],
        "alpha_geom": [], "nis": [], "n_matches": [], "in_failure": [],
    }

    lidar_idx = 0
    last_pcrb_pos = np.trace(P[3:6, 3:6])
    last_pcrb_rot = np.trace(P[0:3, 0:3])
    last_drift_pos = last_pcrb_pos
    last_drift_rot = last_pcrb_rot
    for sample in imu_data:
        t = sample["t"]
        gyro = sample["gyro"]
        acc = sample["acc"]
        state, P = predict(state, P, gyro, acc, DT_IMU)

        # PCRB prediction handled at scan boundaries (below) to avoid
        # 23k matrix inversions inside the IMU loop.

        while lidar_idx < len(lidar_data) and lidar_data[lidar_idx]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[lidar_idx]
            scan_t = scan["t"]
            points_body, in_failure = inject_failures(scan, scan_t)

            # PCRB predict step at scan boundary: sync J_pcrb to current EKF
            # pose belief, then add measurement info during update below.
            # This is the "follow-the-filter" PCRB approximation — exact PCRB
            # would require sigma-point evaluation of E[H^T R^-1 H], deferred.
            J_pcrb = np.linalg.inv(P[0:6, 0:6] + 1e-12 * np.eye(6))

            if len(points_body) >= 5 and len(online_map) >= NN_K:
                state, P, info = iekf_update_with_health(
                    state, P, points_body, online_map,
                )
                if info["meas_info_pose"] is not None:
                    if info["lambda_min_meas"] > 0:
                        lambda_min_window.append(info["lambda_min_meas"])
                alpha, ac, ai, ag = health_alphas(info, lambda_min_window)

                # PCRB measurement update + divergence gap. Both gated on
                # finite, valid measurement info.
                if info["meas_info_pose"] is not None:
                    J_pcrb = J_pcrb + info["meas_info_pose"]
                    try:
                        single_step_floor = np.linalg.inv(
                            info["meas_info_pose"] + 1e-9 * np.eye(6))
                        div_gap = div_gap + (1.0 - alpha) * single_step_floor
                    except np.linalg.LinAlgError:
                        pass

                scan_history["alpha"].append(alpha)
                scan_history["alpha_conv"].append(ac)
                scan_history["alpha_inno"].append(ai)
                scan_history["alpha_geom"].append(ag)
                scan_history["nis"].append(info["NIS"])
                scan_history["n_matches"].append(info["n_matches"])
            else:
                scan_history["alpha"].append(0.0)
                scan_history["alpha_conv"].append(0.0)
                scan_history["alpha_inno"].append(0.0)
                scan_history["alpha_geom"].append(0.0)
                scan_history["nis"].append(np.nan)
                scan_history["n_matches"].append(len(points_body))
                div_gap = div_gap + 0.01 * np.eye(6)  # blind step inflation

            scan_history["scan_t"].append(scan_t)
            scan_history["in_failure"].append(in_failure)

            # Cache scan-boundary PCRB + drift trace (used by IMU-rate history)
            try:
                P_pcrb_block = np.linalg.inv(J_pcrb + 1e-12 * np.eye(6))
                P_drift_block = P_pcrb_block + div_gap
                last_pcrb_pos = np.trace(P_pcrb_block[3:6, 3:6])
                last_pcrb_rot = np.trace(P_pcrb_block[0:3, 0:3])
                last_drift_pos = np.trace(P_drift_block[3:6, 3:6])
                last_drift_rot = np.trace(P_drift_block[0:3, 0:3])
            except np.linalg.LinAlgError:
                pass

            world_points = (state.R @ scan["points_body"].T).T + state.p
            online_map.add_points(world_points)
            lidar_idx += 1

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        err_body = state.R.T @ (state.p - p_true)
        history["t"].append(t)
        history["p_est"].append(state.p.copy())
        history["p_true"].append(p_true.copy())
        history["rot_err"].append(rot_err)
        history["err_body"].append(err_body)
        history["P_filter_pos_trace"].append(np.trace(P[3:6, 3:6]))
        history["P_filter_rot_trace"].append(np.trace(P[0:3, 0:3]))
        history["P_pcrb_pos_trace"].append(last_pcrb_pos)
        history["P_pcrb_rot_trace"].append(last_pcrb_rot)
        history["P_drift_pos_trace"].append(last_drift_pos)
        history["P_drift_rot_trace"].append(last_drift_rot)

    for k in list(history.keys()):
        history[k] = np.array(history[k])
    for k in list(scan_history.keys()):
        scan_history[k] = np.array(scan_history[k])
    return history, scan_history


def plot_and_summarize(hist, scan_hist):
    t = hist["t"]
    err_pos = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    err_rot = np.linalg.norm(hist["rot_err"], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(t, err_pos, "k-", linewidth=1.4, label="‖pos err‖ (m)")
    ax.plot(t, np.sqrt(hist["P_filter_pos_trace"]), "b--", linewidth=1.0,
            label="√trace(P_filter)")
    ax.plot(t, np.sqrt(hist["P_pcrb_pos_trace"]), "g-", linewidth=1.0,
            label="√trace(PCRB)")
    ax.plot(t, np.sqrt(hist["P_drift_pos_trace"]), "r-", linewidth=1.4,
            label="√trace(P_drift) = PCRB + div gap")
    for t0, t1, c, lbl in [(INJECT_OUTLIER_T[0], INJECT_OUTLIER_T[1], "tab:orange", "outliers"),
                            (INJECT_BLOCKAGE_T[0], INJECT_BLOCKAGE_T[1], "tab:purple", "blockage")]:
        ax.axvspan(t0, t1, alpha=0.15, color=c, label=f"inject {lbl}")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("position (m)")
    ax.set_title("Pos error vs published uncertainty")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.plot(t, err_rot, "k-", linewidth=1.4, label="‖rot err‖ (rad)")
    ax.plot(t, np.sqrt(hist["P_filter_rot_trace"]), "b--", linewidth=1.0,
            label="√trace(P_filter)")
    ax.plot(t, np.sqrt(hist["P_pcrb_rot_trace"]), "g-", linewidth=1.0,
            label="√trace(PCRB)")
    ax.plot(t, np.sqrt(hist["P_drift_rot_trace"]), "r-", linewidth=1.4,
            label="√trace(P_drift)")
    for t0, t1, c in [(INJECT_OUTLIER_T[0], INJECT_OUTLIER_T[1], "tab:orange"),
                       (INJECT_BLOCKAGE_T[0], INJECT_BLOCKAGE_T[1], "tab:purple")]:
        ax.axvspan(t0, t1, alpha=0.15, color=c)
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("rad")
    ax.set_title("Rot error vs published uncertainty")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.plot(scan_hist["scan_t"], scan_hist["alpha"], "k-", linewidth=1.3, label="α combined")
    ax.plot(scan_hist["scan_t"], scan_hist["alpha_conv"], "tab:blue", linewidth=0.8, label="α_conv")
    ax.plot(scan_hist["scan_t"], scan_hist["alpha_inno"], "tab:green", linewidth=0.8, label="α_inno")
    ax.plot(scan_hist["scan_t"], scan_hist["alpha_geom"], "tab:orange", linewidth=0.8, label="α_geom")
    for t0, t1, c in [(INJECT_OUTLIER_T[0], INJECT_OUTLIER_T[1], "tab:orange"),
                       (INJECT_BLOCKAGE_T[0], INJECT_BLOCKAGE_T[1], "tab:purple")]:
        ax.axvspan(t0, t1, alpha=0.15, color=c)
    ax.set_xlabel("t (s)"); ax.set_ylabel("α")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Per-scan health α  (1 = healthy, 0 = failed)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    nees_filter = np.zeros_like(t)
    nees_drift = np.zeros_like(t)
    for k in range(len(t)):
        eps = hist["p_est"][k] - hist["p_true"][k]
        var_filter = hist["P_filter_pos_trace"][k] / 3.0
        var_drift = hist["P_drift_pos_trace"][k] / 3.0
        nees_filter[k] = (eps @ eps) / max(var_filter, 1e-12)
        nees_drift[k] = (eps @ eps) / max(var_drift, 1e-12)
    ax.plot(t, nees_filter, "b--", linewidth=1.0, label="NEES vs P_filter")
    ax.plot(t, nees_drift, "r-", linewidth=1.0, label="NEES vs P_drift")
    ax.axhline(3.0, color="black", linestyle="-", linewidth=0.8, label="target (3)")
    for t0, t1, c in [(INJECT_OUTLIER_T[0], INJECT_OUTLIER_T[1], "tab:orange"),
                       (INJECT_BLOCKAGE_T[0], INJECT_BLOCKAGE_T[1], "tab:purple")]:
        ax.axvspan(t0, t1, alpha=0.15, color=c)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("NEES (pos, scalarized)")
    ax.set_title("NEES — P_drift should stay closer to 3 across failures")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_pose_drift_estimator.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")

    print("\n=== Health summary ===")
    healthy_mask = ~scan_hist["in_failure"]
    failure_mask = scan_hist["in_failure"]
    if healthy_mask.any():
        print(f"  Healthy scans ({healthy_mask.sum()}): α mean = {scan_hist['alpha'][healthy_mask].mean():.3f}")
    if failure_mask.any():
        print(f"  Failure scans ({failure_mask.sum()}): α mean = {scan_hist['alpha'][failure_mask].mean():.3f}")
    print(f"\nFinal position error : {np.linalg.norm(hist['p_est'][-1] - hist['p_true'][-1]):.4f} m")
    print(f"Final P_filter trace : {hist['P_filter_pos_trace'][-1]:.4e}")
    print(f"Final PCRB    trace  : {hist['P_pcrb_pos_trace'][-1]:.4e}")
    print(f"Final P_drift trace  : {hist['P_drift_pos_trace'][-1]:.4e}")


if __name__ == "__main__":
    hist, scan_hist = run()
    plot_and_summarize(hist, scan_hist)

"""Unified experiment runner for buckets #1, #2, #3.

Selected via the CONFIG environment variable:

  CONFIG=baseline  → Bucket #1: vanilla IESKF, scalar R, no S2S inflation.
  CONFIG=s2s       → Bucket #2: vanilla IESKF + S2S CRLB drift inflation on
                     the *published* covariance only (filter state unchanged).
  CONFIG=perpoint  → Bucket #3: per-point measurement covariance in the IESKF
                     R (GICP-style: σ_r² + nᵀΣ_pn).
  CONFIG=full      → Buckets #2 + #3 combined: per-point R in the IESKF AND
                     S2S CRLB inflation on the published covariance.

SIM_ENV selects the environment (cube / corridor / room_corridor /
corridor_grid). Outputs are tagged `exp_{config}_{env}` and saved to
sim_data/ alongside the baseline outputs.

Saves CSVs for trajectory, error, filter cov, published cov, P_drift, NEES
(both filter and published) and four diagnostic plots.
"""

import csv
import os
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import ENVIRONMENT
from sim_imu_trajectory import IMU_RATE, trajectory_at
from sim_iekf_3d import (
    DT_IMU, NN_K, OnlineMap, State, compute_scan_to_scan_covariance,
    iekf_update, initial_state_and_cov, predict, rodrigues_log,
    S2S_ADAPTIVE_REJECT_RATIO, S2S_ADAPTIVE_WINDOW,
    S2S_MIN_ROT_RAD, S2S_MIN_TRANS_M, S2S_MIN_VALID_POINTS,
)

CONFIG = os.environ.get("CONFIG", "baseline").lower()
assert CONFIG in {"baseline", "s2s", "perpoint", "full"}, (
    f"Unknown CONFIG={CONFIG}; expected baseline|s2s|perpoint|full"
)
USE_PERPOINT = CONFIG in {"perpoint", "full"}
USE_S2S = CONFIG in {"s2s", "full"}

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)
TAG = f"exp_{CONFIG}_{ENVIRONMENT}"
CSV_DIR = os.path.join(OUT_DIR, f"{TAG}_csv")
os.makedirs(CSV_DIR, exist_ok=True)


def run_experiment(imu_data, lidar_data):
    state, P = initial_state_and_cov()
    online_map = OnlineMap()

    prev_scan_points = None
    prev_tree = None
    R_prev_s2s = None
    t_prev_s2s = None
    P_drift = np.zeros((6, 6))  # world-frame, [pos; rot] block order
    r_s2s_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    p_rel_trace_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)

    log = {
        "t": [],
        "p_est": [], "p_true": [],
        "err_pos": [], "err_rot": [],
        "P_pos_diag": [], "P_rot_diag": [],
        "P_pub_pos_diag": [], "P_pub_rot_diag": [],
        "P_drift_pos_trace": [], "P_drift_rot_trace": [],
        "nees_pos_filter": [], "nees_rot_filter": [],
        "nees_pos_pub": [], "nees_rot_pub": [],
        "map_size": [],
    }

    lidar_idx = 0
    for sample in imu_data:
        t = sample["t"]
        gyro = sample["gyro"]
        acc = sample["acc"]

        state, P = predict(state, P, gyro, acc, DT_IMU)

        while lidar_idx < len(lidar_data) and lidar_data[lidar_idx]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[lidar_idx]
            if len(scan["points_body"]) > 0:
                if len(online_map) >= NN_K:
                    state, P = iekf_update(
                        state, P, scan["points_body"], online_map,
                        use_fej=False,
                        use_degen_suppression=False,
                        use_perpoint_cov=USE_PERPOINT,
                    )

                if USE_S2S and prev_tree is not None:
                    t_rel_gate = np.linalg.norm(state.p - t_prev_s2s)
                    rot_rel_gate = np.linalg.norm(
                        rodrigues_log(R_prev_s2s.T @ state.R))
                    if t_rel_gate > S2S_MIN_TRANS_M or rot_rel_gate > S2S_MIN_ROT_RAD:
                        P_rel, dbg = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan_points, prev_tree,
                            R_prev_s2s, t_prev_s2s,
                            use_perpoint_cov=USE_PERPOINT,
                        )
                        r_s2s_val = dbg["R_s2s"]
                        p_rel_trace = float(np.trace(P_rel))

                        reject = False
                        if len(r_s2s_window) >= 5:
                            r_med = float(np.median(r_s2s_window))
                            if r_s2s_val > S2S_ADAPTIVE_REJECT_RATIO * max(r_med, 1e-12):
                                reject = True
                        if not reject and len(p_rel_trace_window) >= 5:
                            p_med = float(np.median(p_rel_trace_window))
                            if p_rel_trace > S2S_ADAPTIVE_REJECT_RATIO * max(p_med, 1e-12):
                                reject = True

                        if not reject and dbg["valid_count"] >= S2S_MIN_VALID_POINTS:
                            Adj_prev = np.zeros((6, 6))
                            Adj_prev[0:3, 0:3] = R_prev_s2s
                            Adj_prev[3:6, 3:6] = R_prev_s2s
                            P_drift += Adj_prev @ P_rel @ Adj_prev.T
                            if r_s2s_val > 0:
                                r_s2s_window.append(r_s2s_val)
                            if p_rel_trace > 0:
                                p_rel_trace_window.append(p_rel_trace)

                if USE_S2S:
                    prev_scan_points = scan["points_body"].copy()
                    prev_tree = cKDTree(prev_scan_points)
                    R_prev_s2s = state.R.copy()
                    t_prev_s2s = state.p.copy()

                world_points = (state.R @ scan["points_body"].T).T + state.p
                online_map.add_points(world_points)
            lidar_idx += 1

        # Rotate world-frame P_drift back into current body tangent.
        Adj_cT = np.zeros((6, 6))
        Adj_cT[0:3, 0:3] = state.R.T
        Adj_cT[3:6, 3:6] = state.R.T
        P_drift_body = Adj_cT @ P_drift @ Adj_cT.T

        # Filter cov is in [rot, pos, ...] order; P_drift is in [pos; rot].
        P_pub = P.copy()
        P_pub[3:6, 3:6] += P_drift_body[0:3, 0:3]
        P_pub[0:3, 0:3] += P_drift_body[3:6, 3:6]

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        pos_err = state.p - p_true

        def nees(err, M):
            try:
                return float(err @ np.linalg.solve(M, err))
            except np.linalg.LinAlgError:
                return np.nan

        nees_p_f = nees(pos_err, P[3:6, 3:6])
        nees_r_f = nees(rot_err, P[0:3, 0:3])
        nees_p_p = nees(pos_err, P_pub[3:6, 3:6])
        nees_r_p = nees(rot_err, P_pub[0:3, 0:3])

        log["t"].append(t)
        log["p_est"].append(state.p.copy())
        log["p_true"].append(p_true.copy())
        log["err_pos"].append(pos_err.copy())
        log["err_rot"].append(rot_err.copy())
        log["P_pos_diag"].append(np.diag(P[3:6, 3:6]).copy())
        log["P_rot_diag"].append(np.diag(P[0:3, 0:3]).copy())
        log["P_pub_pos_diag"].append(np.diag(P_pub[3:6, 3:6]).copy())
        log["P_pub_rot_diag"].append(np.diag(P_pub[0:3, 0:3]).copy())
        log["P_drift_pos_trace"].append(float(np.trace(P_drift[0:3, 0:3])))
        log["P_drift_rot_trace"].append(float(np.trace(P_drift[3:6, 3:6])))
        log["nees_pos_filter"].append(nees_p_f)
        log["nees_rot_filter"].append(nees_r_f)
        log["nees_pos_pub"].append(nees_p_p)
        log["nees_rot_pub"].append(nees_r_p)
        log["map_size"].append(len(online_map))

    for key in [
        "t", "p_est", "p_true", "err_pos", "err_rot",
        "P_pos_diag", "P_rot_diag", "P_pub_pos_diag", "P_pub_rot_diag",
        "P_drift_pos_trace", "P_drift_rot_trace",
        "nees_pos_filter", "nees_rot_filter",
        "nees_pos_pub", "nees_rot_pub", "map_size",
    ]:
        log[key] = np.array(log[key])
    return log


def save_csvs(log):
    t = log["t"]
    n = len(t)

    with open(os.path.join(CSV_DIR, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z"])
        for i in range(n):
            w.writerow([t[i],
                        log["p_est"][i, 0], log["p_est"][i, 1], log["p_est"][i, 2],
                        log["p_true"][i, 0], log["p_true"][i, 1], log["p_true"][i, 2]])

    err_pos_norm = np.linalg.norm(log["err_pos"], axis=1)
    err_rot_norm = np.linalg.norm(log["err_rot"], axis=1)
    with open(os.path.join(CSV_DIR, "error.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t",
                    "err_x", "err_y", "err_z", "err_pos_norm",
                    "err_rx", "err_ry", "err_rz", "err_rot_norm"])
        for i in range(n):
            w.writerow([t[i],
                        log["err_pos"][i, 0], log["err_pos"][i, 1], log["err_pos"][i, 2],
                        err_pos_norm[i],
                        log["err_rot"][i, 0], log["err_rot"][i, 1], log["err_rot"][i, 2],
                        err_rot_norm[i]])

    with open(os.path.join(CSV_DIR, "covariance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t",
                    "std_pos_x", "std_pos_y", "std_pos_z",
                    "std_rot_x", "std_rot_y", "std_rot_z",
                    "std_pub_pos_x", "std_pub_pos_y", "std_pub_pos_z",
                    "std_pub_rot_x", "std_pub_rot_y", "std_pub_rot_z"])
        for i in range(n):
            sp = np.sqrt(log["P_pos_diag"][i])
            sr = np.sqrt(log["P_rot_diag"][i])
            sp_pub = np.sqrt(log["P_pub_pos_diag"][i])
            sr_pub = np.sqrt(log["P_pub_rot_diag"][i])
            w.writerow([t[i], sp[0], sp[1], sp[2], sr[0], sr[1], sr[2],
                        sp_pub[0], sp_pub[1], sp_pub[2],
                        sr_pub[0], sr_pub[1], sr_pub[2]])

    with open(os.path.join(CSV_DIR, "p_drift.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "p_drift_pos_trace", "p_drift_rot_trace"])
        for i in range(n):
            w.writerow([t[i], log["P_drift_pos_trace"][i], log["P_drift_rot_trace"][i]])

    with open(os.path.join(CSV_DIR, "nees.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "nees_pos_filter", "nees_rot_filter",
                    "nees_pos_pub", "nees_rot_pub"])
        for i in range(n):
            w.writerow([t[i],
                        log["nees_pos_filter"][i], log["nees_rot_filter"][i],
                        log["nees_pos_pub"][i], log["nees_rot_pub"][i]])

    print(f"Saved CSVs to {CSV_DIR}")


def plot_trajectory(log):
    p_est, p_true = log["p_est"], log["p_true"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p_true[:, 0], p_true[:, 1], "k-", linewidth=1.6, label="ground truth")
    ax.plot(p_est[:, 0], p_est[:, 1], "r-", linewidth=1.1, alpha=0.85,
            label=f"{CONFIG} IESKF")
    ax.scatter(p_true[0, 0], p_true[0, 1], color="green", marker="o", s=60, zorder=5, label="start")
    ax.scatter(p_true[-1, 0], p_true[-1, 1], color="blue", marker="s", s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title(f"Trajectory (top-down) — {CONFIG} on {ENVIRONMENT}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_trajectory.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_error(log):
    t = log["t"]
    err_p = np.linalg.norm(log["err_pos"], axis=1)
    err_r = np.linalg.norm(log["err_rot"], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t, err_p, "k-", linewidth=1.1, label="‖e_p‖")
    axes[0].plot(t, np.abs(log["err_pos"][:, 0]), "r-", linewidth=0.7, alpha=0.6, label="|e_x|")
    axes[0].plot(t, np.abs(log["err_pos"][:, 1]), "g-", linewidth=0.7, alpha=0.6, label="|e_y|")
    axes[0].plot(t, np.abs(log["err_pos"][:, 2]), "b-", linewidth=0.7, alpha=0.6, label="|e_z|")
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("position error (m)")
    axes[0].set_title("Position error"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].plot(t, err_r, "k-", linewidth=1.1, label="‖e_R‖")
    axes[1].plot(t, np.abs(log["err_rot"][:, 0]), "r-", linewidth=0.7, alpha=0.6, label="|roll|")
    axes[1].plot(t, np.abs(log["err_rot"][:, 1]), "g-", linewidth=0.7, alpha=0.6, label="|pitch|")
    axes[1].plot(t, np.abs(log["err_rot"][:, 2]), "b-", linewidth=0.7, alpha=0.6, label="|yaw|")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("rotation error (rad)")
    axes[1].set_title("Rotation error"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_error.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_covariance(log):
    t = log["t"]
    std_p = np.sqrt(log["P_pos_diag"])
    std_r = np.sqrt(log["P_rot_diag"])
    std_pp = np.sqrt(log["P_pub_pos_diag"])
    std_pr = np.sqrt(log["P_pub_rot_diag"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, c, name in [(0, "r", "x"), (1, "g", "y"), (2, "b", "z")]:
        axes[0].plot(t, std_p[:, j], color=c, linewidth=1.0, linestyle="--", alpha=0.6,
                     label=f"σ_{name} (filter)")
        if USE_S2S:
            axes[0].plot(t, std_pp[:, j], color=c, linewidth=1.0, label=f"σ_{name} (pub)")
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("position std (m)")
    axes[0].set_title("Position covariance (per-axis std)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    for j, c, name in [(0, "r", "roll"), (1, "g", "pitch"), (2, "b", "yaw")]:
        axes[1].plot(t, std_r[:, j], color=c, linewidth=1.0, linestyle="--", alpha=0.6,
                     label=f"σ_{name} (filter)")
        if USE_S2S:
            axes[1].plot(t, std_pr[:, j], color=c, linewidth=1.0, label=f"σ_{name} (pub)")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("rotation std (rad)")
    axes[1].set_title("Rotation covariance (per-axis std)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_covariance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_nees(log):
    t = log["t"]
    lo, hi = chi2.ppf([0.025, 0.975], df=3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    series = [
        (axes[0], log["nees_pos_filter"], log["nees_pos_pub"], "Position NEES"),
        (axes[1], log["nees_rot_filter"], log["nees_rot_pub"], "Rotation NEES"),
    ]
    for ax, fvals, pvals, title in series:
        ax.plot(t, fvals, "r-", linewidth=0.7, alpha=0.7, label="filter")
        if USE_S2S:
            ax.plot(t, pvals, "b-", linewidth=0.7, alpha=0.7, label="published (s2s)")
        ax.axhline(3.0, color="k", linestyle="-", linewidth=1.0, label="expected (3)")
        ax.axhline(lo, color="g", linestyle="--", linewidth=1.0, label=f"95% lo ({lo:.2f})")
        ax.axhline(hi, color="orange", linestyle="--", linewidth=1.0, label=f"95% hi ({hi:.2f})")
        finite = fvals[np.isfinite(fvals)]
        if len(finite):
            ax.text(0.02, 0.95, f"mean(filter)={finite.mean():.2f}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("t (s)"); ax.set_ylabel("NEES"); ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_nees.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_p_drift(log):
    if not USE_S2S:
        return
    t = log["t"]
    err_pos_norm = np.linalg.norm(log["err_pos"], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(t, log["P_drift_pos_trace"], "b-", linewidth=1.1)
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("trace(P_drift pos) (m²)")
    axes[0].set_title("S2S P_drift — position growth"); axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].plot(t, log["P_drift_rot_trace"], "b-", linewidth=1.1)
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("trace(P_drift rot) (rad²)")
    axes[1].set_title("S2S P_drift — rotation growth"); axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    axes[2].plot(t, err_pos_norm, "k-", linewidth=1.0, label="actual ‖e_p‖")
    axes[2].plot(t, np.sqrt(np.sum(log["P_pos_diag"], axis=1)),
                 "r--", linewidth=1.0, label="filter pos std")
    axes[2].plot(t, np.sqrt(np.sum(log["P_pub_pos_diag"], axis=1)),
                 "b-", linewidth=1.0, label="published pos std")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("position (m)")
    axes[2].set_title("Actual vs filter vs published"); axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_p_drift.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def summarize(log):
    err_p = np.linalg.norm(log["err_pos"], axis=1)
    err_r = np.linalg.norm(log["err_rot"], axis=1)
    nees_pf = log["nees_pos_filter"][np.isfinite(log["nees_pos_filter"])]
    nees_rf = log["nees_rot_filter"][np.isfinite(log["nees_rot_filter"])]
    nees_pp = log["nees_pos_pub"][np.isfinite(log["nees_pos_pub"])]
    nees_rp = log["nees_rot_pub"][np.isfinite(log["nees_rot_pub"])]
    lo, hi = chi2.ppf([0.025, 0.975], df=3)
    print(f"\n[CONFIG={CONFIG}  env={ENVIRONMENT}]")
    print(f"  Position error : mean={err_p.mean():.4f} m  max={err_p.max():.4f} m  final={err_p[-1]:.4f} m")
    print(f"  Rotation error : mean={err_r.mean():.4f} rad max={err_r.max():.4f} rad final={err_r[-1]:.4f} rad")
    print(f"  NEES pos filter: mean={nees_pf.mean():.2f}  in 95%=[{lo:.2f},{hi:.2f}] : {np.mean((nees_pf>=lo)&(nees_pf<=hi)):.1%}")
    print(f"  NEES rot filter: mean={nees_rf.mean():.2f}  in 95%=[{lo:.2f},{hi:.2f}] : {np.mean((nees_rf>=lo)&(nees_rf<=hi)):.1%}")
    if USE_S2S:
        print(f"  NEES pos pub   : mean={nees_pp.mean():.2f}  in 95%: {np.mean((nees_pp>=lo)&(nees_pp<=hi)):.1%}")
        print(f"  NEES rot pub   : mean={nees_rp.mean():.2f}  in 95%: {np.mean((nees_rp>=lo)&(nees_rp<=hi)):.1%}")
        print(f"  Final P_drift  : pos_trace={log['P_drift_pos_trace'][-1]:.5f} m²  "
              f"rot_trace={log['P_drift_rot_trace'][-1]:.5f} rad²")


if __name__ == "__main__":
    print(f"[exp_run] CONFIG={CONFIG}  ENVIRONMENT={ENVIRONMENT}  "
          f"USE_PERPOINT={USE_PERPOINT}  USE_S2S={USE_S2S}")
    print(f"Loading simulated data from {OUT_DIR}...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans, "
          f"IMU rate={IMU_RATE} Hz")

    np.random.seed(123)
    log = run_experiment(imu_data, lidar_data)

    summarize(log)
    save_csvs(log)
    plot_trajectory(log)
    plot_error(log)
    plot_covariance(log)
    plot_nees(log)
    plot_p_drift(log)

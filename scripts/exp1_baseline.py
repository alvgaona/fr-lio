"""Experiment 1: FAST-LIO2-style baseline IESKF on cube room.

Bucket #1 of the thesis experiment plan. No CRLB inflation, no per-point
covariance, no loop closure. Saves CSVs for later TikZ plotting and renders
four diagnostic figures:
  - trajectory (top-down) estimated vs ground truth
  - position and yaw error over time
  - filter covariance (per-axis std) over time
  - NEES (consistency) for position and rotation
"""

import csv
import os

import numpy as np
from scipy.spatial import cKDTree  # noqa: F401  (kept for parity with sim_iekf_3d)
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import ENVIRONMENT
from sim_imu_trajectory import IMU_RATE, trajectory_at
from sim_iekf_3d import (
    DT_IMU, NN_K, OnlineMap, State, iekf_update, initial_state_and_cov, predict,
    rodrigues_log,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)
TAG = f"exp1_baseline_{ENVIRONMENT}"
CSV_DIR = os.path.join(OUT_DIR, f"{TAG}_csv")
os.makedirs(CSV_DIR, exist_ok=True)


def run_baseline(imu_data, lidar_data):
    """Baseline IESKF loop with full per-step logging.

    No s2s CRLB inflation, no per-point covariance, no loop closure.
    """
    state, P = initial_state_and_cov()
    online_map = OnlineMap()

    log = {
        "t": [],
        "p_est": [], "p_true": [],
        "R_est": [], "R_true": [],
        "err_pos": [],          # 3-vec body-frame error
        "err_rot": [],          # 3-vec rot error (so3)
        "P_pos_diag": [],       # diag of P[3:6,3:6]
        "P_rot_diag": [],       # diag of P[0:3,0:3]
        "nees_pos": [],
        "nees_rot": [],
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
                        use_perpoint_cov=False,
                    )
                world_points = (state.R @ scan["points_body"].T).T + state.p
                online_map.add_points(world_points)
            lidar_idx += 1

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        pos_err = state.p - p_true

        Ppos = P[3:6, 3:6]
        Prot = P[0:3, 0:3]
        try:
            nees_p = float(pos_err @ np.linalg.solve(Ppos, pos_err))
        except np.linalg.LinAlgError:
            nees_p = np.nan
        try:
            nees_r = float(rot_err @ np.linalg.solve(Prot, rot_err))
        except np.linalg.LinAlgError:
            nees_r = np.nan

        log["t"].append(t)
        log["p_est"].append(state.p.copy())
        log["p_true"].append(p_true.copy())
        log["R_est"].append(state.R.copy())
        log["R_true"].append(R_true.copy())
        log["err_pos"].append(pos_err.copy())
        log["err_rot"].append(rot_err.copy())
        log["P_pos_diag"].append(np.diag(Ppos).copy())
        log["P_rot_diag"].append(np.diag(Prot).copy())
        log["nees_pos"].append(nees_p)
        log["nees_rot"].append(nees_r)
        log["map_size"].append(len(online_map))

    for key in ["t", "p_est", "p_true", "err_pos", "err_rot",
                "P_pos_diag", "P_rot_diag", "nees_pos", "nees_rot", "map_size"]:
        log[key] = np.array(log[key])
    return log


def save_csvs(log):
    """Write CSVs for downstream TikZ plotting."""
    t = log["t"]
    n = len(t)

    with open(os.path.join(CSV_DIR, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z"])
        for i in range(n):
            w.writerow([t[i],
                        log["p_est"][i, 0], log["p_est"][i, 1], log["p_est"][i, 2],
                        log["p_true"][i, 0], log["p_true"][i, 1], log["p_true"][i, 2]])

    with open(os.path.join(CSV_DIR, "error.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t",
                    "err_x", "err_y", "err_z", "err_pos_norm",
                    "err_rx", "err_ry", "err_rz", "err_rot_norm"])
        err_pos_norm = np.linalg.norm(log["err_pos"], axis=1)
        err_rot_norm = np.linalg.norm(log["err_rot"], axis=1)
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
                    "std_rot_x", "std_rot_y", "std_rot_z"])
        for i in range(n):
            std_p = np.sqrt(log["P_pos_diag"][i])
            std_r = np.sqrt(log["P_rot_diag"][i])
            w.writerow([t[i], std_p[0], std_p[1], std_p[2], std_r[0], std_r[1], std_r[2]])

    with open(os.path.join(CSV_DIR, "nees.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "nees_pos", "nees_rot"])
        for i in range(n):
            w.writerow([t[i], log["nees_pos"][i], log["nees_rot"][i]])

    print(f"Saved CSVs to {CSV_DIR}")


def plot_trajectory(log):
    p_est, p_true = log["p_est"], log["p_true"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p_true[:, 0], p_true[:, 1], "k-", linewidth=1.6, label="ground truth")
    ax.plot(p_est[:, 0], p_est[:, 1], "r-", linewidth=1.1, alpha=0.85, label="baseline IESKF")
    ax.scatter(p_true[0, 0], p_true[0, 1], color="green", marker="o", s=60, zorder=5, label="start")
    ax.scatter(p_true[-1, 0], p_true[-1, 1], color="blue", marker="s", s=60, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Trajectory (top-down)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_trajectory.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_error(log):
    t = log["t"]
    err_pos_norm = np.linalg.norm(log["err_pos"], axis=1)
    err_rot_norm = np.linalg.norm(log["err_rot"], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(t, err_pos_norm, "k-", linewidth=1.1, label="‖e_p‖")
    axes[0].plot(t, np.abs(log["err_pos"][:, 0]), "r-", linewidth=0.7, alpha=0.6, label="|e_x|")
    axes[0].plot(t, np.abs(log["err_pos"][:, 1]), "g-", linewidth=0.7, alpha=0.6, label="|e_y|")
    axes[0].plot(t, np.abs(log["err_pos"][:, 2]), "b-", linewidth=0.7, alpha=0.6, label="|e_z|")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("position error (m)")
    axes[0].set_title("Position error")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(t, err_rot_norm, "k-", linewidth=1.1, label="‖e_R‖")
    axes[1].plot(t, np.abs(log["err_rot"][:, 0]), "r-", linewidth=0.7, alpha=0.6, label="|roll|")
    axes[1].plot(t, np.abs(log["err_rot"][:, 1]), "g-", linewidth=0.7, alpha=0.6, label="|pitch|")
    axes[1].plot(t, np.abs(log["err_rot"][:, 2]), "b-", linewidth=0.7, alpha=0.6, label="|yaw|")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("rotation error (rad)")
    axes[1].set_title("Rotation error")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_error.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_covariance(log):
    t = log["t"]
    std_pos = np.sqrt(log["P_pos_diag"])
    std_rot = np.sqrt(log["P_rot_diag"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(t, std_pos[:, 0], "r-", linewidth=1.0, label="σ_x")
    axes[0].plot(t, std_pos[:, 1], "g-", linewidth=1.0, label="σ_y")
    axes[0].plot(t, std_pos[:, 2], "b-", linewidth=1.0, label="σ_z")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("position std (m)")
    axes[0].set_title("Filter position covariance (per-axis std)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(t, std_rot[:, 0], "r-", linewidth=1.0, label="σ_roll")
    axes[1].plot(t, std_rot[:, 1], "g-", linewidth=1.0, label="σ_pitch")
    axes[1].plot(t, std_rot[:, 2], "b-", linewidth=1.0, label="σ_yaw")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("rotation std (rad)")
    axes[1].set_title("Filter rotation covariance (per-axis std)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_covariance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_nees(log):
    t = log["t"]
    nees_p = log["nees_pos"]
    nees_r = log["nees_rot"]
    lo, hi = chi2.ppf([0.025, 0.975], df=3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, vals, title in [
        (axes[0], nees_p, "Position NEES (3-DOF)"),
        (axes[1], nees_r, "Rotation NEES (3-DOF)"),
    ]:
        ax.plot(t, vals, "b-", linewidth=0.8)
        ax.axhline(3.0, color="k", linestyle="-", linewidth=1.0, label="expected (3)")
        ax.axhline(lo, color="g", linestyle="--", linewidth=1.0, label=f"95% lower ({lo:.2f})")
        ax.axhline(hi, color="r", linestyle="--", linewidth=1.0, label=f"95% upper ({hi:.2f})")
        finite = vals[np.isfinite(vals)]
        if len(finite):
            ax.text(0.02, 0.95, f"mean={finite.mean():.2f}", transform=ax.transAxes,
                    va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("NEES")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{TAG}_nees.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def summarize(log):
    err_pos_norm = np.linalg.norm(log["err_pos"], axis=1)
    err_rot_norm = np.linalg.norm(log["err_rot"], axis=1)
    nees_p = log["nees_pos"][np.isfinite(log["nees_pos"])]
    nees_r = log["nees_rot"][np.isfinite(log["nees_rot"])]
    lo, hi = chi2.ppf([0.025, 0.975], df=3)

    print(f"\n[Bucket #1 — FAST-LIO2 baseline IESKF, env={ENVIRONMENT}]")
    print(f"  Position error : mean={err_pos_norm.mean():.4f} m  "
          f"max={err_pos_norm.max():.4f} m  final={err_pos_norm[-1]:.4f} m")
    print(f"  Rotation error : mean={err_rot_norm.mean():.4f} rad  "
          f"max={err_rot_norm.max():.4f} rad  final={err_rot_norm[-1]:.4f} rad")
    print(f"  NEES pos        : mean={nees_p.mean():.3f}  in 95%=[{lo:.2f},{hi:.2f}] : "
          f"{np.mean((nees_p>=lo)&(nees_p<=hi)):.1%}")
    print(f"  NEES rot        : mean={nees_r.mean():.3f}  in 95%=[{lo:.2f},{hi:.2f}] : "
          f"{np.mean((nees_r>=lo)&(nees_r<=hi)):.1%}")


if __name__ == "__main__":
    print(f"Loading simulated data from {OUT_DIR}...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
    lidar_data = list(lidar_arr)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans, "
          f"IMU rate={IMU_RATE} Hz")

    print("Running baseline IESKF...")
    np.random.seed(123)
    log = run_baseline(imu_data, lidar_data)

    summarize(log)
    save_csvs(log)
    plot_trajectory(log)
    plot_error(log)
    plot_covariance(log)
    plot_nees(log)

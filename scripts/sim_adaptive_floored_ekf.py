"""Script G: Single-filter adaptive EKF with PCRB-floored published covariance.

THE GOAL — three properties from one filter:
  1. Pure odometry → published P grows monotonically (CRLB spirit).
  2. Measurements → P can shrink, but never below information-theoretic floor.
  3. Honest — published P ≈ actual MSE (NEES ≈ DOF), without hand-tuned R/Q.

THE RECIPE:
  - Run a standard EKF/IESKF as usual on x_k = [tx, ty, theta].
  - Adaptive R̂_k: variational update from innovation residuals (Sage-Husa
    style with smoothing), so R is not hand-tuned.
  - PCRB recursion (Tichavský) maintained inside the filter cycle:
        J_{k+1} = (Q + F J_k^-1 F^T)^-1 + H^T R̂^-1 H
  - Published cov:  P_pub_k = PSD-max(P_ekf_k, J_k^-1).
        "PSD max" = the smallest M such that M ≽ A and M ≽ B (Loewner order),
        computed via eigendecomposition of (A − B).

Pure odometry: J_{k+1} = (Q + F J_k^-1 F^T)^-1 → J_k^-1 grows by Q each step.
So P_pub grows in dead reckoning. With measurements, J grows → J^-1 shrinks,
but P_pub is upper-bounded only by the actual information available, never
artificially loose.

Test scenario:
  - Constant-velocity trajectory through a room with walls.
  - Lidar measurements in windows [0, 4s] and [12, 16s].
  - Pure odometry in window [4, 12s].
  - MC trials. Show NEES is consistent in all phases.

Run: python sim_adaptive_floored_ekf.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)

N_DIM = 3  # SE(2): [tx, ty, theta]
WALLS = np.array([
    [1.0, 0.0, -5.0],
    [-1.0, 0.0, -5.0],
    [0.0, 1.0, -5.0],
    [0.0, -1.0, -5.0],
])
DT = 0.1
N_STEPS = 200
V_NOMINAL = np.array([0.10, 0.05, 0.02])  # constant nominal velocity
Q_TRUE = np.diag([0.02, 0.02, 0.01]) ** 2

N_POINTS = 14
SIGMA_POINT_TRUE = 0.20
SIGMA_POINT_INIT = 0.40  # filter starts with WRONG R guess (over-conservative)

# Measurement windows
MEAS_WINDOWS = [(0.0, 4.0), (12.0, 16.0)]

# Adaptive R̂ smoothing
ADAPT_R_ALPHA = 0.08   # exponential smoothing weight on new residual cov
ADAPT_R_FLOOR = 1e-4   # never drop R below this
ADAPT_R_CEIL = 4.0     # never raise above this

N_MC = 200


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_body_points_at(x, seed=42):
    tx, ty, theta = x
    sensor_world = np.array([tx, ty])
    R = rot2(theta)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=N_POINTS)
    body = np.zeros((N_POINTS, 2))
    for i, a in enumerate(angles):
        dir_world = np.array([np.cos(a), np.sin(a)])
        best_t = np.inf
        for n0, n1, d in WALLS:
            n = np.array([n0, n1])
            denom = n @ dir_world
            if abs(denom) < 1e-9:
                continue
            t = -(n @ sensor_world + d) / denom
            if 1e-3 < t < best_t:
                best_t = t
        if not np.isfinite(best_t):
            best_t = 10.0
        hit_world = sensor_world + best_t * dir_world
        body[i] = R.T @ (hit_world - sensor_world)
    return body


def residuals_and_jac(x, body_points):
    tx, ty, theta = x
    R = rot2(theta)
    pw = body_points @ R.T + np.array([tx, ty])
    N = body_points.shape[0]
    r = np.zeros(N)
    J = np.zeros((N, 3))
    c, s = np.cos(theta), np.sin(theta)
    for i in range(N):
        best_abs = np.inf
        best_signed = 0.0
        best_n = (0.0, 0.0)
        for n0, n1, d in WALLS:
            sd = n0 * pw[i, 0] + n1 * pw[i, 1] + d
            if abs(sd) < best_abs:
                best_abs = abs(sd); best_signed = sd; best_n = (n0, n1)
        r[i] = best_signed
        n0, n1 = best_n
        px, py = body_points[i]
        dRp = np.array([-s * px - c * py, c * px - s * py])
        J[i, 0] = n0; J[i, 1] = n1; J[i, 2] = n0 * dRp[0] + n1 * dRp[1]
    return r, J


def loewner_max(A, B):
    """Smallest PSD matrix C such that C ≽ A and C ≽ B.

    Computed in basis where (A − B) is diagonal: C = A + (B − A)_+ where (·)_+
    keeps positive eigenvalues only.
    """
    D = B - A
    w, V = np.linalg.eigh(D)
    w_pos = np.maximum(w, 0.0)
    return A + V @ np.diag(w_pos) @ V.T


def measurement_in_window(t):
    return any(t0 <= t <= t1 for t0, t1 in MEAS_WINDOWS)


def simulate_truth(rng):
    xs = np.zeros((N_STEPS + 1, N_DIM))
    xs[0] = np.zeros(N_DIM)
    for k in range(N_STEPS):
        w = rng.multivariate_normal(np.zeros(N_DIM), Q_TRUE)
        xs[k + 1] = xs[k] + DT * V_NOMINAL + w
    return xs


def run_one_trial(seed, truth=None, r_mode="adaptive"):
    """r_mode: 'adaptive' (default), 'fixed_wrong' (σ=SIGMA_POINT_INIT),
    or 'fixed_truth' (σ=SIGMA_POINT_TRUE — the oracle we shouldn't have)."""
    if truth is None:
        truth_rng = np.random.default_rng(seed * 31 + 11)
        truth = simulate_truth(truth_rng)
    rng = np.random.default_rng(seed)
    P0 = np.diag([0.1, 0.1, 0.05]) ** 2

    x_ekf = truth[0] + rng.multivariate_normal(np.zeros(N_DIM), P0)
    P_ekf = P0.copy()
    if r_mode == "fixed_truth":
        sigma_r_hat = SIGMA_POINT_TRUE
    else:
        sigma_r_hat = SIGMA_POINT_INIT
    J_pcrb = np.linalg.inv(P0)

    estimates = np.zeros((N_STEPS + 1, N_DIM))
    P_pub_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    P_ekf_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    P_pcrb_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    sigma_r_hist = np.zeros(N_STEPS + 1)

    estimates[0] = x_ekf
    P_pub_full[0] = P0
    P_ekf_full[0] = P0
    P_pcrb_full[0] = P0
    sigma_r_hist[0] = sigma_r_hat
    truth_out = truth  # return per-trial truth so MC can use it

    for k in range(N_STEPS):
        t = (k + 1) * DT

        # EKF predict (F = I for the position-heading dynamics here).
        x_ekf = x_ekf + DT * V_NOMINAL
        P_ekf = P_ekf + Q_TRUE
        # PCRB predict.
        J_pcrb = np.linalg.inv(np.linalg.inv(J_pcrb) + Q_TRUE)

        # Measurement update only inside windows.
        if measurement_in_window(t):
            body_true = make_body_points_at(truth[k + 1])
            body_noisy = body_true + rng.normal(scale=SIGMA_POINT_TRUE,
                                                size=body_true.shape)
            r_residual, H = residuals_and_jac(x_ekf, body_noisy)
            R_meas = (sigma_r_hat ** 2) * np.eye(len(r_residual))
            S = H @ P_ekf @ H.T + R_meas
            K = P_ekf @ H.T @ np.linalg.inv(S)
            x_ekf = x_ekf - K @ r_residual
            P_ekf = (np.eye(N_DIM) - K @ H) @ P_ekf

            if r_mode == "adaptive":
                r_post, _ = residuals_and_jac(x_ekf, body_noisy)
                emp_var = float(np.var(r_post))
                new_sigma_sq = max(ADAPT_R_FLOOR, min(ADAPT_R_CEIL, emp_var))
                sigma_r_hat = float(np.sqrt(
                    (1 - ADAPT_R_ALPHA) * sigma_r_hat ** 2
                    + ADAPT_R_ALPHA * new_sigma_sq
                ))

            # PCRB measurement update using current R̂.
            inv_R = 1.0 / (sigma_r_hat ** 2)
            J_pcrb = J_pcrb + inv_R * (H.T @ H)

        # Published covariance: PSD-max(P_ekf, PCRB).
        P_pcrb = np.linalg.inv(J_pcrb)
        P_pub = loewner_max(P_ekf, P_pcrb)

        estimates[k + 1] = x_ekf
        P_ekf_full[k + 1] = P_ekf
        P_pcrb_full[k + 1] = P_pcrb
        P_pub_full[k + 1] = P_pub
        sigma_r_hist[k + 1] = sigma_r_hat

    return estimates, P_pub_full, P_ekf_full, P_pcrb_full, sigma_r_hist, truth_out


def main():
    print("Running {} MC trials × 3 R modes (each with its own truth path)...".format(N_MC))

    runs = {}
    for mode in ("fixed_wrong", "fixed_truth", "adaptive"):
        est_all = np.zeros((N_MC, N_STEPS + 1, N_DIM))
        truth_all = np.zeros((N_MC, N_STEPS + 1, N_DIM))
        P_pub_all = np.zeros((N_MC, N_STEPS + 1, N_DIM, N_DIM))
        P_ekf_all = np.zeros((N_MC, N_STEPS + 1, N_DIM, N_DIM))
        P_pcrb_all = np.zeros((N_MC, N_STEPS + 1, N_DIM, N_DIM))
        sigma_r_all = np.zeros((N_MC, N_STEPS + 1))
        for m in range(N_MC):
            est, Pp, Pe, Pc, sr, tr = run_one_trial(seed=1000 + m, r_mode=mode)
            est_all[m] = est; truth_all[m] = tr
            P_pub_all[m] = Pp; P_ekf_all[m] = Pe; P_pcrb_all[m] = Pc
            sigma_r_all[m] = sr
        runs[mode] = {
            "est": est_all, "truth": truth_all, "P_pub": P_pub_all,
            "P_ekf": P_ekf_all, "P_pcrb": P_pcrb_all, "sigma_r": sigma_r_all,
        }

    print("\n=== Position RMSE per phase (lower is better) ===")
    t_axis = np.arange(N_STEPS + 1) * DT
    phase_masks = {
        "meas (0-4s)":   (t_axis >= 0.0)  & (t_axis <= 4.0),
        "odom (4-12s)":  (t_axis >= 4.0)  & (t_axis <= 12.0),
        "meas (12-16s)": (t_axis >= 12.0) & (t_axis <= 16.0),
        "odom (16-20s)": (t_axis >= 16.0) & (t_axis <= 20.0),
        "OVERALL":       np.ones(N_STEPS + 1, dtype=bool),
    }
    header = f"  {'phase':18s} | " + " | ".join(f"{m:>14s}" for m in runs.keys())
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, mask in phase_masks.items():
        row = f"  {name:18s} | "
        cells = []
        for mode in runs:
            d = runs[mode]
            err_pos = np.linalg.norm(d["est"][:, :, 0:2] - d["truth"][:, :, 0:2], axis=2)
            rmse = float(np.sqrt(np.mean(err_pos[:, mask] ** 2)))
            cells.append(f"{rmse:>14.4f}")
        row += " | ".join(cells)
        print(row)

    # Set up plotting vars from adaptive run for the existing plot block.
    estimates_all = runs["adaptive"]["est"]
    truth_all = runs["adaptive"]["truth"]
    P_pub_all = runs["adaptive"]["P_pub"]
    P_ekf_all = runs["adaptive"]["P_ekf"]
    P_pcrb_all = runs["adaptive"]["P_pcrb"]
    sigma_r_all = runs["adaptive"]["sigma_r"]

    # Empirical MSE per step using per-trial truth.
    mse = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    for k in range(N_STEPS + 1):
        eps = estimates_all[:, k, :] - truth_all[:, k, :]
        mse[k] = (eps.T @ eps) / N_MC

    # NEES per step using each trial's own truth + each trial's own P.
    nees_pub = np.zeros(N_STEPS + 1)
    nees_ekf = np.zeros(N_STEPS + 1)
    for k in range(N_STEPS + 1):
        per_trial_pub = np.zeros(N_MC)
        per_trial_ekf = np.zeros(N_MC)
        for m in range(N_MC):
            eps = estimates_all[m, k] - truth_all[m, k]
            try:
                per_trial_pub[m] = eps @ np.linalg.solve(P_pub_all[m, k], eps)
                per_trial_ekf[m] = eps @ np.linalg.solve(P_ekf_all[m, k], eps)
            except np.linalg.LinAlgError:
                per_trial_pub[m] = np.nan
                per_trial_ekf[m] = np.nan
        nees_pub[k] = np.nanmean(per_trial_pub)
        nees_ekf[k] = np.nanmean(per_trial_ekf)
    truth = truth_all[0]  # for plotting only

    # Print summary per phase.
    t_axis = np.arange(N_STEPS + 1) * DT
    phase_masks = {
        "meas (0-4s)":  (t_axis >= 0.0)  & (t_axis <= 4.0),
        "odom (4-12s)": (t_axis >= 4.0)  & (t_axis <= 12.0),
        "meas (12-16s)": (t_axis >= 12.0) & (t_axis <= 16.0),
        "odom (16-20s)": (t_axis >= 16.0) & (t_axis <= 20.0),
    }
    print("\n=== NEES per phase (target = {}) ===".format(N_DIM))
    for name, mask in phase_masks.items():
        print(f"  {name:18s}: NEES(P_pub) = {nees_pub[mask].mean():.3f}   "
              f"NEES(P_ekf only) = {nees_ekf[mask].mean():.3f}")
    print(f"\nσ_r adaptive history: init={SIGMA_POINT_INIT:.3f}  "
          f"converged≈{sigma_r_all[:, -1].mean():.3f}  "
          f"truth={SIGMA_POINT_TRUE:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", linewidth=1.4, label="truth")
    for m in range(min(8, N_MC)):
        ax.plot(estimates_all[m, :, 0], estimates_all[m, :, 1],
                color="tab:blue", alpha=0.3, linewidth=0.7)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Trajectory + sample EKF runs")
    ax.set_aspect("equal"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    P_pub_mean = P_pub_all.mean(axis=0)
    P_ekf_mean = P_ekf_all.mean(axis=0)
    P_pcrb_mean = P_pcrb_all.mean(axis=0)
    ax.plot(t_axis, np.array([np.trace(P_pub_mean[k]) for k in range(N_STEPS + 1)]),
            "r-", linewidth=1.5, label="trace(P_pub) — PSD max")
    ax.plot(t_axis, np.array([np.trace(P_ekf_mean[k]) for k in range(N_STEPS + 1)]),
            "b-", linewidth=1.2, label="trace(P_ekf)")
    ax.plot(t_axis, np.array([np.trace(P_pcrb_mean[k]) for k in range(N_STEPS + 1)]),
            "g-", linewidth=1.2, label="trace(PCRB floor)")
    ax.plot(t_axis, np.array([np.trace(mse[k]) for k in range(N_STEPS + 1)]),
            "k--", linewidth=1.6, label="trace(empirical MSE)")
    for (t0, t1) in MEAS_WINDOWS:
        ax.axvspan(t0, t1, alpha=0.15, color="tab:green", label=None)
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("trace")
    ax.set_title("Cov trace over time (green = measurement windows)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.plot(t_axis, sigma_r_all.mean(axis=0), "tab:purple", linewidth=1.5,
            label="σ̂_r (adaptive)")
    ax.axhline(SIGMA_POINT_TRUE, color="black", linestyle="--", linewidth=1.0,
               label=f"truth = {SIGMA_POINT_TRUE}")
    ax.axhline(SIGMA_POINT_INIT, color="tab:red", linestyle=":", linewidth=1.0,
               label=f"initial guess = {SIGMA_POINT_INIT}")
    for (t0, t1) in MEAS_WINDOWS:
        ax.axvspan(t0, t1, alpha=0.15, color="tab:green")
    ax.set_xlabel("t (s)"); ax.set_ylabel("σ̂_r")
    ax.set_title("Adaptive measurement-noise estimate")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.plot(t_axis, nees_pub, "r-", linewidth=1.2, label="NEES vs P_pub")
    ax.plot(t_axis, nees_ekf, "b--", linewidth=1.0, label="NEES vs P_ekf alone")
    ax.axhline(N_DIM, color="black", linestyle="-", linewidth=0.8,
               label=f"target = {N_DIM}")
    for (t0, t1) in MEAS_WINDOWS:
        ax.axvspan(t0, t1, alpha=0.15, color="tab:green")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("NEES")
    ax.set_title("NEES consistency")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_adaptive_floored_ekf.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

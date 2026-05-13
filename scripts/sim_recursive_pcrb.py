"""Script D: recursive bias-corrected PCRB — unified P_drift replacement recipe.

Combines:
  - Script 1 / C: MSE = Cov + b bᵀ for biased estimators.
  - Script 2: PCRB Tichavský recursion (the right recursive bound).
  - Script B: online bias estimation via the self-consistent predictor.

Setup is a dynamic SE(2) tracking problem with point-to-plane lidar against
fixed walls — same flavor as the real filter but in the plane.

State: x_k = [tx, ty, theta], constant-velocity nominal motion + process noise.
Measurement: K point-to-plane residuals against known walls each step.

At each step we maintain four covariances in parallel:
  P_ekf       : the EKF/IESKF Bayesian posterior (what the filter publishes)
  P_pcrb_std  : standard PCRB (Tichavský recursion), unbiased CRLB analog
  P_pcrb_bias : bias-corrected PCRB recursion: J_k tracked + recursive b_k
                and the MSE form  (I + ∂b/∂x) J_k^-1 (I + ∂b/∂x)^T + b_k b_k^T
  P_drift_old : single-step CRLB accumulator — what the current filter does

Run: python sim_recursive_pcrb.py
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

# Walls (planes in 2D world)
WALLS = np.array([
    [1.0, 0.0, -3.0],
    [-1.0, 0.0, -3.0],
    [0.0, 1.0, -3.0],
    [0.0, -1.0, -3.0],
    [np.sqrt(0.5), np.sqrt(0.5), -3.5],  # diagonal wall to break symmetry
])

DT = 1.0
V_NOMINAL_WORLD = np.array([0.05, 0.03, 0.01])  # gentle motion to keep linearization valid
Q = np.diag([0.02, 0.02, 0.01]) ** 2

N_POINTS = 16
SIGMA_POINT = 0.10  # cleaner sensor → EKF less prone to linearization runaway

N_STEPS = 30
N_MC = 300
MAX_ITERS = 8


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def step_dynamics(x, w):
    """Constant-velocity in world frame plus process noise."""
    return x + DT * V_NOMINAL_WORLD + w


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
            best_t = 5.0
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


def measurement_FIM(x, body_points):
    """Sum_i H_i^T R^-1 H_i for the residuals at x — the per-step measurement
    information matrix the PCRB recursion needs."""
    _, J = residuals_and_jac(x, body_points)
    return (1.0 / SIGMA_POINT ** 2) * (J.T @ J)


def ekf_step(x_prev, P_prev, body_noisy):
    """Predict + iterated MAP update."""
    x_pred = step_dynamics(x_prev, np.zeros(N_DIM))
    P_pred = P_prev + Q
    P_pred_inv = np.linalg.inv(P_pred)

    x = x_pred.copy()
    inv_var = 1.0 / (SIGMA_POINT ** 2)
    for _ in range(MAX_ITERS):
        r, J = residuals_and_jac(x, body_noisy)
        H = inv_var * (J.T @ J) + P_pred_inv
        g = inv_var * (J.T @ r) + P_pred_inv @ (x - x_pred)
        dx = -np.linalg.solve(H, g)
        x = x + dx
        if np.linalg.norm(dx) < 1e-7:
            break
    P_new = np.linalg.inv(H)
    return x, P_new, x_pred, P_pred


def pcrb_recursion_step(J_prev, x_true_next, body_at_x_true):
    """Standard Tichavský PCRB: J_{k+1} = (Q + F J_k^-1 F^T)^-1 + H^T R^-1 H.
    Here F = I (the dynamics adds a known constant), so:
        J_{k+1} = (J_k^-1 + Q)^-1 + measurement_FIM(x_true_next).
    """
    pred = np.linalg.inv(np.linalg.inv(J_prev) + Q)
    meas = measurement_FIM(x_true_next, body_at_x_true)
    return pred + meas


def predicted_bias(x_hat, mu_prior, P_pred, body_noisy):
    """Self-consistent online bias predictor from script B, generalized:
        b̂ = (H^T R^-1 H + P_pred^-1)^-1 P_pred^-1 (mu_prior - x_hat)
    """
    _, J = residuals_and_jac(x_hat, body_noisy)
    meas_info = (1.0 / SIGMA_POINT ** 2) * (J.T @ J)
    info_post = meas_info + np.linalg.inv(P_pred)
    return np.linalg.inv(info_post) @ np.linalg.inv(P_pred) @ (mu_prior - x_hat)


def main():
    # Common ground truth trajectory.
    rng_truth = np.random.default_rng(7)
    truth = np.zeros((N_STEPS + 1, N_DIM))
    truth[0] = np.array([0.0, 0.0, 0.0])
    for k in range(N_STEPS):
        w = rng_truth.multivariate_normal(np.zeros(N_DIM), Q)
        truth[k + 1] = step_dynamics(truth[k], w)

    # Storage for MC EKF runs.
    ekf_estimates = np.zeros((N_MC, N_STEPS + 1, N_DIM))
    ekf_P = np.zeros((N_MC, N_STEPS + 1, N_DIM, N_DIM))
    biases_predicted = np.zeros((N_MC, N_STEPS + 1, N_DIM))

    P0 = np.diag([0.10, 0.10, 0.05]) ** 2

    print("Running {} MC EKF trials over {} steps...".format(N_MC, N_STEPS))
    # Initial offset: random per trial, zero-mean — models a well-calibrated
    # filter (offset drawn from N(0, P0)). The systematic-bias regime is left
    # to a separate experiment (it's not what a typical FAST-LIO start looks like).
    for m in range(N_MC):
        rng = np.random.default_rng(1000 + m)
        x = truth[0] + rng.multivariate_normal(np.zeros(N_DIM), P0)
        P = P0.copy()
        ekf_estimates[m, 0] = x
        ekf_P[m, 0] = P
        for k in range(N_STEPS):
            body_true = make_body_points_at(truth[k + 1])
            body_noisy = body_true + rng.normal(scale=SIGMA_POINT, size=body_true.shape)
            x_new, P_new, x_pred, P_pred = ekf_step(x, P, body_noisy)
            b_hat = predicted_bias(x_new, x_pred, P_pred, body_noisy)
            biases_predicted[m, k + 1] = b_hat
            x = x_new; P = P_new
            ekf_estimates[m, k + 1] = x
            ekf_P[m, k + 1] = P

    # PCRB recursion (deterministic, runs once along ground truth).
    J_pcrb = np.linalg.inv(P0)
    pcrb_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    pcrb_full[0] = P0
    for k in range(N_STEPS):
        body_true = make_body_points_at(truth[k + 1])
        J_pcrb = pcrb_recursion_step(J_pcrb, truth[k + 1], body_true)
        pcrb_full[k + 1] = np.linalg.inv(J_pcrb)

    # Single-step CRLB accumulator (current `P_drift` analog): sums per-step
    # measurement-only inverse FIMs.
    p_drift_old = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    p_drift_old[0] = P0
    for k in range(N_STEPS):
        body_true = make_body_points_at(truth[k + 1])
        meas = measurement_FIM(truth[k + 1], body_true)
        # Per-step single CRLB = meas^-1 (with tiny floor for stability).
        single_step = np.linalg.inv(meas + 1e-9 * np.eye(N_DIM))
        p_drift_old[k + 1] = p_drift_old[k] + single_step

    # Empirical MSE per step and empirical bias per step.
    empirical_mse = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    empirical_bias = np.zeros((N_STEPS + 1, N_DIM))
    for k in range(N_STEPS + 1):
        eps = ekf_estimates[:, k, :] - truth[k][None, :]  # (N_MC, 3)
        empirical_mse[k] = (eps.T @ eps) / N_MC
        empirical_bias[k] = eps.mean(axis=0)

    # Bias-corrected PCRB: PCRB + (mean predicted bias) (mean predicted bias)^T
    mean_predicted_bias = biases_predicted.mean(axis=0)
    pcrb_bias_mse = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    for k in range(N_STEPS + 1):
        b = mean_predicted_bias[k]
        pcrb_bias_mse[k] = pcrb_full[k] + np.outer(b, b)

    # NEES against each candidate published cov.
    nees = {name: np.zeros(N_STEPS + 1) for name in
            ["EKF P", "PCRB (unbiased)", "PCRB + b̂ b̂^T (MSE)",
             "P_drift (old single-step)"]}
    for k in range(N_STEPS + 1):
        eps = ekf_estimates[:, k, :] - truth[k][None, :]
        P_ekf_mean = ekf_P[:, k].mean(axis=0)
        for name, P in [("EKF P", P_ekf_mean),
                        ("PCRB (unbiased)", pcrb_full[k]),
                        ("PCRB + b̂ b̂^T (MSE)", pcrb_bias_mse[k]),
                        ("P_drift (old single-step)", p_drift_old[k])]:
            try:
                Pinv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                nees[name][k] = np.nan
                continue
            nees[name][k] = np.einsum("ni,ij,nj->n", eps, Pinv, eps).mean()

    # Summary over the second half (skip transient).
    half = N_STEPS // 2
    print("\n=== Mean NEES over steps {}..{} (target = {} ± ~{:.2f}) ===".format(
        half, N_STEPS, N_DIM, 1.96 * np.sqrt(2 * N_DIM / N_MC)))
    for name, arr in nees.items():
        m = np.nanmean(arr[half:])
        if m > N_DIM * 1.1:
            verdict = "OVERCONFIDENT"
        elif m < N_DIM * 0.9:
            verdict = "conservative"
        else:
            verdict = "consistent"
        print("  {:30s}: mean NEES = {:7.3f}   [{}]".format(name, m, verdict))

    # Compare predicted bias trace to empirical bias trace.
    bias_pred_norm = np.linalg.norm(mean_predicted_bias, axis=1)
    bias_emp_norm = np.linalg.norm(empirical_bias, axis=1)
    print("\nFinal-step bias norms:")
    print("  empirical (MC truth) = {:.4f}".format(bias_emp_norm[-1]))
    print("  predicted (SC mean)  = {:.4f}".format(bias_pred_norm[-1]))
    print("  ratio                = {:.3f}".format(
        bias_pred_norm[-1] / max(bias_emp_norm[-1], 1e-9)))

    # Plot
    t = np.arange(N_STEPS + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", linewidth=1.6, label="truth")
    for m in range(min(8, N_MC)):
        ax.plot(ekf_estimates[m, :, 0], ekf_estimates[m, :, 1],
                color="tab:blue", alpha=0.3, linewidth=0.8)
    ax.scatter(WALLS[:, 0] * (-WALLS[:, 2]) / np.maximum(np.linalg.norm(WALLS[:, :2], axis=1) ** 2, 1e-9),
               WALLS[:, 1] * (-WALLS[:, 2]) / np.maximum(np.linalg.norm(WALLS[:, :2], axis=1) ** 2, 1e-9),
               marker="s", color="red", s=60, label="wall anchors")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Trajectory + sample EKF runs"); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.plot(t, np.array([np.trace(empirical_mse[k]) for k in range(N_STEPS + 1)]),
            "k-", linewidth=2.0, label="empirical MSE")
    ax.plot(t, np.array([np.trace(pcrb_full[k]) for k in range(N_STEPS + 1)]),
            "g-", linewidth=1.5, label="PCRB (recursive)")
    ax.plot(t, np.array([np.trace(pcrb_bias_mse[k]) for k in range(N_STEPS + 1)]),
            color="tab:purple", linewidth=1.5, label="PCRB + b̂ b̂^T")
    ax.plot(t, np.array([np.trace(ekf_P[:, k].mean(axis=0)) for k in range(N_STEPS + 1)]),
            "b-", linewidth=1.2, label="EKF P (mean)")
    ax.plot(t, np.array([np.trace(p_drift_old[k]) for k in range(N_STEPS + 1)]),
            color="orange", linewidth=1.2, label="P_drift (single-step accum)")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("trace(P)")
    ax.set_title("Cov trace vs time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    ax.plot(t, bias_emp_norm, "k-", linewidth=1.6, label="||b|| empirical (MC)")
    ax.plot(t, bias_pred_norm, color="tab:purple", linewidth=1.2,
            label="||b̂|| predicted (SC mean)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("bias norm")
    ax.set_title("Empirical vs predicted bias norm")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    lo = N_DIM - 1.96 * np.sqrt(2 * N_DIM / N_MC)
    hi = N_DIM + 1.96 * np.sqrt(2 * N_DIM / N_MC)
    for name, color in [("EKF P", "tab:blue"),
                        ("PCRB (unbiased)", "tab:green"),
                        ("PCRB + b̂ b̂^T (MSE)", "tab:purple"),
                        ("P_drift (old single-step)", "tab:orange")]:
        ax.plot(t, nees[name], color=color, linewidth=1.2, label=name)
    ax.axhline(N_DIM, color="black", linestyle="-", linewidth=0.8,
               label=f"target ({N_DIM})")
    ax.fill_between(t, lo, hi, color="gray", alpha=0.2, label="95% band")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("NEES")
    ax.set_title("NEES consistency across published covs")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_recursive_pcrb.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

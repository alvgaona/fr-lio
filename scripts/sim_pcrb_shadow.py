"""Direction 2: Posterior Cramér-Rao Bound (PCRB) as a shadow filter alongside an EKF.

Script 1 showed that the standard CRLB is the wrong bound to publish for a
biased estimator, and that the MSE form (Cov + bbᵀ) is what NEES actually
tests. This script makes that picture recursive: instead of a single static
update, we run a dynamic 2D position tracker and accumulate a *Posterior*
Cramér-Rao bound (Tichavský/Muravchik/Nehorai 1998) step-by-step, alongside
the EKF.

The PCRB recursion for x_{k+1} = F x_k + w_k, z_k = h(x_k) + v_k is

    J_{k+1} = (Q + F J_k^-1 F^T)^-1 + E[H_{k+1}^T R^-1 H_{k+1}]

The PCRB on MSE at step k is J_k^-1. It depends only on the model
(dynamics + measurement Jacobians + noise covariances), not on any specific
estimator — it's the information-theoretic floor for any estimator that
processes the same measurements.

Setup:
  - 2D position state x_k ∈ R^2, constant-velocity motion (velocity known).
  - K beacons at known positions; measurement z_i = ||x - b_i|| + N(0, sigma_r^2).
    Range is nonlinear in position → EKF is biased, same flavor as
    point-to-plane in our real filter.
  - Process noise w_k ~ N(0, Q).
  - Run N_MC Monte Carlo trials. At each step, collect:
      * EKF estimate and covariance P_ekf
      * Standard CRLB at this step: (sum_i H_i^T R^-1 H_i)^-1
        (single-step measurement FIM only — what "P_drift accumulation" mimics)
      * PCRB J_k^-1 (recursive)
      * Empirical MSE = E[(x̂ - x_true)(x̂ - x_true)^T]

Outputs (in sim_data/):
  - sim_pcrb_shadow.png: trace(P) vs time for each form, plus a NEES panel.
  - Console summary of NEES consistency for each form.

Run: python sim_pcrb_shadow.py
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

N_DIM = 2

# Time / dynamics
DT = 0.5
N_STEPS = 60
V_NOMINAL = np.array([0.5, 0.2])  # known constant velocity (m/s)
Q = np.diag([0.02, 0.02]) ** 2  # process noise cov (m^2)
F = np.eye(N_DIM)  # x_{k+1} = x_k + DT * V + w; state-transition for the position is identity

# Beacons (range sensors at known fixed positions).
# Poor-geometry placement: only 2 beacons, both on the same side of the
# trajectory, so the y-direction is strongly observable but the x-direction
# is weakly observable and nonlinear. This is the regime where EKF
# linearization bias matters and where the PCRB should sit visibly above the
# (overconfident) EKF P.
BEACONS = np.array([
    [-1.0, -4.0],
    [1.0, -4.0],
])
N_BEACONS = BEACONS.shape[0]
SIGMA_R = 1.5  # range measurement noise std (m) — large relative to geometry
R_MEAS = (SIGMA_R ** 2) * np.eye(N_BEACONS)
R_INV = np.linalg.inv(R_MEAS)

# Initial state and prior
X0_TRUE = np.array([0.0, 0.0])
P0 = np.diag([0.5, 0.5]) ** 2

# Monte Carlo
N_MC = 1000
RNG = np.random.default_rng(0)


def predict_mean(x):
    return x + DT * V_NOMINAL


def range_h(x, beacons=BEACONS):
    """h_i(x) = ||x - b_i||."""
    diffs = x[None, :] - beacons  # (K, 2)
    return np.linalg.norm(diffs, axis=1)


def range_jacobian(x, beacons=BEACONS):
    """H_i = (x - b_i)^T / ||x - b_i||  (1 x 2 per beacon)."""
    diffs = x[None, :] - beacons
    dists = np.linalg.norm(diffs, axis=1, keepdims=True)
    dists = np.maximum(dists, 1e-9)
    return diffs / dists  # (K, 2)


def ekf_step(x_prev, P_prev, z, Q_assumed=Q):
    """One EKF cycle: predict + measurement update with range Jacobian at the
    predicted mean. Q_assumed is the process noise the EKF *believes* — pass
    a smaller-than-true Q to simulate an overconfident filter."""
    x_pred = predict_mean(x_prev)
    P_pred = F @ P_prev @ F.T + Q_assumed

    H = range_jacobian(x_pred)
    y = z - range_h(x_pred)
    S = H @ P_pred @ H.T + R_MEAS
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(N_DIM) - K @ H) @ P_pred
    return x_new, P_new


def pcrb_step(J_prev, x_true):
    """Tichavský recursion for J_{k+1}.

    J_{k+1} = (Q + F J_k^-1 F^T)^-1 + H^T R^-1 H

    The measurement information term is evaluated at the true state. This is
    the standard PCRB practice — the bound is a function of the true
    trajectory (a property of the problem, not the estimator).
    """
    J_prev_inv = np.linalg.inv(J_prev)
    pred_info = np.linalg.inv(Q + F @ J_prev_inv @ F.T)
    H = range_jacobian(x_true)
    meas_info = H.T @ R_INV @ H
    return pred_info + meas_info


def standard_crlb_at(x_true):
    """Single-step measurement-only CRLB: (sum_i H_i^T R^-1 H_i)^-1.

    Mirrors what the current `P_drift` accumulation does — it sums per-step
    measurement FIMs and ignores both the prior and the dynamics. We compare
    against this to show how much the recursive PCRB differs in shape.
    """
    H = range_jacobian(x_true)
    info = H.T @ R_INV @ H
    return np.linalg.inv(info + 1e-9 * np.eye(N_DIM))


def simulate_truth(rng):
    """Generate one ground-truth trajectory under the dynamics."""
    xs = np.zeros((N_STEPS + 1, N_DIM))
    xs[0] = X0_TRUE
    for k in range(N_STEPS):
        w = rng.multivariate_normal(np.zeros(N_DIM), Q)
        xs[k + 1] = predict_mean(xs[k]) + w
    return xs


def main():
    # Shared ground truth — PCRB and EKF run on the *same* trajectory.
    # The Monte Carlo varies measurement noise (so the EKF sees different z's
    # but the underlying x_true is shared). This is the standard PCRB setup.
    truth = simulate_truth(np.random.default_rng(7))

    # Run two EKFs in parallel: one well-tuned, one overconfident (Q under-estimated).
    Q_well = Q
    # Overconfident: filter believes there is essentially no process noise at all,
    # so it overweights its propagated prior and refuses to widen its covariance.
    Q_over = 1e-6 * np.eye(N_DIM)
    ekf_variants = {
        "EKF well-tuned": Q_well,
        "EKF overconfident": Q_over,
    }

    ekf_estimates = {name: np.zeros((N_MC, N_STEPS + 1, N_DIM)) for name in ekf_variants}
    ekf_P_full = {name: np.zeros((N_MC, N_STEPS + 1, N_DIM, N_DIM)) for name in ekf_variants}

    print("Running {} Monte Carlo trials for each EKF variant...".format(N_MC))
    for m in range(N_MC):
        rng = np.random.default_rng(1000 + m)
        # Same measurement noise stream for both variants — only their Q_assumed differs.
        noise_seq = [rng.normal(scale=SIGMA_R, size=N_BEACONS) for _ in range(N_STEPS)]
        init_offset = rng.multivariate_normal(np.zeros(N_DIM), P0)
        for name, Q_a in ekf_variants.items():
            x = X0_TRUE + init_offset
            P = P0.copy()
            ekf_estimates[name][m, 0] = x
            ekf_P_full[name][m, 0] = P
            for k in range(N_STEPS):
                z = range_h(truth[k + 1]) + noise_seq[k]
                x, P = ekf_step(x, P, z, Q_assumed=Q_a)
                ekf_estimates[name][m, k + 1] = x
                ekf_P_full[name][m, k + 1] = P

    # PCRB recursion (deterministic — runs once, not per trial).
    J = np.linalg.inv(P0)  # initial information
    pcrb_traces = np.zeros(N_STEPS + 1)
    pcrb_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    pcrb_full[0] = P0
    pcrb_traces[0] = np.trace(P0)
    for k in range(N_STEPS):
        J = pcrb_step(J, truth[k + 1])
        pcrb_full[k + 1] = np.linalg.inv(J)
        pcrb_traces[k + 1] = np.trace(pcrb_full[k + 1])

    # Standard single-step CRLB along the trajectory (no recursion).
    std_crlb_traces = np.zeros(N_STEPS + 1)
    std_crlb_full = np.zeros((N_STEPS + 1, N_DIM, N_DIM))
    for k in range(N_STEPS + 1):
        Pk = standard_crlb_at(truth[k])
        std_crlb_full[k] = Pk
        std_crlb_traces[k] = np.trace(Pk)

    # Empirical MSE per variant.
    empirical_mse_full = {name: np.zeros((N_STEPS + 1, N_DIM, N_DIM)) for name in ekf_variants}
    empirical_mse_traces = {name: np.zeros(N_STEPS + 1) for name in ekf_variants}
    for name in ekf_variants:
        for k in range(N_STEPS + 1):
            eps = ekf_estimates[name][:, k, :] - truth[k][None, :]
            empirical_mse_full[name][k] = (eps.T @ eps) / N_MC
            empirical_mse_traces[name][k] = np.trace(empirical_mse_full[name][k])

    # NEES per step for each variant against (its own P) and against PCRB.
    nees = {}
    for name in ekf_variants:
        for label, P_source in [
            (f"{name} vs own P", "own"),
            (f"{name} vs PCRB", "pcrb"),
        ]:
            arr = np.zeros(N_STEPS + 1)
            for k in range(N_STEPS + 1):
                eps = ekf_estimates[name][:, k, :] - truth[k][None, :]
                if P_source == "own":
                    P = ekf_P_full[name][:, k].mean(axis=0)
                else:
                    P = pcrb_full[k]
                Pinv = np.linalg.inv(P)
                arr[k] = np.einsum("ni,ij,nj->n", eps, Pinv, eps).mean()
            nees[label] = arr
    # Also NEES against std CRLB using well-tuned EKF estimates (one curve).
    nees["EKF well-tuned vs Std CRLB"] = np.zeros(N_STEPS + 1)
    for k in range(N_STEPS + 1):
        eps = ekf_estimates["EKF well-tuned"][:, k, :] - truth[k][None, :]
        Pinv = np.linalg.inv(std_crlb_full[k])
        nees["EKF well-tuned vs Std CRLB"][k] = np.einsum("ni,ij,nj->n", eps, Pinv, eps).mean()

    print("\n=== Final step (k={}) covariance traces ===".format(N_STEPS))
    print("  trace(PCRB)                 = {:.4f}".format(pcrb_traces[-1]))
    print("  trace(Std CRLB single-step) = {:.4f}".format(std_crlb_traces[-1]))
    for name in ekf_variants:
        print("  [{}]".format(name))
        print("    trace(emp MSE) = {:.4f}".format(empirical_mse_traces[name][-1]))
        print("    trace(P)       = {:.4f}".format(ekf_P_full[name][:, -1].mean(axis=0).trace()))

    half = N_STEPS // 2
    print("\n=== Mean NEES over second half (target = {} ± ~{:.2f}) ===".format(
        N_DIM, 1.96 * np.sqrt(2 * N_DIM / N_MC)))
    for label, arr in nees.items():
        m = arr[half:].mean()
        if m > N_DIM * 1.1:
            verdict = "OVERCONFIDENT"
        elif m < N_DIM * 0.9:
            verdict = "conservative"
        else:
            verdict = "consistent"
        print("  {:35s}: mean NEES = {:6.3f}   [{}]".format(label, m, verdict))

    # Plot
    t = np.arange(N_STEPS + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(truth[:, 0], truth[:, 1], "k-", linewidth=1.5, label="truth")
    ax.scatter(BEACONS[:, 0], BEACONS[:, 1], marker="^", color="red",
               s=80, label="beacons", zorder=5)
    ax.plot(ekf_estimates["EKF well-tuned"][0, :, 0],
            ekf_estimates["EKF well-tuned"][0, :, 1],
            "b-", alpha=0.5, linewidth=0.8, label="one well-tuned EKF run")
    ax.plot(ekf_estimates["EKF overconfident"][0, :, 0],
            ekf_estimates["EKF overconfident"][0, :, 1],
            "r-", alpha=0.5, linewidth=0.8, label="one overconfident EKF run")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Trajectory and beacons")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    ax.plot(t, empirical_mse_traces["EKF well-tuned"], "k-", linewidth=2.0,
            label="emp MSE (well-tuned)")
    ax.plot(t, empirical_mse_traces["EKF overconfident"], "k--", linewidth=2.0,
            label="emp MSE (overconfident)")
    ax.plot(t, pcrb_traces, "g-", linewidth=1.8, label="PCRB (recursive)")
    ax.plot(t, np.array([ekf_P_full["EKF well-tuned"][:, k].mean(axis=0).trace()
                         for k in range(N_STEPS + 1)]),
            "b-", linewidth=1.2, label="P (well-tuned EKF)")
    ax.plot(t, np.array([ekf_P_full["EKF overconfident"][:, k].mean(axis=0).trace()
                         for k in range(N_STEPS + 1)]),
            "r-", linewidth=1.2, label="P (overconfident EKF)")
    ax.plot(t, std_crlb_traces, "orange", linewidth=1.0, alpha=0.7,
            label="Std CRLB (single-step)")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("trace(P) (m^2)")
    ax.set_title("Cov trace vs time — PCRB is the floor")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    P_well_mean = ekf_P_full["EKF well-tuned"].mean(axis=0)
    P_over_mean = ekf_P_full["EKF overconfident"].mean(axis=0)
    ax.plot(t, empirical_mse_full["EKF overconfident"][:, 0, 0], "k-",
            linewidth=1.6, label="emp MSE xx (over)")
    ax.plot(t, P_over_mean[:, 0, 0], "r-", linewidth=1.2, label="P xx (over)")
    ax.plot(t, P_well_mean[:, 0, 0], "b-", linewidth=1.2, label="P xx (well)")
    ax.plot(t, pcrb_full[:, 0, 0], "g-", linewidth=1.5, label="PCRB xx")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)"); ax.set_ylabel("variance (m^2)")
    ax.set_title("x-axis variance: overconfident P dips below PCRB")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    ax.plot(t, nees["EKF well-tuned vs own P"], "b-", linewidth=1.2,
            label="well-tuned vs own P")
    ax.plot(t, nees["EKF overconfident vs own P"], "r-", linewidth=1.2,
            label="overconfident vs own P")
    ax.plot(t, nees["EKF well-tuned vs PCRB"], "g-", linewidth=1.2,
            label="well-tuned vs PCRB")
    ax.plot(t, nees["EKF overconfident vs PCRB"], "g--", linewidth=1.2,
            label="overconfident vs PCRB")
    ax.axhline(N_DIM, color="black", linestyle="-", linewidth=0.8,
               label=f"target ({N_DIM})")
    ax.fill_between(t,
                    N_DIM - 1.96 * np.sqrt(2 * N_DIM / N_MC),
                    N_DIM + 1.96 * np.sqrt(2 * N_DIM / N_MC),
                    color="gray", alpha=0.2, label="95% band")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)"); ax.set_ylabel("NEES (avg over MC)")
    ax.set_title("NEES — overconfident EKF blows up, PCRB stays sane")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_pcrb_shadow.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

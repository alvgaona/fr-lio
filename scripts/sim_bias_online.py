"""Direction 1, part B: online estimation of the IESKF bias b without
Monte Carlo.

Script 1 (sim_biased_crlb.py) needed N_MC trials to estimate b and ∂b/∂x.
The real filter has one scan and no access to ground truth, so we need an
*online* approximation. This script demonstrates two:

  1. Self-consistent (SC) predictor.
     For a MAP estimator with Gaussian prior + linearized measurement,
        x̂ ≈ (J + P0^-1)^-1 (J x_true + P0^-1 μ_prior)
     with J = H^T R^-1 H. Subtracting x_true:
        b(x_true) = (J + P0^-1)^-1 P0^-1 (μ_prior - x_true)
     We don't know x_true so we substitute x̂ and iterate. One or two
     iterations suffice — converges fast because the formula is linear in
     (μ_prior - x).

  2. Unscented (UT) predictor.
     Sigma points from N(x̂, P_post). For each sigma point x_i, evaluate
     b(x_i) using the formula above. Average with UT weights. Captures the
     uncertainty in "where might truth actually be" via P_post.

Both predictors are evaluated against the MC ground-truth bias from script 1's
setup. The deliverable is: if SC or UT bias predictions match MC truth, we
have a recipe for publishing the MSE form online.

Run: python sim_bias_online.py
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

# Reuse script 1's setup verbatim so the bias picture is comparable.
N_DIM = 2
WALLS = np.array([
    [1.0, 0.0, -3.0],
    [-1.0, 0.0, -3.0],
    [0.0, 1.0, -3.0],
    [0.0, -1.0, -3.0],
])
N_POINTS = 8
SIGMA_POINT = 0.30
PRIOR_STD = np.array([0.25, 0.18])
P0 = np.diag(PRIOR_STD ** 2)
P0_INV = np.linalg.inv(P0)
X_TRUE = np.array([0.6, 0.5])
PRIOR_MEAN_OFFSET = np.array([0.20, 0.10])
PRIOR_MEAN = X_TRUE + PRIOR_MEAN_OFFSET  # the (biased) prior the filter starts from

N_MC = 4000
MAX_ITERS = 15
CONV_TOL = 1e-7


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_body_points_at(x):
    """Body-frame scan points implied by sensor at x."""
    tx, theta = x
    sensor_world = np.array([tx, 0.0])
    R = rot2(theta)
    # Same nominal ray pattern as script 1 (cached here once).
    rng = np.random.default_rng(42)
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


BODY_POINTS_AT_TRUE = make_body_points_at(X_TRUE)


def residuals_and_jac(x, body_points):
    tx, theta = x
    R = rot2(theta)
    pw = body_points @ R.T + np.array([tx, 0.0])
    N = body_points.shape[0]
    r = np.zeros(N)
    J = np.zeros((N, 2))
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
        px, py = body_points[i]
        dRp = np.array([-s * px - c * py, c * px - s * py])
        n0, n1 = best_n
        J[i, 0] = n0
        J[i, 1] = n0 * dRp[0] + n1 * dRp[1]
    return r, J


def ieskf_map(body_points_noisy, x_prior_mean):
    x = x_prior_mean.copy()
    inv_var = 1.0 / (SIGMA_POINT ** 2)
    for _ in range(MAX_ITERS):
        r, J = residuals_and_jac(x, body_points_noisy)
        H = inv_var * (J.T @ J) + P0_INV
        g = inv_var * (J.T @ r) + P0_INV @ (x - x_prior_mean)
        dx = -np.linalg.solve(H, g)
        x = x + dx
        if np.linalg.norm(dx) < CONV_TOL:
            break
    return x


def posterior_information(x, body_points):
    """J + P0^-1 evaluated at x using body_points geometry."""
    _, Jh = residuals_and_jac(x, body_points)
    J = (1.0 / SIGMA_POINT ** 2) * (Jh.T @ Jh)
    return J + P0_INV, J


def bias_self_consistent(body_points, x_hat, mu_prior, n_iters=3):
    """SC bias predictor. Substitute x_hat for x_true in the analytic
    formula, then iterate by updating x_corrected = x_hat - b̂ and
    recomputing b̂ at the corrected estimate."""
    x_corr = x_hat.copy()
    for _ in range(n_iters):
        info_post, _ = posterior_information(x_corr, body_points)
        info_post_inv = np.linalg.inv(info_post)
        b_hat = info_post_inv @ P0_INV @ (mu_prior - x_corr)
        x_corr = x_hat - b_hat
    return b_hat


def bias_unscented(body_points, x_hat, mu_prior, P_post, kappa=1.0):
    """UT bias predictor. Sigma points from N(x_hat, P_post); evaluate the
    analytic bias formula at each and average with UT weights."""
    n = N_DIM
    L = np.linalg.cholesky((n + kappa) * P_post)
    sigma_pts = [x_hat]
    for i in range(n):
        sigma_pts.append(x_hat + L[:, i])
        sigma_pts.append(x_hat - L[:, i])
    weights = np.zeros(2 * n + 1)
    weights[0] = kappa / (n + kappa)
    weights[1:] = 1.0 / (2 * (n + kappa))
    b_acc = np.zeros(n)
    for w, xs in zip(weights, sigma_pts):
        info_post, _ = posterior_information(xs, body_points)
        info_post_inv = np.linalg.inv(info_post)
        b_acc += w * info_post_inv @ P0_INV @ (mu_prior - xs)
    return b_acc


def run_mc():
    rng = np.random.default_rng(123)
    estimates = np.zeros((N_MC, N_DIM))
    b_sc = np.zeros((N_MC, N_DIM))
    b_ut = np.zeros((N_MC, N_DIM))

    for k in range(N_MC):
        body_noisy = BODY_POINTS_AT_TRUE + rng.normal(scale=SIGMA_POINT,
                                                      size=BODY_POINTS_AT_TRUE.shape)
        x_hat = ieskf_map(body_noisy, x_prior_mean=PRIOR_MEAN)
        info_post, _ = posterior_information(x_hat, body_noisy)
        P_post = np.linalg.inv(info_post)
        estimates[k] = x_hat
        b_sc[k] = bias_self_consistent(body_noisy, x_hat, PRIOR_MEAN)
        b_ut[k] = bias_unscented(body_noisy, x_hat, PRIOR_MEAN, P_post)
    return estimates, b_sc, b_ut


def main():
    print("Running MC ({} trials) ...".format(N_MC))
    estimates, b_sc, b_ut = run_mc()
    bias_mc_truth = estimates.mean(axis=0) - X_TRUE
    bias_sc_mean = b_sc.mean(axis=0)
    bias_ut_mean = b_ut.mean(axis=0)

    print("\n=== Bias comparison ===")
    print("  MC ground-truth bias        = {}".format(bias_mc_truth))
    print("  Self-consistent (mean of b̂) = {}".format(bias_sc_mean))
    print("  Unscented (mean of b̂)       = {}".format(bias_ut_mean))
    print("  ||SC - MC||                 = {:.4e}".format(np.linalg.norm(bias_sc_mean - bias_mc_truth)))
    print("  ||UT - MC||                 = {:.4e}".format(np.linalg.norm(bias_ut_mean - bias_mc_truth)))

    # Per-trial NEES with each predictor.
    # For each trial we form P_pub = P_post + b̂ b̂^T (MSE form using per-trial bias)
    # and compute NEES = eps^T P_pub^-1 eps where eps = x_true - x̂.
    nees_records = {}
    eps_all = X_TRUE[None, :] - estimates

    for name, b_arr in [("no bias correction", np.zeros_like(b_sc)),
                        ("MC-mean bias (oracle)", np.tile(bias_mc_truth, (N_MC, 1))),
                        ("self-consistent (per-trial)", b_sc),
                        ("unscented (per-trial)", b_ut)]:
        nees_list = []
        for k in range(N_MC):
            body_noisy = BODY_POINTS_AT_TRUE  # geometry-only — use noiseless body for P
            info_post, _ = posterior_information(estimates[k], body_noisy)
            P_post = np.linalg.inv(info_post)
            P_pub = P_post + np.outer(b_arr[k], b_arr[k])
            try:
                nees = float(eps_all[k] @ np.linalg.solve(P_pub, eps_all[k]))
            except np.linalg.LinAlgError:
                nees = np.nan
            nees_list.append(nees)
        nees_records[name] = np.array(nees_list)

    lo = N_DIM - 1.96 * np.sqrt(2 * N_DIM / N_MC)
    hi = N_DIM + 1.96 * np.sqrt(2 * N_DIM / N_MC)
    print("\n=== NEES (target = {}, 95% band [{:.3f}, {:.3f}]) ===".format(N_DIM, lo, hi))
    for name, nees in nees_records.items():
        m = np.nanmean(nees)
        if m > hi:
            verdict = "OVERCONFIDENT"
        elif m < lo:
            verdict = "conservative"
        else:
            verdict = "consistent"
        print("  {:30s}: mean NEES = {:6.3f}   [{}]".format(name, m, verdict))

    # Plot bias scatter and NEES histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.scatter(b_sc[:, 0], b_sc[:, 1], s=4, alpha=0.2, color="tab:orange",
               label="self-consistent")
    ax.scatter(b_ut[:, 0], b_ut[:, 1], s=4, alpha=0.2, color="tab:green",
               label="unscented")
    ax.scatter([bias_mc_truth[0]], [bias_mc_truth[1]], color="black",
               marker="x", s=120, label="MC truth", zorder=5)
    ax.scatter([bias_sc_mean[0]], [bias_sc_mean[1]], color="tab:orange",
               marker="*", s=200, edgecolor="black",
               label="SC mean", zorder=5)
    ax.scatter([bias_ut_mean[0]], [bias_ut_mean[1]], color="tab:green",
               marker="*", s=200, edgecolor="black",
               label="UT mean", zorder=5)
    ax.set_xlabel("b_tx (m)"); ax.set_ylabel("b_theta (rad)")
    ax.set_title("Per-trial bias estimates vs MC truth")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    bins = np.linspace(0, 20, 60)
    for name, color in [("no bias correction", "tab:red"),
                        ("MC-mean bias (oracle)", "black"),
                        ("self-consistent (per-trial)", "tab:orange"),
                        ("unscented (per-trial)", "tab:green")]:
        nees = nees_records[name]
        nees = nees[np.isfinite(nees)]
        ax.hist(nees, bins=bins, alpha=0.4, color=color,
                label=f"{name} (mean={nees.mean():.2f})")
    ax.axvline(N_DIM, color="black", linestyle="--", linewidth=1.0,
               label=f"target = {N_DIM}")
    ax.set_xlabel("NEES"); ax.set_ylabel("count")
    ax.set_title("Per-trial NEES distribution")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_bias_online.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

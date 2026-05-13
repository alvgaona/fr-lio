"""Script C: extend the biased CRLB demonstration to full SE(2): state = [tx, ty, theta].

Script 1 used [tx, theta] only. Real LIO has both translation axes plus
rotation, with cross-coupling between them in the point-to-plane residual.
The question this script tests: does the script-1 conclusion (standard CRLB
fails NEES, MSE form passes) still hold when we add the second translation
axis, where there's nontrivial coupling between tx, ty, and theta?

Setup identical to script 1 except:
  - x = [tx, ty, theta]  (3-DOF SE(2))
  - sensor world position = (tx, ty)  (not just (tx, 0))
  - prior mean offset has all three components

Run: python sim_biased_crlb_3dof.py
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

N_DIM = 3
WALLS = np.array([
    [1.0, 0.0, -3.0],
    [-1.0, 0.0, -3.0],
    [0.0, 1.0, -3.0],
    [0.0, -1.0, -3.0],
])
N_POINTS = 10
SIGMA_POINT = 0.30
PRIOR_STD = np.array([0.25, 0.25, 0.18])
P0 = np.diag(PRIOR_STD ** 2)
P0_INV = np.linalg.inv(P0)
X_TRUE = np.array([0.6, -0.4, 0.5])
PRIOR_MEAN_OFFSET = np.array([0.20, -0.15, 0.10])
PRIOR_MEAN = X_TRUE + PRIOR_MEAN_OFFSET
N_MC = 6000
MAX_ITERS = 15
CONV_TOL = 1e-7


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_body_points_at(x):
    tx, ty, theta = x
    sensor_world = np.array([tx, ty])
    R = rot2(theta)
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
        J[i, 0] = n0
        J[i, 1] = n1
        J[i, 2] = n0 * dRp[0] + n1 * dRp[1]
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


def monte_carlo(x_true, n_trials=N_MC):
    rng = np.random.default_rng(123)
    body_at_x = make_body_points_at(x_true)
    estimates = np.zeros((n_trials, N_DIM))
    for k in range(n_trials):
        noise = rng.normal(scale=SIGMA_POINT, size=body_at_x.shape)
        estimates[k] = ieskf_map(body_at_x + noise, x_prior_mean=x_true + PRIOR_MEAN_OFFSET)
    return estimates


def bias_jacobian(x_true, delta=0.02, n_trials=N_MC // 3):
    db_dx = np.zeros((N_DIM, N_DIM))
    rng = np.random.default_rng(456)
    for j in range(N_DIM):
        e = np.zeros(N_DIM); e[j] = 1.0
        # Run two perturbed MCs sharing seeds for variance reduction.
        body_plus = make_body_points_at(x_true + delta * e)
        body_minus = make_body_points_at(x_true - delta * e)
        est_plus = np.zeros((n_trials, N_DIM))
        est_minus = np.zeros((n_trials, N_DIM))
        for k in range(n_trials):
            noise = rng.normal(scale=SIGMA_POINT, size=body_plus.shape)
            est_plus[k] = ieskf_map(body_plus + noise,
                                     x_prior_mean=(x_true + delta * e) + PRIOR_MEAN_OFFSET)
            est_minus[k] = ieskf_map(body_minus + noise,
                                      x_prior_mean=(x_true - delta * e) + PRIOR_MEAN_OFFSET)
        b_plus = est_plus.mean(axis=0) - (x_true + delta * e)
        b_minus = est_minus.mean(axis=0) - (x_true - delta * e)
        db_dx[:, j] = (b_plus - b_minus) / (2 * delta)
    return db_dx


def posterior_fim(x_true):
    body_at_x = make_body_points_at(x_true)
    _, Jh = residuals_and_jac(x_true, body_at_x)
    return (1.0 / SIGMA_POINT ** 2) * (Jh.T @ Jh) + P0_INV


def main():
    print("Running MC ({} trials) ...".format(N_MC))
    estimates = monte_carlo(X_TRUE)
    bias = estimates.mean(axis=0) - X_TRUE
    cov_emp = np.cov(estimates.T)

    print("Estimating db/dx ...")
    db_dx = bias_jacobian(X_TRUE)

    J = posterior_fim(X_TRUE)
    crlb_std = np.linalg.inv(J)
    M = np.eye(N_DIM) + db_dx
    crlb_corr_cov = M @ crlb_std @ M.T
    crlb_corr_mse = crlb_corr_cov + np.outer(bias, bias)

    print("\n=== Results at x_true = {} ===".format(X_TRUE))
    print("Empirical bias        = {}".format(bias))
    print("||bias|| / ||offset|| = {:.3f}  (fraction of prior offset that survives)".format(
        np.linalg.norm(bias) / np.linalg.norm(PRIOR_MEAN_OFFSET)))

    print("\ndet(empirical Cov)       = {:.3e}".format(np.linalg.det(cov_emp)))
    print("det(standard CRLB)       = {:.3e}".format(np.linalg.det(crlb_std)))
    print("det(bias-corrected Cov)  = {:.3e}".format(np.linalg.det(crlb_corr_cov)))
    print("det(bias-corrected MSE)  = {:.3e}".format(np.linalg.det(crlb_corr_mse)))

    eps = X_TRUE[None, :] - estimates
    nees_results = {}
    for name, P in [
        ("standard CRLB",      crlb_std),
        ("bias-corrected Cov", crlb_corr_cov),
        ("bias-corrected MSE", crlb_corr_mse),
        ("empirical Cov",      cov_emp),
    ]:
        Pinv = np.linalg.inv(P)
        nees = np.einsum("ki,ij,kj->k", eps, Pinv, eps)
        nees_results[name] = nees

    lo = N_DIM - 1.96 * np.sqrt(2 * N_DIM / N_MC)
    hi = N_DIM + 1.96 * np.sqrt(2 * N_DIM / N_MC)
    print("\n=== NEES (target {}, 95% band [{:.3f}, {:.3f}]) ===".format(N_DIM, lo, hi))
    for name, nees in nees_results.items():
        m = nees.mean()
        if m > hi:
            verdict = "OVERCONFIDENT"
        elif m < lo:
            verdict = "conservative"
        else:
            verdict = "consistent"
        print("  {:24s}: mean NEES = {:6.3f}   [{}]".format(name, m, verdict))

    # Plot: marginal histograms per axis and NEES distribution.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = ["tx (m)", "ty (m)", "theta (rad)"]
    for j in range(3):
        ax = axes.flat[j]
        ax.hist(estimates[:, j], bins=60, density=True, alpha=0.5,
                color="gray", label="MC estimates")
        ax.axvline(X_TRUE[j], color="black", linestyle="-", linewidth=1.5,
                   label="x_true")
        ax.axvline(X_TRUE[j] + bias[j], color="red", linestyle="--",
                   linewidth=1.5, label="E[x̂]")
        ax.axvline(PRIOR_MEAN[j], color="blue", linestyle=":",
                   linewidth=1.5, label="μ_prior")
        ax.set_xlabel(labels[j])
        ax.set_title("Marginal: {}".format(labels[j]))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes.flat[3]
    bins = np.linspace(0, 25, 80)
    for name, color in [("standard CRLB", "tab:orange"),
                        ("bias-corrected Cov", "tab:red"),
                        ("bias-corrected MSE", "tab:green"),
                        ("empirical Cov", "tab:blue")]:
        nees = nees_results[name]
        ax.hist(nees, bins=bins, alpha=0.4, color=color,
                label=f"{name} (mean={nees.mean():.2f})")
    ax.axvline(N_DIM, color="black", linestyle="--", linewidth=1.2,
               label=f"target = {N_DIM}")
    ax.set_xlabel("NEES"); ax.set_ylabel("count")
    ax.set_title("NEES distribution")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_biased_crlb_3dof.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

"""Script E: Bias-Corrected Variational (BCV) estimator — the novel direction.

Standard IESKF/MAP minimizes a second-order Taylor approximation of the
posterior log-density. The third-order term in that expansion is what
produces estimator bias for nonlinear residuals (e.g. point-to-plane where
the rotation enters via cos/sin of theta).

Variational inference (VI) over a Gaussian family q(x) = N(m, Σ) with a
second-order Taylor on the residuals gives back the IESKF point estimate and
EKF posterior covariance — no novelty. The novelty is in keeping the
third-order term:

    r_i(x) ≈ r_i(m) + h_i^T δ + (1/2) δ^T H_i δ,   δ = x - m

with H_i = Hessian of r_i. Then E_q[r_i^2] under Gaussian δ contains a
"curvature-bias" term proportional to r_i(m) tr(H_i Σ). The ELBO gradient
in m becomes:

    0 = (1/σ²) Σ_i [r_i(m) + (1/2) tr(H_i Σ)] ∇r_i(m) + P0^-1 (m - μ)

The new bracket [r_i(m) + (1/2) tr(H_i Σ)] is a curvature-corrected residual:
m is pulled to minimize residuals while accounting for the Σ-weighted
curvature of each residual. This recovers what online bias estimation
(script B) tried to approximate, but as a *one-shot* fixed-point estimator
rather than a post-hoc correction.

Setup is the SE(2) point-to-plane problem from script C. We compare:
  - Vanilla MAP (= IESKF): biased
  - BCV (this script): expected to be less biased
  - Both against MC ground truth

Run: python sim_bcv_estimator.py
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
# Prior centered at truth so prior-pull bias = 0. Any remaining bias must
# come from measurement nonlinearity → this is the regime BCV should fix.
PRIOR_MEAN_OFFSET = np.zeros(N_DIM)
PRIOR_MEAN = X_TRUE + PRIOR_MEAN_OFFSET
N_MC = 6000
MAX_ITERS = 20
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


def residuals_jac_and_hess(x, body_points):
    """Return r (N,), J (N, 3), and H (N, 3, 3) — the Hessian tensor of each
    residual w.r.t. x. For point-to-plane in SE(2), only the [theta, theta]
    entry of each H_i is nonzero:
        ∂²r_i / ∂θ² = -n_i^T R(theta) p_i      (=== -(residual + d) projected)
    Specifically  R''(theta) p = -R(theta) p, so n^T R'' p = -n^T R p.
    """
    tx, ty, theta = x
    R = rot2(theta)
    pw = body_points @ R.T + np.array([tx, ty])
    N = body_points.shape[0]
    r = np.zeros(N)
    J = np.zeros((N, 3))
    H = np.zeros((N, 3, 3))
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
        # First derivatives.
        dRp = np.array([-s * px - c * py, c * px - s * py])
        J[i, 0] = n0; J[i, 1] = n1
        J[i, 2] = n0 * dRp[0] + n1 * dRp[1]
        # Second derivative: ∂²r/∂θ² = -n^T R(θ) p = -(r_i - d_i_dropped...)
        # Compute directly: R''(θ)p = -R(θ)p, so n^T R''(θ) p:
        Rp = R @ np.array([px, py])
        H[i, 2, 2] = -(n0 * Rp[0] + n1 * Rp[1])
    return r, J, H


def map_estimate(body_points_noisy, mu_prior):
    """Vanilla IESKF/MAP."""
    x = mu_prior.copy()
    inv_var = 1.0 / (SIGMA_POINT ** 2)
    for _ in range(MAX_ITERS):
        r, J, _ = residuals_jac_and_hess(x, body_points_noisy)
        info_post = inv_var * (J.T @ J) + P0_INV
        g = inv_var * (J.T @ r) + P0_INV @ (x - mu_prior)
        dx = -np.linalg.solve(info_post, g)
        x = x + dx
        if np.linalg.norm(dx) < CONV_TOL:
            break
    Sigma = np.linalg.inv(info_post)
    return x, Sigma


def bcv_estimate(body_points_noisy, mu_prior):
    """Bias-Corrected Variational estimator.

    Fixed point iteration:
        Σ ← (J^T R^-1 J + P0^-1)^-1                  (standard posterior cov)
        r_eff_i ← r_i(m) + (1/2) tr(H_i Σ)            (curvature correction)
        m ← solve normal equation with r_eff

    The tr(H_i Σ) term shifts the residual by the second-order bias each
    residual generates under Gaussian noise. The shift drives m toward the
    state that would produce the *observed* residual after the bias has
    been accounted for — net effect: m moves closer to truth.
    """
    x = mu_prior.copy()
    inv_var = 1.0 / (SIGMA_POINT ** 2)
    Sigma = P0.copy()
    for _ in range(MAX_ITERS):
        r, J, H = residuals_jac_and_hess(x, body_points_noisy)
        # Curvature correction per residual: c_i = (1/2) tr(H_i Σ)
        c_vec = 0.5 * np.einsum("ijk,kj->i", H, Sigma)
        r_eff = r + c_vec

        info_post = inv_var * (J.T @ J) + P0_INV
        g = inv_var * (J.T @ r_eff) + P0_INV @ (x - mu_prior)
        dx = -np.linalg.solve(info_post, g)
        x = x + dx
        Sigma = np.linalg.inv(info_post)
        if np.linalg.norm(dx) < CONV_TOL:
            break
    return x, Sigma


def monte_carlo():
    rng = np.random.default_rng(123)
    estimates_map = np.zeros((N_MC, N_DIM))
    estimates_bcv = np.zeros((N_MC, N_DIM))
    P_map_acc = np.zeros((N_DIM, N_DIM))
    P_bcv_acc = np.zeros((N_DIM, N_DIM))
    for k in range(N_MC):
        body_noisy = BODY_POINTS_AT_TRUE + rng.normal(scale=SIGMA_POINT,
                                                      size=BODY_POINTS_AT_TRUE.shape)
        m_map, P_map = map_estimate(body_noisy, PRIOR_MEAN)
        m_bcv, P_bcv = bcv_estimate(body_noisy, PRIOR_MEAN)
        estimates_map[k] = m_map
        estimates_bcv[k] = m_bcv
        P_map_acc += P_map
        P_bcv_acc += P_bcv
    return (estimates_map, estimates_bcv,
            P_map_acc / N_MC, P_bcv_acc / N_MC)


def main():
    print("Running MC ({} trials) ...".format(N_MC))
    est_map, est_bcv, P_map_mean, P_bcv_mean = monte_carlo()

    bias_map = est_map.mean(axis=0) - X_TRUE
    bias_bcv = est_bcv.mean(axis=0) - X_TRUE
    cov_map = np.cov(est_map.T)
    cov_bcv = np.cov(est_bcv.T)

    print("\n=== Bias comparison ===")
    print("  MAP bias   = {}".format(bias_map))
    print("  BCV bias   = {}".format(bias_bcv))
    print("  ||bias|| MAP = {:.4f}".format(np.linalg.norm(bias_map)))
    print("  ||bias|| BCV = {:.4f}".format(np.linalg.norm(bias_bcv)))
    print("  Bias reduction factor: {:.2f}x".format(
        np.linalg.norm(bias_map) / max(np.linalg.norm(bias_bcv), 1e-9)))

    print("\n=== Covariance comparison ===")
    print("  trace(empirical Cov MAP) = {:.4e}".format(np.trace(cov_map)))
    print("  trace(empirical Cov BCV) = {:.4e}".format(np.trace(cov_bcv)))
    print("  trace(P_map mean)        = {:.4e}".format(np.trace(P_map_mean)))
    print("  trace(P_bcv mean)        = {:.4e}".format(np.trace(P_bcv_mean)))

    # NEES: eps = x_true - x_hat, P = published cov.
    nees_results = {}
    for name, eps_arr, P in [
        ("MAP / own P",       X_TRUE[None, :] - est_map, P_map_mean),
        ("MAP / MSE = P + bbᵀ", X_TRUE[None, :] - est_map, P_map_mean + np.outer(bias_map, bias_map)),
        ("BCV / own P",       X_TRUE[None, :] - est_bcv, P_bcv_mean),
        ("BCV / MSE = P + bbᵀ", X_TRUE[None, :] - est_bcv, P_bcv_mean + np.outer(bias_bcv, bias_bcv)),
    ]:
        Pinv = np.linalg.inv(P)
        nees = np.einsum("ki,ij,kj->k", eps_arr, Pinv, eps_arr)
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
        print("  {:25s}: mean NEES = {:6.3f}   [{}]".format(name, m, verdict))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    labels = ["tx (m)", "ty (m)", "theta (rad)"]
    for j in range(3):
        ax = axes.flat[j]
        ax.hist(est_map[:, j], bins=60, density=True, alpha=0.4, color="tab:red", label="MAP")
        ax.hist(est_bcv[:, j], bins=60, density=True, alpha=0.4, color="tab:green", label="BCV")
        ax.axvline(X_TRUE[j], color="black", linestyle="-", linewidth=1.5, label="x_true")
        ax.axvline(PRIOR_MEAN[j], color="blue", linestyle=":", linewidth=1.2, label="μ_prior")
        ax.set_xlabel(labels[j])
        ax.set_title("Marginal {}".format(labels[j]))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes.flat[3]
    bins = np.linspace(0, 25, 80)
    for name, color in [("MAP / own P", "tab:red"),
                        ("BCV / own P", "tab:green"),
                        ("MAP / MSE = P + bbᵀ", "tab:orange"),
                        ("BCV / MSE = P + bbᵀ", "tab:purple")]:
        nees = nees_results[name]
        ax.hist(nees, bins=bins, alpha=0.35, color=color,
                label=f"{name} (mean={nees.mean():.2f})")
    ax.axvline(N_DIM, color="black", linestyle="--", linewidth=1.2,
               label=f"target = {N_DIM}")
    ax.set_xlabel("NEES"); ax.set_ylabel("count")
    ax.set_title("Per-trial NEES distribution")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = f"{OUT_DIR}/sim_bcv_estimator.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

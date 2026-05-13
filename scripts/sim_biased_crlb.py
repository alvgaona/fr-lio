"""Direction 1: bias-corrected CRLB for an IESKF-style point-to-plane MAP estimator.

The standard CRLB Cov(x̂) >= J^-1 only holds for unbiased estimators. The IESKF
is a MAP estimator on nonlinear point-to-plane residuals — both the iterated
linearization and the prior introduce bias b(x) = E[x̂] - x. The bound that
actually applies is

    Cov(x̂) >= (I + db/dx) J^-1 (I + db/dx)^T          (van Trees, biased CRLB)

with the MSE form adding b b^T on top. This script demonstrates the gap on a
minimal 2-DOF static-pose problem (tx, theta) with point-to-plane residuals
against a small map of walls, and shows that the bias-corrected bound is the
right one to publish.

Setup:
  - 2D world with 4 walls (planes). State x = [tx, theta].
  - Body-frame scan of N points lying on the walls, perturbed by sensor noise.
  - Prior: N(x_prior_mean, P0) with x_prior_mean = x_true.
  - Estimator: iterated Gauss-Newton on -log p(z|x) - log p(x) (=== IESKF MAP).

Outputs (in sim_data/):
  - sim_biased_crlb.png: 2-sigma ellipses for empirical Cov, posterior CRLB
    (J^-1), and the bias-corrected CRLB on the (tx, theta) plane.
  - Console summary with the determinants and a "breach ratio".

Run: python sim_biased_crlb.py
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

N_DIM = 2  # x = [tx, theta]

# Walls (lines in 2D): n^T x + d = 0 with unit normal n. A square room.
WALLS = np.array([
    [1.0, 0.0, -3.0],
    [-1.0, 0.0, -3.0],
    [0.0, 1.0, -3.0],
    [0.0, -1.0, -3.0],
])

N_POINTS = 8                 # few measurements => prior + nonlinearity dominate
SIGMA_POINT = 0.30           # high body-frame point noise std (m)
PRIOR_STD = np.array([0.25, 0.18])  # prior std on [tx, theta]
P0 = np.diag(PRIOR_STD ** 2)
P0_INV = np.linalg.inv(P0)

X_TRUE = np.array([0.6, 0.5])
# Prior mean is offset from truth: mimics a propagated IESKF prior that already
# carries its own bias from IMU pre-integration. This is the regime where the
# MAP estimator is meaningfully biased.
PRIOR_MEAN_OFFSET = np.array([0.20, 0.10])

N_MC = 6000
MAX_ITERS = 15
CONV_TOL = 1e-7


def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sample_body_points(rng):
    """Draw N_POINTS true body-frame scan points from rays hitting walls.

    We place the sensor at x_true and shoot rays uniformly over [0, 2pi),
    intersect with the nearest wall, then express the hit point in the body
    frame so the same body-frame ray pattern is reused across MC trials
    (noise will perturb each draw)."""
    tx, theta = X_TRUE
    sensor_world = np.array([tx, 0.0])  # ty is fixed (state is only tx, theta)
    R = rot2(theta)
    angles = rng.uniform(0, 2 * np.pi, size=N_POINTS)
    body_points = np.zeros((N_POINTS, 2))
    for i, a in enumerate(angles):
        dir_world = np.array([np.cos(a), np.sin(a)])
        # Find nearest positive intersection with any wall.
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
        body_points[i] = R.T @ (hit_world - sensor_world)
    return body_points


# Generate a fixed ray pattern (body-frame geometry) once. Per-trial noise is
# added to body points to simulate sensor noise.
BODY_POINTS_NOMINAL = sample_body_points(np.random.default_rng(42))


def world_points(x, body_points):
    """Transform body-frame points into the world using state x = [tx, theta]."""
    tx, theta = x
    R = rot2(theta)
    return body_points @ R.T + np.array([tx, 0.0])


def residuals_and_jac(x, body_points):
    """Return residuals r (N,) and Jacobian J (N, 2) for point-to-plane.

    Each point is assigned to its nearest wall (data association evaluated at
    the current x). The residual for point p in body frame is
        r_i = n^T (R(theta) p + [tx, 0]) + d
    with Jacobian
        dr/d(tx)   = n_x
        dr/d(theta)= n^T R'(theta) p
    where R'(theta) p = [-sin*p_x - cos*p_y, cos*p_x - sin*p_y].
    """
    tx, theta = x
    R = rot2(theta)
    pw = body_points @ R.T + np.array([tx, 0.0])
    N = body_points.shape[0]
    r = np.zeros(N)
    J = np.zeros((N, 2))
    c, s = np.cos(theta), np.sin(theta)
    for i in range(N):
        # Nearest wall by signed-distance magnitude.
        best_idx = 0
        best_abs = np.inf
        best_signed = 0.0
        for k, (n0, n1, d) in enumerate(WALLS):
            sd = n0 * pw[i, 0] + n1 * pw[i, 1] + d
            if abs(sd) < best_abs:
                best_abs = abs(sd)
                best_idx = k
                best_signed = sd
        n0, n1, _ = WALLS[best_idx]
        r[i] = best_signed
        px, py = body_points[i]
        # d/d(theta) of R(theta) p
        dRp_dtheta = np.array([-s * px - c * py, c * px - s * py])
        J[i, 0] = n0
        J[i, 1] = n0 * dRp_dtheta[0] + n1 * dRp_dtheta[1]
    return r, J


def ieskf_map(body_points_noisy, x_prior_mean):
    """Iterated Gauss-Newton MAP estimate (== IESKF on a single static update).

    Minimizes ||r(x)||^2 / sigma_p^2 + (x - x_prior_mean)^T P0^-1 (x - x_prior_mean).
    """
    x = x_prior_mean.copy()
    inv_var = 1.0 / (SIGMA_POINT ** 2)
    for _ in range(MAX_ITERS):
        r, J = residuals_and_jac(x, body_points_noisy)
        # Gauss-Newton normal equation including prior.
        H = inv_var * (J.T @ J) + P0_INV
        g = inv_var * (J.T @ r) + P0_INV @ (x - x_prior_mean)
        dx = -np.linalg.solve(H, g)
        x = x + dx
        if np.linalg.norm(dx) < CONV_TOL:
            break
    return x


def monte_carlo(x_true, rng, n_trials=N_MC):
    """Run n_trials of the IESKF MAP at the given x_true. Return estimates."""
    # Generate the noiseless body-frame points implied by x_true. Walls are
    # fixed in the world, so we re-derive nominal body points for this x_true
    # using the world-frame hit set from the nominal geometry. To keep the
    # geometry comparable across small perturbations of x_true, we recompute
    # the body points assuming the sensor sits at x_true.
    tx, theta = x_true
    sensor_world = np.array([tx, 0.0])
    R = rot2(theta)
    # Re-derive nominal hits using the original ray angles (fixed across runs).
    # For simplicity, we just reuse BODY_POINTS_NOMINAL transformed through
    # (x_true - X_TRUE): convert nominal body points to world via X_TRUE,
    # then back to body via x_true. This keeps the geometry stable.
    world_nominal = BODY_POINTS_NOMINAL @ rot2(X_TRUE[1]).T + np.array([X_TRUE[0], 0.0])
    body_at_x = (world_nominal - sensor_world) @ R  # R.T inverse rotation
    estimates = np.zeros((n_trials, N_DIM))
    for k in range(n_trials):
        noise = rng.normal(scale=SIGMA_POINT, size=body_at_x.shape)
        body_noisy = body_at_x + noise
        # Prior mean offset from truth (mimics a biased propagated prior).
        x_prior_mean = x_true + PRIOR_MEAN_OFFSET
        estimates[k] = ieskf_map(body_noisy, x_prior_mean=x_prior_mean)
    return estimates


def bias_jacobian(x_true, rng, delta=np.array([0.02, 0.02]), n_trials=N_MC // 2):
    """Estimate db/dx by central differences around x_true."""
    db_dx = np.zeros((N_DIM, N_DIM))
    for j in range(N_DIM):
        e = np.zeros(N_DIM); e[j] = 1.0
        x_plus = x_true + delta[j] * e
        x_minus = x_true - delta[j] * e
        est_plus = monte_carlo(x_plus, rng, n_trials)
        est_minus = monte_carlo(x_minus, rng, n_trials)
        b_plus = est_plus.mean(axis=0) - x_plus
        b_minus = est_minus.mean(axis=0) - x_minus
        db_dx[:, j] = (b_plus - b_minus) / (2 * delta[j])
    return db_dx


def posterior_fim(x_true):
    """J = sum_i (1/sigma^2) grad h_i grad h_i^T + P0^-1, evaluated at truth.

    Uses the noiseless body points implied by x_true. For Gaussian noise this
    is the exact Bayesian (posterior) information matrix.
    """
    tx, theta = x_true
    R = rot2(theta)
    sensor_world = np.array([tx, 0.0])
    world_nominal = BODY_POINTS_NOMINAL @ rot2(X_TRUE[1]).T + np.array([X_TRUE[0], 0.0])
    body_at_x = (world_nominal - sensor_world) @ R
    _, J = residuals_and_jac(x_true, body_at_x)
    return (1.0 / SIGMA_POINT ** 2) * (J.T @ J) + P0_INV


def ellipse_points(cov, center, n_sigma=2.0, n=200):
    """2-sigma ellipse polygon for plotting."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    scaled = vecs @ np.diag(n_sigma * np.sqrt(vals)) @ circle
    return scaled + center[:, None]


def main():
    rng = np.random.default_rng(123)

    print("Running Monte Carlo at x_true ...")
    estimates = monte_carlo(X_TRUE, rng)
    mean_hat = estimates.mean(axis=0)
    bias = mean_hat - X_TRUE
    cov_emp = np.cov(estimates.T)

    print("Estimating db/dx ...")
    db_dx = bias_jacobian(X_TRUE, rng)

    print("Computing FIM and bias-corrected CRLB ...")
    J = posterior_fim(X_TRUE)
    crlb_std = np.linalg.inv(J)
    M = np.eye(N_DIM) + db_dx
    crlb_corr_cov = M @ crlb_std @ M.T
    crlb_corr_mse = crlb_corr_cov + np.outer(bias, bias)

    print("\n=== Results at x_true = {} ===".format(X_TRUE))
    print("Empirical bias b      = {}".format(bias))
    print("Empirical Cov(x̂)     =\n{}".format(cov_emp))
    print("Standard CRLB J^-1    =\n{}".format(crlb_std))
    print("db/dx                 =\n{}".format(db_dx))
    print("Bias-corrected (Cov)  =\n{}".format(crlb_corr_cov))
    print("Bias-corrected (MSE)  =\n{}".format(crlb_corr_mse))

    det_emp = np.linalg.det(cov_emp)
    det_std = np.linalg.det(crlb_std)
    det_corr = np.linalg.det(crlb_corr_cov)
    det_mse = np.linalg.det(crlb_corr_mse)
    print("\ndet(empirical Cov)       = {:.3e}".format(det_emp))
    print("det(standard CRLB)       = {:.3e}".format(det_std))
    print("det(bias-corrected Cov)  = {:.3e}".format(det_corr))
    print("det(bias-corrected MSE)  = {:.3e}".format(det_mse))

    # NEES consistency check. For each MC sample compute
    # eps_k = x_true - x̂_k, then NEES_k = eps_k^T P^-1 eps_k.
    # E[NEES] should equal N_DIM (=2) if P is the right cov-of-error matrix.
    # Bias contributes to the error eps, so only the MSE form (which includes
    # b b^T) is expected to be consistent — Cov forms ignore the bias offset.
    from scipy.stats import chi2
    eps = X_TRUE[None, :] - estimates  # (N_MC, 2)
    nees_dict = {}
    for name, P in [
        ("standard CRLB",      crlb_std),
        ("bias-corrected Cov", crlb_corr_cov),
        ("bias-corrected MSE", crlb_corr_mse),
        ("empirical Cov",      cov_emp),
    ]:
        P_inv = np.linalg.inv(P)
        nees = np.einsum("ki,ij,kj->k", eps, P_inv, eps)
        nees_dict[name] = nees

    # 95% interval for the *average* of N_MC chi^2(N_DIM) draws, by CLT:
    # mean ~ N(N_DIM, 2 N_DIM / N_MC). For NEES = 2, N_MC = 6000:
    #   sigma = sqrt(4 / 6000) ≈ 0.026; 95% band ≈ [1.95, 2.05].
    lo_mean = N_DIM - 1.96 * np.sqrt(2 * N_DIM / N_MC)
    hi_mean = N_DIM + 1.96 * np.sqrt(2 * N_DIM / N_MC)
    print("\nNEES consistency (E[NEES] should be {} for a consistent cov)".format(N_DIM))
    print("95% band for the mean over {} trials: [{:.3f}, {:.3f}]".format(
        N_MC, lo_mean, hi_mean))
    for name, nees in nees_dict.items():
        m = nees.mean()
        verdict = "consistent" if lo_mean <= m <= hi_mean else (
            "OVERCONFIDENT" if m > hi_mean else "conservative")
        print("  {:24s}: mean NEES = {:7.3f}   [{}]".format(name, m, verdict))

    # Plot ellipses centered at X_TRUE in (tx, theta) space.
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(estimates[:, 0], estimates[:, 1], s=2, alpha=0.15,
               color="gray", label="MC samples")
    ax.scatter([X_TRUE[0]], [X_TRUE[1]], color="black", marker="x", s=80,
               label="x_true", zorder=5)
    ax.scatter([mean_hat[0]], [mean_hat[1]], color="red", marker="+", s=80,
               label="E[x̂] (biased)", zorder=5)

    for cov, color, label in [
        (cov_emp,        "tab:blue",   "empirical Cov(x̂)"),
        (crlb_std,       "tab:orange", "standard CRLB (J^-1)"),
        (crlb_corr_cov,  "tab:green",  "bias-corrected CRLB"),
    ]:
        pts = ellipse_points(cov, X_TRUE)
        ax.plot(pts[0], pts[1], color=color, linewidth=2.0, label=label)

    ax.set_xlabel("tx (m)")
    ax.set_ylabel("theta (rad)")
    ax.set_title("Biased CRLB vs standard CRLB (2-sigma ellipses)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("auto")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = f"{OUT_DIR}/sim_biased_crlb.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

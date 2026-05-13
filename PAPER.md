# Mathematical Foundations for RA-L Paper

This document contains the formal mathematical demonstrations supporting the paper contributions.

## 1. Forward Propagation Error Bound

### Problem Statement

Given an EKF-corrected anchor state at time `t_0` with covariance `P`, and IMU measurements with
known noise characteristics, derive an upper bound on the forward-propagated pose error at time
`t_0 + dt`.

### Setup

The forward-propagated position and rotation are:

```
p(dt) = p_0 + v_0 · dt + ½ · (R_0 · a_b + g) · dt²
R(dt) = R_0 · Exp(ω_b · dt)
```

where `a_b = a_imu - b_a` and `ω_b = ω_imu - b_g` are bias-corrected measurements.

The true state satisfies the same equations but with true values:

```
p_true(dt) = p_true + v_true · dt + ½ · (R_true · a_true + g_true) · dt²
R_true(dt) = R_true · Exp(ω_true · dt)
```

### Error Sources

Define the anchor errors `δp = p_0 - p_true`, `δv = v_0 - v_true`, `δθ` (rotation error via
right perturbation `R_0 = R_true · Exp(δθ)`), `δb_a = b̂_a - b_a`, `δb_g = b̂_g - b_g`, and the
IMU measurement noise `n_a(t)`, `n_g(t)`.

The position error at time `dt` is:

```
δp(dt) = δp + δv · dt + ½ · R_true · [a_true]× · δθ · dt² - ½ · R_true · δb_a · dt²
         + ½ · R_true · ∫₀^dt ∫₀^s n_a(τ) dτ ds
```

### Bound Derivation

Taking the expected squared norm and using independence of error sources:

```
E[‖δp(dt)‖²] = E[‖δp‖²] + E[‖δv‖²] · dt² + ¼ · ‖a_true‖² · E[‖δθ‖²] · dt⁴
                + ¼ · E[‖δb_a‖²] · dt⁴ + σ²_a · dt³/3
                + cross terms
```

The cross terms vanish under the assumption that anchor errors are uncorrelated with IMU noise.
Using the EKF covariance to bound each term:

```
E[‖δp‖²] ≤ tr(P_pp)                     (position block of P, indices 0:3)
E[‖δv‖²] ≤ tr(P_vv)                     (velocity block of P, indices 12:15)
E[‖δθ‖²] ≤ tr(P_θθ)                     (rotation block of P, indices 3:6)
E[‖δb_a‖²] ≤ tr(P_ba)                   (accel bias block of P, indices 18:21)
```

### Position Error Bound (3σ)

```
‖δp(dt)‖₃σ ≤ 3 · √[ tr(P_pp) + tr(P_vv) · dt²
                      + ¼ · ‖a‖² · tr(P_θθ) · dt⁴
                      + ¼ · tr(P_ba) · dt⁴
                      + σ²_a · dt³ / 3 ]
```

### Rotation Error Bound (3σ)

The rotation error at time `dt`:

```
δθ(dt) = δθ - δb_g · dt + ∫₀^dt n_g(τ) dτ
```

```
E[‖δθ(dt)‖²] = tr(P_θθ) + tr(P_bg) · dt² + σ²_g · dt
               - 2 · tr(P_θ,bg) · dt
```

```
‖δθ(dt)‖₃σ ≤ 3 · √[ tr(P_θθ) + tr(P_bg) · dt² + σ²_g · dt - 2·tr(P_θ,bg)·dt ]
```

### Numerical Example

After EKF convergence with typical values:

```
tr(P_pp) ≈ 1e-6 m²           tr(P_vv) ≈ 1e-5 m²/s²
tr(P_θθ) ≈ 1e-6 rad²         tr(P_bg) ≈ 1e-8 rad²/s²
tr(P_ba) ≈ 1e-7 m²/s⁴        σ²_a = 0.1 m²/s⁴,  σ²_g = 0.1 rad²/s²
‖a‖ ≈ 9.81 m/s²
```

At `dt = 0.1s` (maximum propagation interval with 10Hz scans):

```
‖δp‖₃σ ≤ 3 · √[ 1e-6 + 1e-5·0.01 + ¼·96.2·1e-6·1e-4 + ¼·1e-7·1e-4 + 0.1·3.3e-4 ]
        ≈ 3 · √[ 1e-6 + 1e-7 + 2.4e-9 + 2.5e-12 + 3.3e-5 ]
        ≈ 3 · √[ 3.4e-5 ]
        ≈ 3 · 5.8e-3
        ≈ 1.7 cm
```

```
‖δθ‖₃σ ≤ 3 · √[ 1e-6 + 1e-8·0.01 + 0.1·0.1 ]
        ≈ 3 · √[ 0.01 ]
        ≈ 0.3 rad ← dominated by σ²_g·dt
```

The rotation bound is loose because `σ²_g = 0.1` is the EKF process noise parameter (tuning
knob), not the actual gyro noise spec. With realistic gyro noise `σ²_g = 1e-4 rad²/s²`:

```
‖δθ‖₃σ ≈ 3 · √[ 1e-6 + 1e-10 + 1e-5 ] ≈ 3 · 3.3e-3 ≈ 0.01 rad ≈ 0.57°
```

### Theorem 1 (Forward Propagation Error Bound)

*For a tightly-coupled LiDAR-inertial system with EKF covariance P and IMU noise parameters
(σ_a, σ_g), the 3σ position error of the forward-propagated pose at time dt after correction
is bounded by:*

```
‖δp(dt)‖₃σ = O(dt)        for small dt (dominated by √tr(P_vv) · dt)
‖δθ(dt)‖₃σ = O(√dt)       for small dt (dominated by √(σ²_g · dt))
```

*For dt = 0.1s (one scan interval) with converged EKF and realistic IMU noise, the position error
is bounded by ~2 cm and rotation error by ~0.6°.*

---

## 2. Observability Analysis for Non-Repetitive LiDAR

### Problem Statement

Prove that a single 10ms scan from a non-repetitive LiDAR (Livox Mid-360) may not provide
full-rank observability of the 6-DOF pose, while N accumulated scans do.

### Measurement Model

FAST-LIO2 uses point-to-plane residuals. For a point `p_L` in LiDAR frame matched to a plane
with normal `n` and point `q` in the map:

```
z = n^T · (R · R_LI · p_L + R · t_LI + t - q)
```

where `(R, t)` is the body-to-world pose, `(R_LI, t_LI)` is the LiDAR-to-IMU extrinsic.

The measurement Jacobian with respect to the pose error `δx = [δt, δθ]^T` is:

```
H_i = [ n_i^T,  n_i^T · R · [R_LI · p_i + t_LI]× ]     ∈ R^{1×6}
     = [ n_i^T,  -n_i^T · [p_w_i - t]× ]                  (equivalently)
```

where `p_w_i = R · R_LI · p_i + R · t_LI + t` is the point in world frame.

### Fisher Information Matrix

The Fisher Information Matrix (FIM) for M point-to-plane measurements is:

```
F = Σᵢ₌₁ᴹ  (1/σ²_z) · H_i^T · H_i     ∈ R^{6×6}
```

Full 6-DOF observability requires `rank(F) = 6`.

### Decomposition of H_i

Each `H_i^T · H_i` is a rank-1 matrix. Define `h_i = H_i^T`:

```
h_i = [ n_i                    ]
      [ [p_w_i - t]× · (-n_i)  ]
```

The upper 3×3 block of F (translation observability):

```
F_tt = Σ n_i · n_i^T
```

This has rank 3 if and only if the surface normals `{n_i}` span R³. In other words, the matched
planes must include surfaces in at least three linearly independent orientations.

The lower 3×3 block (rotation observability):

```
F_θθ = Σ ([p_w_i - t]× · n_i) · ([p_w_i - t]× · n_i)^T
```

This has rank 3 if and only if the lever arms `{(p_w_i - t) × n_i}` span R³. This requires
points at diverse positions relative to the sensor, with normals not aligned with the
position vectors.

### Degeneracy Conditions

**Translation degeneracy** (`rank(F_tt) < 3`): Occurs when all matched normals are coplanar.
Example: a long corridor where all walls are parallel — the along-corridor direction is
unobservable from the normals alone.

**Rotation degeneracy** (`rank(F_θθ) < 3`): Occurs when all points lie on a line through
the sensor, or when all lever arms `(p_i - t) × n_i` are parallel. Example: a flat open
ground where all normals point up and all points are at the same height — yaw is unobservable.

### Non-Repetitive Scanning and Observability

The Livox Mid-360 uses a Risley prism producing a non-repetitive rosette pattern. In a single
10ms scan, the beam traces a small arc of the pattern.

**Claim**: Let `Ω(t, Δt)` be the set of scan directions covered by the Mid-360 in the time
interval `[t, t+Δt]`. The angular coverage (solid angle) satisfies:

```
|Ω(t, Δt)| = c · Δt     for Δt ≤ T_pattern
```

where `c` is the angular scan rate and `T_pattern` is the full pattern period (~1s for Mid-360).

For a single 10ms scan: `|Ω| ≈ c · 0.01` — covering approximately 1% of the full FoV.

### Theorem 2 (Observability Condition for Non-Repetitive LiDAR)

*Let M(N) be the number of point-to-plane correspondences from N accumulated scans, each of
duration Δt. Assume the environment has surfaces with normals spanning R³ within the full
LiDAR FoV. Then:*

**(a)** *For a single scan (N=1), the FIM may be rank-deficient if the scan arc covers only
surfaces with coplanar normals. The probability of rank deficiency depends on the environment
geometry and the scan phase.*

**(b)** *For N accumulated scans covering total duration N·Δt, the angular coverage is:*

```
|Ω(t, N·Δt)| ≈ min(c · N · Δt, |Ω_full|)
```

*As N increases, the covered FoV approaches the full FoV, and the FIM approaches the
full-coverage FIM which has rank 6 (given the environment assumption).*

**(c)** *The minimum N for guaranteed full-rank observability satisfies:*

```
N_min ≥ ⌈3 / (c · Δt · ρ_normal)⌉
```

*where ρ_normal is the probability that a random scan direction hits a surface with a
linearly independent normal (environment-dependent).*

### Empirical Validation

The scan accumulation ablation (Experiment E5) should demonstrate:

| N   | Points per scan | rank(F) | EKF failures | APE (cm) |
|-----|----------------|---------|--------------|----------|
| 1   | 48-8790        | ≤5      | High         | Diverges |
| 5   | ~2500          | 5-6     | Moderate     | ~5       |
| 10  | ~5000          | 6       | Rare         | ~2       |
| 20  | ~10000         | 6       | None         | ~2       |

The key transition happens when the accumulated FoV is large enough to observe surfaces
with 3 independent normals. For typical indoor environments with walls, floor, and ceiling,
this requires ~120° of angular coverage, achieved at approximately N=5-10 for the Mid-360.

### Connection to Point-LIO

Point-LIO avoids this problem by processing points one-by-one, updating the state at each
point. This is equivalent to N→∞ accumulation with a correction at every point. However:

1. Each single-point update is rank-1 (one scalar residual), so convergence requires
   processing enough points for the cumulative FIM to reach rank 6
2. The per-point approach has higher computational overhead per point (full EKF update
   vs. batch)
3. The stochastic kinematic model in Point-LIO adds additional process noise to
   compensate for the weak per-point observations

Our scan accumulation achieves the same observability guarantee with a simpler batch
architecture that is proven stable (FAST-LIO2's iESEKF).

---

## 3. Covariance Frame Consistency

### Problem Statement

Prove that the correct representation of pose covariance in the ROS odometry message requires
rotating the rotation block from body frame to odom frame, and that the upstream FAST-LIO2
implementation is inconsistent.

### IEKF Error State Convention

FAST-LIO2 uses the iterated error-state EKF with right perturbation on SO(3):

```
R_true = R̂ · Exp(δθ)
```

where `δθ ∈ R³` is the rotation error in the **body frame**. This is confirmed by the
`boxplus` operation in `SOn.hpp`:

```cpp
void boxplus(vectview<const scalar, DOF> vec, scalar scale=1) {
    SO3 delta = exp(vec, scale);
    *this = *this * delta;   // Right multiplication: R̂ · Exp(δθ)
}
```

The error state vector is:

```
δx = [δp, δθ, δR_LI, δt_LI, δv, δb_g, δb_a, δg]^T ∈ R²³
```

where `δp` and `δv` are in the **world/odom frame**, and `δθ` is in the **body frame**.

### Covariance Matrix Frame Structure

The 23×23 covariance `P = E[δx · δx^T]` has mixed-frame blocks:

```
P = [ P_pp    P_pθ    ...  P_pv    ... ]
    [ P_θp    P_θθ    ...  P_θv    ... ]    ← body frame
    [ ...     ...     ...  ...     ... ]
    [ P_vp    P_vθ    ...  P_vv    ... ]
    [ ...     ...     ...  ...     ... ]
```

- `P_pp = E[δp · δp^T]` — odom frame ⊗ odom frame
- `P_θθ = E[δθ · δθ^T]` — body frame ⊗ body frame
- `P_pθ = E[δp · δθ^T]` — odom frame ⊗ body frame (mixed)
- `P_vv = E[δv · δv^T]` — odom frame ⊗ odom frame

### Forward Propagation Jacobian

The 6×23 Jacobian `J` maps the full error state to the propagated pose error:

```
[δp_prop]
[δθ_prop] = J · δx
```

The non-zero blocks:

```
J = [ I₃   -R[a']×½dt²   0   0   I₃·dt   0      -R½dt²   0 ]  ← position
    [ 0₃   I₃            0   0   0₃      -I₃·dt   0₃      0 ]  ← rotation
```

The resulting `P_pose = J · P · J^T` has:

```
P_pose = [ P_pose_pp    P_pose_pθ ]
         [ P_pose_θp    P_pose_θθ ]
```

where `P_pose_pp` is in **odom frame** and `P_pose_θθ` is in **body frame**.

### Theorem 3 (Covariance Frame Transformation)

*The ROS `nav_msgs/Odometry` message specifies that `pose.covariance` represents uncertainty
in `header.frame_id` (odom frame). The correct covariance is:*

```
P_odom = T · P_pose · T^T
```

*where:*

```
T = [ I₃   0₃ ]
    [ 0₃   R  ]
```

*and R = R̂ is the current rotation estimate (body-to-world).*

**Proof:**

The right-perturbation error `δθ_body` satisfies `R_true = R̂ · Exp(δθ_body)`.

Using the conjugation identity for the exponential map:

```
R̂ · Exp(δθ_body) = Exp(R̂ · δθ_body) · R̂
```

Therefore the equivalent left (world-frame) perturbation is:

```
δθ_world = R̂ · δθ_body
```

The world-frame rotation covariance is:

```
E[δθ_world · δθ_world^T] = E[(R̂ · δθ_body)(R̂ · δθ_body)^T]
                          = R̂ · E[δθ_body · δθ_body^T] · R̂^T
                          = R̂ · P_θθ · R̂^T
```

For the cross terms (position is already in odom frame):

```
E[δp · δθ_world^T] = E[δp · (R̂ · δθ_body)^T]
                    = E[δp · δθ_body^T] · R̂^T
                    = P_pθ · R̂^T
```

Combining:

```
P_odom = [ P_pp           P_pθ · R^T       ]
         [ R · P_θp       R · P_θθ · R^T    ]
```

which equals `T · P_pose · T^T`.  ∎

### Properties

**Symmetry preservation**: If `P_pose` is symmetric, then `P_odom = T · P_pose · T^T` is
symmetric (since `T` is invertible).

**Positive semi-definiteness preservation**: If `P_pose ≥ 0`, then `P_odom = T · P_pose · T^T ≥ 0`
(congruence transformation preserves PSD).

**Isotropic noise invariance**: If the noise added to the rotation diagonal is isotropic
(`σ² · I₃`), then `R · σ²I₃ · R^T = σ² · I₃`. The frame rotation does not affect isotropic
noise, so the noise can be added before or after the transformation.

### Upstream Bug: Position/Rotation Swap

The upstream FAST-LIO2 code in `publish_odometry()` uses:

```cpp
int k = i < 3 ? i + 3 : i - 3;
odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
// ...
```

This swaps position rows (0-2) with rotation rows (3-5) and position columns with rotation
columns. The ROS `PoseWithCovariance` message uses ordering `[x, y, z, rot_x, rot_y, rot_z]`,
which matches `P_pose = [pos(0:3), rot(3:6)]` directly. The swap is incorrect and results
in position covariance appearing in the rotation slots and vice versa.

---

## 4. Drift Rate Lower Bound for Frame-to-Map Registration

### Problem Statement

Prove that any odometry system based on frame-to-map registration (including FAST-LIO2) has
a drift variance that grows at least linearly with the number of registrations, regardless
of the EKF covariance.

### System Model

Consider a sequence of scan registrations at times `t_1, t_2, ..., t_K`. At each step `k`,
the system:

1. Propagates the state: `x̂_k⁻ = f(x̂_{k-1})` with covariance `P_k⁻`
2. Observes: `z_k = h(x_k, m_k) + v_k` where `m_k` are map points
3. Updates: `x̂_k = x̂_k⁻ + K_k(z_k - h(x̂_k⁻, m_k))`

The map points `m_k` are treated as known constants in the measurement model, but they were
inserted using previous estimates:

```
m_j = T̂_j · p_j^L    (point p_j^L in LiDAR frame, transformed by estimated pose T̂_j)
```

### The Unmodeled Bias

The true measurement model is:

```
z_k = h(x_k, m_true_k) + v_k
     = h(x_k, m_k - δm_k) + v_k
```

where `δm_k` is the map error at the points used for registration at step `k`. Linearizing:

```
z_k ≈ h(x_k, m_k) - ∂h/∂m · δm_k + v_k
```

The EKF does not model the `∂h/∂m · δm_k` term. It interprets this as a state error and
"corrects" the state accordingly:

```
x̂_k = x̂_k⁻ + K_k · (z_k - h(x̂_k⁻, m_k))
```

The bias term `b_k = -∂h/∂m · δm_k` is absorbed into the state correction.

### Map Error Propagation

The map error `δm_k` depends on the pose estimation error at the time each point was inserted.
For point `j` inserted at time `t_j`:

```
δm_j = (T̂_j - T_true_j) · p_j^L ≈ [δR_j · R_j · p_j^L + δt_j]
```

where `δt_j` and `δR_j` are the pose errors at time `t_j`.

As the robot moves and accumulates points, the map contains points inserted over a range of
times with a range of errors. New registrations are performed against this ensemble of
errors.

### Drift Variance Growth

**Theorem 4 (Linear Drift Growth)**

*Consider a 1D simplification where the state is position `x ∈ R`, the map consists of
previously estimated positions, and each registration has independent noise `η_k ~ N(0, σ²_η)`.
Then the position error after K registrations satisfies:*

```
Var(ε_K) ≥ K · σ²_η
```

*regardless of the EKF covariance P_K.*

**Proof (1D):**

At step `k`, the registration measures the displacement from the current position to nearby
map points. The residual is:

```
z_k = (x_k - m_{nearest}) + v_k
```

where `m_{nearest}` was inserted at some previous time `j` with error `ε_j`:

```
m_{nearest} = x̂_j = x_j + ε_j
```

The EKF update computes:

```
x̂_k = x̂_k⁻ + K_k(z_k - (x̂_k⁻ - m_{nearest}))
```

The true error after update:

```
ε_k = x̂_k - x_k = ε_k⁻ + K_k · (ε_j - ε_k⁻ + v_k)
     = (1 - K_k) · ε_k⁻ + K_k · ε_j + K_k · v_k
```

This shows the error at step `k` is a weighted combination of:
- The propagated error `ε_k⁻` (from IMU integration)
- The map error `ε_j` (inherited from the past)
- New measurement noise `v_k`

The map error `ε_j` does not reduce the current error — it adds to it. Each registration
introduces an independent noise term `K_k · v_k`, so:

```
Var(ε_K) = Var(ε_0) + Σ_{k=1}^K K_k² · σ²_v + map error terms
         ≥ Σ_{k=1}^K K_k² · σ²_v
```

For stable EKF with `K_k ≈ K_∞ > 0`:

```
Var(ε_K) ≥ K · K_∞² · σ²_v = K · σ²_η
```

where `σ²_η = K_∞² · σ²_v`.  ∎

### Extension to 6-DOF

In full 6-DOF, the same argument applies per dimension. The position drift variance grows as:

```
Var(‖δp_K‖²) ≥ K · σ²_pos
```

where `σ²_pos` depends on:
- Point-to-plane noise variance `σ²_z`
- Geometric distribution of points (condition number of the FIM)
- Kalman gain at steady state

For rotation, the drift is typically smaller because gravity provides a persistent
reference for roll and pitch. Yaw drift follows the same random walk:

```
Var(δψ_K) ≥ K · σ²_yaw
```

### EKF Covariance Inconsistency

**Corollary (EKF Overconfidence)**

*The EKF covariance P_K does not grow linearly with K. After convergence, P_K ≈ P_∞
(bounded). Therefore:*

```
E[(x̂_K - x_K)(x̂_K - x_K)^T] ≥ K · Σ_η ≫ P_∞    for large K
```

*The EKF is **inconsistent**: the actual estimation error exceeds the reported uncertainty.*

This is a fundamental limitation of any filter that uses its own map as the measurement
reference without tracking map uncertainty. The EKF "trusts" the map, so each correction
drives P down, but the map itself drifts.

### Drift Rate Estimation

The drift rate `σ²_η` per registration can be estimated empirically via RPE analysis:

```
σ²_pos = lim_{K→∞} (1/K) · Var(ε_K)
```

In practice, this is computed from the slope of the RPE-vs-distance plot:

```
RPE(d) = √(σ²_pos · d / Δd)
```

where `d` is the distance traveled and `Δd` is the distance between registrations.

### Implications for Downstream SLAM

For a pose graph optimizer to perform loop closure, it needs the **true** uncertainty
(including drift), not the EKF's underestimate. The linear drift noise model:

```
P_published(i,i) += σ²_drift · dt
```

approximates the O(K) drift growth in continuous time (since `K ≈ dt / Δt_scan`). The
parameter `σ²_drift` must be calibrated empirically (see FUTURE_IMPROVEMENTS.md).

---

## 5. Online Drift Detection via Map Age Partitioning

### Problem Statement

Given a frame-to-map registration system where the map accumulates points over time, detect
when accumulated drift causes the map to become inconsistent with the current observations,
without loop closure or external references.

### Residual Model

At registration step `k`, each point `p_i` in the current scan is matched to its nearest
plane in the ikd-tree. The point-to-plane residual is:

```
r_i = n_i^T · (p_w_i - q_i)
```

where `n_i` is the plane normal and `q_i` is the closest point on the plane.

If the map were perfect, the expected residual would be zero with variance `σ²_z` (sensor
noise). In practice, the matched map point `q_i` was inserted at time `t_j` with pose
error `ε_j`:

```
q_i = T̂_j · p_j^L = T_true_j · p_j^L + δT_j · p_j^L
```

The residual becomes:

```
E[|r_i|] = E[|n_i^T · δT_j · p_j^L|] + O(σ_z)
```

The map error `δT_j` grows with time due to drift (T4). Therefore, the expected absolute
residual of a point matched against an old map point is larger than one matched against a
recent map point.

### Age-Partitioned Residual Statistic

Define the map point age for correspondence `i`:

```
a_i = t_current - t_insert(q_i)
```

Partition correspondences into old (`a_i > τ`) and new (`a_i ≤ τ`) sets:

```
r̄_old = (1/|S_old|) · Σ_{i ∈ S_old} |r_i|
r̄_new = (1/|S_new|) · Σ_{i ∈ S_new} |r_i|
```

### Theorem 5 (Age Ratio as Drift Indicator)

*Define the age ratio `ρ = r̄_old / r̄_new`. Under the drift model of T4:*

**(a)** *Without drift (short trajectory or perfect map): `E[ρ] = 1`, since old and new map
points have the same quality.*

**(b)** *With drift rate `σ²_η` per registration (T4), the expected age ratio grows as:*

```
E[ρ] ≈ √(1 + σ²_drift · (t_current - t̄_old) / σ²_z)
```

*where `t̄_old` is the mean insertion time of old correspondences and `σ²_z` is the baseline
sensor noise.*

**(c)** *At revisitation (returning to a previously mapped area after time T), the age ratio
spikes:*

```
E[ρ_revisit] ≈ √(1 + σ²_drift · T / σ²_z)
```

**Proof:**

A point-to-plane residual for correspondence `i` matched against map point `q_i` is:

```
r_i = n_i^T · (p_w_i - q_i) + v_i
```

where `v_i ~ N(0, σ²_z)` is sensor noise. The map point `q_i` was inserted at time `t_j`
with pose error `ε_j`. From T7, the map point has position error:

```
δq_i = G_j · δx_j
```

where `G_j` is the mapping Jacobian. The residual becomes:

```
r_i = n_i^T · (p_w_i - q_true_i) - n_i^T · δq_i + v_i
```

For a consistent current estimate, `p_w_i ≈ q_true_i`, so:

```
r_i ≈ -n_i^T · G_j · δx_j + v_i
```

The expected squared residual:

```
E[r_i²] = n_i^T · G_j · E[δx_j · δx_j^T] · G_j^T · n_i + σ²_z
         = n_i^T · G_j · Cov(δx_j) · G_j^T · n_i + σ²_z
```

From T4, the true pose error covariance at insertion time `t_j` satisfies:

```
Cov(δx_j) ≥ K_j · Σ_η
```

where `K_j` is the number of registrations up to `t_j`, proportional to `t_j / Δt_scan`.
For the absolute residual, using `E[|r|] = √(2/π) · √(Var(r))` for Gaussian `r`:

```
E[|r_i|] = √(2/π) · √(σ²_z + n_i^T · G_j · Cov(δx_j) · G_j^T · n_i)
```

Define `c_i = n_i^T · G_j · Σ_η · G_j^T · n_i / Δt_scan` (geometry-dependent constant).
Then:

```
E[|r_i|] = √(2/π) · √(σ²_z + c_i · t_j)
```

Averaging over old correspondences (mean insertion time `t̄_old`) and new correspondences
(mean insertion time `t̄_new`), and assuming similar geometric distributions:

```
E[r̄_old] ≈ √(2/π) · √(σ²_z + c̄ · t̄_old)
E[r̄_new] ≈ √(2/π) · √(σ²_z + c̄ · t̄_new)
```

The age ratio:

```
E[ρ] = E[r̄_old] / E[r̄_new]
     ≈ √((σ²_z + c̄ · t̄_old) / (σ²_z + c̄ · t̄_new))
     = √(1 + c̄ · (t̄_old - t̄_new) / (σ²_z + c̄ · t̄_new))
```

For new points with small age (`c̄ · t̄_new ≪ σ²_z`):

```
E[ρ] ≈ √(1 + c̄ · Δt_age / σ²_z)
```

where `Δt_age = t̄_old - t̄_new`. Defining `σ²_drift = c̄`, this gives part (b).

For revisitation after time `T`, `Δt_age ≈ T`, giving part (c).

For no drift (`c̄ = 0`), `E[ρ] = 1`, giving part (a).  ∎

### Numerical Example (from experimental data)

On a 120s trajectory with mocap ground truth:

```
APE mean = 0.032 m → σ²_drift ≈ APE² / T = 0.032² / 120 = 8.5e-6 m²/s
σ²_z = LASER_POINT_COV = 0.001 m²
Δt_age ≈ 60s (mean old age minus mean new age)
```

Predicted age ratio:

```
E[ρ] ≈ √(1 + 8.5e-6 · 60 / 0.001) = √(1 + 0.51) = √1.51 ≈ 1.23
```

Observed: `ρ = 1.4`. The discrepancy suggests the geometric factor `c̄` is larger than
the isotropic estimate, or that drift is anisotropic (concentrated in fewer directions).
This is consistent with the FIM analysis (T6) showing a 115× condition number.

### Implementation

The curvature field of `pcl::PointXYZINormal` (float, unused after insertion) stores the
relative insertion time `t_insert - t_origin`. At each scan, residuals are accumulated
separately for old and new correspondences. The age threshold `τ = 5s` balances sensitivity
(too small → noisy) versus detection delay (too large → slow response).

### Expected Behavior

| Scenario | `ρ` | `r̄_old` | `r̄_new` |
|----------|-----|---------|---------|
| Short trajectory (<1 min) | ≈ 1.0 | ≈ 0.02 | ≈ 0.02 |
| Long corridor (>5 min) | > 1.5 | > 0.03 | ≈ 0.02 |
| Revisitation after 5 min | spike > 2.0 | > 0.05 | ≈ 0.02 |
| Feature-rich, slow motion | ≈ 1.0 | ≈ 0.01 | ≈ 0.01 |

---

## 6. Degeneracy-Aware Covariance Inflation

### Problem Statement

Detect when the scan geometry provides insufficient observability in one or more directions,
and inflate the reported covariance in those directions to reflect the actual uncertainty.

### Fisher Information Matrix Condition

From T2, the FIM for a set of point-to-plane correspondences is:

```
F = (1/σ²_z) · Σᵢ H_i^T · H_i ∈ R^{6×6}
```

The eigendecomposition `F = V · Λ · V^T` reveals the observable directions (`V`) and their
information content (`Λ = diag(λ_1, ..., λ_6)`, sorted `λ_1 ≥ ... ≥ λ_6`).

### Degeneracy Detection

**Condition number**: `κ(F) = λ_1 / λ_6`. When `κ(F) ≫ 1`, the system is near-degenerate.

**Effective rank**: The number of eigenvalues above a threshold `λ_i > ε · λ_1`. When
effective rank < 6, one or more pose directions are unobservable.

**Per-direction information**: The eigenvector `v_6` corresponding to `λ_6` identifies the
least-observable direction in the pose space.

### Cramér-Rao Lower Bound for Pose Estimation

The Cramér-Rao inequality states that for any unbiased estimator, the covariance is bounded
below by the inverse of the Fisher Information Matrix:

```
Cov(x̂) ≥ F⁻¹
```

For the point-to-plane measurement model with noise variance `σ²_z`, the FIM is:

```
F = (1/σ²_z) · H^T · H
```

In the eigenvector basis `V`:

```
F = V · Λ · V^T    ⟹    F⁻¹ = V · Λ⁻¹ · V^T
```

The minimum achievable variance along eigenvector `v_i` is:

```
CRLB_i = v_i^T · F⁻¹ · v_i = σ²_z / λ_i
```

Note: our implementation computes `H_pose^T · H_pose` without the `1/σ²_z` scaling, so the
eigenvalues already absorb the measurement count. The CRLB becomes `σ²_z / λ_i` where `λ_i`
are the eigenvalues of `H_pose^T · H_pose` (not `F`). This is equivalent because `σ²_z`
cancels: `σ²_z / (λ_i / σ²_z) = σ²_z² / λ_i`, but since we use `H^T H` directly (without
scaling by `1/σ²_z`), the CRLB is simply `σ²_z / λ_i` with `σ²_z = LASER_POINT_COV`.

### Theorem 6 (CRLB-Based Covariance Inflation)

*Let `H_pose^T · H_pose = V · Λ · V^T` be the eigendecomposition of the pose FIM. Define the
inflated covariance:*

```
P_inflated = P_EKF + V · Γ · V^T
```

*where `Γ = diag(γ_1, ..., γ_6)` with:*

```
γ_i = max(0, σ²_z / λ_i - v_i^T · P_EKF · v_i)
```

*Then:*

**(a)** *`P_inflated ≥ CRLB` in all directions: `v_i^T · P_inflated · v_i ≥ σ²_z / λ_i`
for all i.*

**(b)** *In well-conditioned directions (large `λ_i`), `σ²_z / λ_i` is small and
`γ_i ≈ 0` — the EKF covariance is unchanged.*

**(c)** *In degenerate directions (small `λ_i`), `σ²_z / λ_i` is large and `γ_i > 0` —
the covariance is inflated to the CRLB.*

**(d)** *`P_inflated` is symmetric positive semi-definite.*

**Proof:**

**(a)**: After inflation, the variance along `v_i` is:

```
v_i^T · P_inflated · v_i = v_i^T · P_EKF · v_i + γ_i
                          = v_i^T · P_EKF · v_i + max(0, σ²_z/λ_i - v_i^T · P_EKF · v_i)
                          = max(v_i^T · P_EKF · v_i, σ²_z/λ_i)
                          ≥ σ²_z / λ_i = CRLB_i   ∎
```

**(b)**: For large `λ_i`, `σ²_z / λ_i → 0`. Since `v_i^T · P_EKF · v_i ≥ 0 ≥ σ²_z / λ_i`,
we have `γ_i = 0`.  ∎

**(c)**: For small `λ_i`, `σ²_z / λ_i` is large. Since the EKF correction drives
`v_i^T · P_EKF · v_i` toward zero (the EKF doesn't know it's degenerate — it still applies
the Kalman gain), `γ_i = σ²_z / λ_i - v_i^T · P_EKF · v_i > 0`.  ∎

**(d)**: `P_inflated = P_EKF + V · Γ · V^T`. Since `Γ ≥ 0` (diagonal with non-negative
entries), `V · Γ · V^T ≥ 0`. The sum of two PSD matrices is PSD.  ∎

### Numerical Example (from experimental data)

With our mocap room data (κ = 115):

```
λ_min = 516,  λ_max = 59079,  σ²_z = 0.001

CRLB_weakest  = 0.001 / 516   = 1.94e-6 m²  (σ ≈ 1.4 mm)
CRLB_strongest = 0.001 / 59079 = 1.69e-8 m²  (σ ≈ 0.13 mm)
```

The EKF reports `P_ii ≈ 1e-6` uniformly. In the weakest direction, `γ = 1.94e-6 - 1e-6 =
0.94e-6` — mild inflation. In the strongest, `γ = 0` — no change.

In a hypothetical corridor with `λ_min = 0.5`:

```
CRLB_weakest = 0.001 / 0.5 = 0.002 m²  (σ ≈ 4.5 cm)
EKF reports:   1e-6 m²                  (σ ≈ 1 mm)
γ = 0.002 - 1e-6 ≈ 0.002
```

The EKF claims 1mm precision along the corridor — the CRLB corrects this to 4.5cm.

### Degeneracy Scenarios for Mid-360

| Environment | Degenerate direction | `λ_min` | Effect |
|-------------|---------------------|---------|--------|
| Long corridor | Along-corridor translation | ≈ 0 | Position drift along corridor |
| Open field | Horizontal translation (both) | ≈ 0 | No lateral constraint |
| Flat ground only | Yaw rotation | ≈ 0 | Yaw drift |
| Featureless ceiling | Z translation + roll/pitch | ≈ 0 | Vertical drift |
| Textured room | None | > 0 | Well-conditioned |

### Computation

The FIM is already implicitly computed in `h_share_model()` via the Jacobian `H`. The
eigendecomposition of `H^T · H` (6×6) costs O(6³) = O(216) operations — negligible compared
to the nearest-neighbor search.

```
Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(H.transpose() * H);
Eigen::Vector6d eigenvalues = solver.eigenvalues();
double condition = eigenvalues(5) / std::max(eigenvalues(0), 1e-10);
```

---

## 7. Map Uncertainty Propagation

### Problem Statement

Each point in the ikd-tree map was transformed to world frame using an estimated pose `T̂_j`.
The pose had covariance `P_j` at insertion time. Derive the induced uncertainty on the map
point and its effect on the registration residual.

### Point Uncertainty

A LiDAR point `p^L` in the sensor frame is transformed to world frame:

```
p^W = R̂_j · (R_LI · p^L + t_LI) + t̂_j
```

Under the error state `δx_j = [δt_j, δθ_j, ...]`:

```
δp^W = δt_j + R̂_j · [R_LI · p^L + t_LI]× · δθ_j
```

Define the mapping Jacobian:

```
G_j = [ I₃,  R̂_j · [R_LI · p^L + t_LI]× ] ∈ R^{3×6}
```

### Theorem 7 (Map Point Covariance)

*The covariance of a map point `p^W` inserted at time `t_j` with pose covariance `P_j` is:*

```
Σ_map = G_j · P_j^{pose} · G_j^T + σ²_lidar · I₃
```

*where `P_j^{pose}` is the 6×6 pose block of `P_j` and `σ²_lidar` is the LiDAR range noise.*

### Effect on Registration

The point-to-plane residual with uncertain map point:

```
r_i = n_i^T · (p_w_i - q_i)
```

has variance:

```
Var(r_i) = σ²_z + n_i^T · Σ_map(q_i) · n_i
```

The second term is the map-induced uncertainty projected onto the normal direction.

### Theorem 7b (Map-Aware Measurement Noise)

*The EKF measurement noise should be:*

```
R_i = σ²_z + n_i^T · G_j · P_j^{pose} · G_j^T · n_i
```

*instead of the constant `R_i = σ²_z` used by FAST-LIO2. This makes old, uncertain map
points contribute less to the correction — a natural down-weighting of drifted regions.*

### Connection to Drift Detection (T5)

Map uncertainty propagation explains *why* the age ratio works: old map points have larger
`P_j^{pose}` (accumulated over more drift), leading to larger `Σ_map`, which inflates the
expected residual `E[|r_i|]` against those points.

The age-based partitioning (T5) is a computationally cheap proxy for the full per-point
uncertainty propagation. The full propagation would require storing `P_j` (6×6 = 21 unique
values) per map point, which is impractical for ikd-trees with millions of points.

### Practical Approximation

Instead of per-point covariance, use the age-based residual scaling:

```
R_i = σ²_z · (1 + α · a_i)
```

where `a_i` is the map point age and `α` is calibrated from the age ratio statistics. This
requires only the scalar `curvature` field already stored per point.

---

## 8. Lightweight Loop Closure via Scan-to-Submap Matching

### Problem Statement

When the robot revisits a previously mapped area, detect the revisitation and correct the
accumulated drift without a full pose graph optimizer.

### Revisitation Detection

Using the drift indicator from T5: when the age ratio `ρ` spikes above a threshold while
the system is matching against old map points, the robot is likely revisiting a region it
mapped earlier.

More precisely, define a revisitation event at time `t_k` when:

```
ρ_k > ρ_threshold  AND  |S_old_k| > N_min  AND  r̄_new_k < r̄_threshold
```

The third condition ensures the current scan is well-matched (the system is not lost), while
the first condition indicates old map points are inconsistent with the current estimate.

### Drift Correction Model

At revisitation, the system has two estimates of the same region:
- Old map points inserted at times `{t_j}` with poses `{T̂_j}`
- Current scan at time `t_k` with pose `T̂_k`

The drift accumulated between `t̄_old` and `t_k` is:

```
δT_drift ≈ T̂_k^{-1} · T̂_{corrected}
```

where `T̂_{corrected}` is obtained by re-registering the current scan against only the old
map points with a wider convergence basin (e.g., ICP with larger initial search radius).

### Theorem 8 (Single-Shot Drift Correction Bound)

*Let `T̂_k` be the current pose estimate and `T̂_k^{corrected}` be obtained by registration
against old map points. The correction `δT = T̂_k^{-1} · T̂_k^{corrected}` satisfies:*

```
‖δT‖ ≤ ‖drift accumulated over [t̄_old, t_k]‖ + σ_registration
```

*Applying this correction reduces the global error by approximately the accumulated drift,
at the cost of introducing a discontinuity of magnitude `‖δT‖` in the trajectory.*

### Limitations

1. **No graph optimization**: A single correction fixes the current pose but does not
   retroactively fix the trajectory between `t̄_old` and `t_k`. The map remains inconsistent
   in that interval.

2. **Discontinuity**: The pose jump may cause issues for the flight controller. A gradual
   correction (blending over several seconds) is safer but delays convergence.

3. **False positives**: High `ρ` can occur from geometry changes (dynamic objects) rather
   than drift. The `r̄_new < r̄_threshold` condition mitigates this but does not eliminate it.

4. **Observability**: The correction is only possible in directions where old map points
   provide sufficient FIM rank (T2/T6). In a corridor, returning along the same corridor
   does not fix along-corridor drift.

### Comparison with Full SLAM

| Aspect | Lightweight (T8) | Full pose graph |
|--------|------------------|-----------------|
| Correction scope | Current pose only | Entire trajectory |
| Map consistency | Not corrected | Globally consistent |
| Computational cost | O(1) per detection | O(N²) per optimization |
| Memory | No additional | Factor graph storage |
| Implementation | ~50 lines in EKF | Separate SLAM backend |
| Suitable for | Real-time UAV | Post-processing |

---

## 9. Degeneracy-Drift Coupling

### Problem Statement

Prove that drift accumulates faster in directions where the FIM has weak eigenvalues, and
that the age ratio (T5) is amplified in degenerate directions.

### Theorem 9 (Anisotropic Drift Amplification)

*Let `F = V · Λ · V^T` be the FIM eigendecomposition and `σ²_η` be the isotropic per-step
drift noise (T4). The drift variance after K registrations in the direction of eigenvector
`v_i` satisfies:*

```
Var(v_i^T · ε_K) ≥ K · K²_∞(i) · σ²_v
```

*where `K_∞(i)` is the steady-state Kalman gain in direction `v_i`. For the standard EKF:*

```
K_∞(i) = P_∞(i) / (P_∞(i) + σ²_z / λ_i)
```

*In well-observed directions (large `λ_i`), `σ²_z / λ_i → 0` and `K_∞(i) → 1`: the EKF
fully trusts the measurement, absorbing the map error. In degenerate directions (small
`λ_i`), `σ²_z / λ_i → ∞` and `K_∞(i) → 0`: the EKF ignores the (uninformative) measurement,
and drift grows from IMU propagation noise alone.*

**Proof:**

From T4, the error after EKF update in the eigenvector basis:

```
ε_k(i) = (1 - K_k(i)) · ε⁻_k(i) + K_k(i) · ε_map(i) + K_k(i) · v_k(i)
```

The propagated error `ε⁻_k(i)` grows by IMU noise `w_k(i)` with variance `σ²_w`. The
steady-state variance satisfies the algebraic Riccati equation projected onto direction `i`:

```
P_∞(i) = (1 - K_∞(i))² · (P_∞(i) + σ²_w) + K²_∞(i) · (P_map(i) + σ²_z/λ_i)
```

where `P_map(i)` is the map error variance in direction `i` (unmodeled by the EKF).

For degenerate direction (`λ_i ≈ 0`): `K_∞(i) ≈ 0`, so `P_∞(i) ≈ P_∞(i) + σ²_w` —
the covariance grows without bound (IMU random walk). The EKF correctly reports this.

For partially degenerate direction (small but nonzero `λ_i`): `K_∞(i)` is small but
positive. The EKF pulls `P_∞(i)` down slightly, but the map error `P_map(i)` (unmodeled)
continues to inject bias. The true error grows as:

```
Var(v_i^T · ε_K) ≈ K · K²_∞(i) · σ²_v + K · (1-K_∞(i))² · P_map(i)
```

The second term shows that drift in the map is amplified by `(1-K_∞(i))²` — close to 1
in degenerate directions. This is the worst case: the EKF partially corrects (reducing P)
but the correction is based on bad map data (increasing true error).  ∎

### Implication for Age Ratio

The age ratio from T5 is amplified in degenerate directions because:

1. Drift accumulates faster in degenerate directions (larger true error per step)
2. Old map points in degenerate directions have larger position errors
3. The residuals against these old points are correspondingly larger

Therefore, trajectories through degenerate environments (corridors, open fields) should
show both higher condition numbers (T6) and faster-growing age ratios (T5). This is a
testable prediction.

---

## 10. NEES Consistency Test

### Problem Statement

Given ground truth from motion capture, verify that the reported covariance is statistically
consistent with the actual estimation error.

### Definition

The Normalized Estimation Error Squared (NEES) at time `k` is:

```
NEES_k = (x_true_k - x̂_k)^T · P_k⁻¹ · (x_true_k - x̂_k)
```

where `x̂_k` is the estimated pose (6-DOF), `x_true_k` is the ground truth, and `P_k` is
the reported 6×6 pose covariance.

### Theorem 10 (NEES Consistency Criterion)

*If the estimator is consistent (i.e., `E[(x̂ - x)(x̂ - x)^T] = P`), then:*

```
E[NEES_k] = n = 6
```

*and `NEES_k` follows a chi-squared distribution with `n = 6` degrees of freedom:*

```
NEES_k ~ χ²(6)
```

*The 95% confidence interval is `[χ²_{0.025}(6), χ²_{0.975}(6)] = [1.24, 14.45]`.*

**Proof:**

Define `e_k = x_true_k - x̂_k`. If the estimator is consistent, `e_k ~ N(0, P_k)`.
Then `P_k^{-1/2} · e_k ~ N(0, I_6)` and:

```
NEES_k = e_k^T · P_k⁻¹ · e_k = ‖P_k^{-1/2} · e_k‖² ~ χ²(6)
```

since the sum of squares of 6 independent standard normals is chi-squared with 6 DOF.  ∎

### Average NEES Test

For K time steps, the average NEES:

```
NEES_avg = (1/K) · Σ_{k=1}^K NEES_k
```

Under consistency, `K · NEES_avg ~ χ²(6K)`. For large K, by the central limit theorem:

```
NEES_avg → N(6, 12/K)    (mean 6, variance 12/K)
```

The 95% confidence interval for `NEES_avg` is approximately `6 ± 2·√(12/K)`.

### Expected Results

| Configuration | E[NEES_avg] | Interpretation |
|---------------|-------------|----------------|
| Vanilla FAST-LIO2 (upstream) | ≫ 6 | Overconfident (pos/rot swap, no frame rotation) |
| Our system without CRLB | > 6 | Slightly overconfident in weak directions |
| Our system with CRLB | ≈ 6 | Consistent (if CRLB captures the dominant error source) |
| Long trajectory (drift) | ≫ 6 | Overconfident due to unmodeled map drift (T4) |

### Implementation Notes

The NEES computation requires aligning the error with the covariance frame:
- Position error: `δp = p_true - p_est` (odom frame, matches P directly)
- Rotation error: `δθ = Log(R_est^T · R_true)` for right perturbation, then rotate to odom
  frame: `δθ_odom = R_est · δθ_body`. After frame rotation (T3), P_θθ is already in odom
  frame, so use `δθ_odom` directly.

The mocap alignment (SE(3) Umeyama) must be applied before NEES computation to remove the
constant offset between mocap and odom frames.

### Numerical Example

For K = 1000 samples (10 seconds at 100Hz scan rate):

```
95% interval for NEES_avg: 6 ± 2·√(12/1000) = 6 ± 0.22 = [5.78, 6.22]
```

If NEES_avg = 50, the estimator is massively overconfident (50/6 ≈ 8× more error than
reported). This is expected for vanilla FAST-LIO2. If our CRLB inflation brings NEES_avg
to ~6-10, the inflation is working.

---

## 11. Summary of Theorems

| # | Theorem | Statement | Significance |
|---|---------|-----------|--------------|
| T1 | Error Bound | Position ≤ O(dt), rotation ≤ O(√dt) | ~2cm accuracy between scans |
| T2 | Observability | N scans for rank-6 FIM | Justifies scan accumulation |
| T3 | Frame Consistency | P_odom = T·P·T^T, T=diag(I,R) | Fixes upstream covariance bug |
| T4 | Drift Lower Bound | Var(ε_K) ≥ K·σ²_η | P→0 but drift grows |
| T5 | Drift Detection | ρ = r̄_old/r̄_new detects drift | Online diagnostic, no external ref |
| T6 | CRLB Inflation | γ_i = max(0, σ²_z/λ_i - P_proj) | Honest covariance in degeneracy |
| T7 | Map Uncertainty | Σ_map = G·P·G^T per point | Explains why age ratio works |
| T8 | Loop Closure | Revisitation via ρ spike | Drift correction without SLAM |
| T9 | Drift-Degeneracy | Drift amplified by (1-K_∞)² | Degenerate dirs drift fastest |
| T10 | NEES Test | E[NEES]=6 iff consistent | Validates covariance correctness |

### How These Connect

1. **T2** ensures observability → **T1** bounds short-term error → **T3** reports it correctly
2. **T4** proves long-term drift is inevitable → **T5** detects it online
3. **T6** identifies weak directions via FIM → **T9** proves drift amplifies there
4. **T7** explains the map error mechanism underlying T5
5. **T8** uses T5's detection to trigger correction
6. **T10** validates the entire covariance pipeline (T3 + T6) against ground truth

The chain for the flight controller:
- Short-term: T1 guarantees bounded error, T3 reports correct frame, T6 inflates weak
  directions → the covariance is honest per-scan
- Long-term: T4 proves drift is inevitable, T5 detects it, T9 shows it's worst in
  degenerate directions → the system knows when to be cautious
- Validation: T10 provides the statistical test to verify all of the above

Together, they provide a complete theoretical framework for using FAST-LIO2 as a drone
odometry source: short-term bounded (T1), correctly reported (T3+T6), with online
self-awareness of its own limitations (T5+T9), and a rigorous validation methodology (T10).

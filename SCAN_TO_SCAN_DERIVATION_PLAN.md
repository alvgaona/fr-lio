# Scan-to-Scan CRLB Drift Covariance Derivation Plan

Plan for the formal mathematical treatment of the scan-to-scan CRLB drift covariance
estimation as implemented in this project. This complements the FEJ-IEKF derivation
by addressing the absolute scaling of the published covariance.

## Motivation

The IEKF in scan-to-map LIO produces covariance that converges and stays bounded
regardless of distance traveled. This is because the map anchors the state — every
correction reduces uncertainty back to the same level. As a result, the published
covariance does not reflect the actual drift accumulated by the system.

The scan-to-scan CRLB approach measures, at each time step, how much new uncertainty
the current scan would have if registered against the previous scan alone (without
map anchoring). This per-step uncertainty is added to the published covariance,
producing a monotonically growing, environment-adaptive drift estimate without
parameters to tune.

## Goals

1. Define the per-step drift covariance as the Cramer-Rao Lower Bound of scan-to-scan
   point-to-plane registration
2. Show that this bound is achievable, environment-adaptive, and tuning-free
3. Prove that adding it to the IEKF covariance produces a monotonically growing,
   physically meaningful uncertainty
4. Validate that the published covariance correlates with actual drift in degenerate
   environments

## Step 1: Scan-to-Scan Registration Problem

### Goal
Define the relative pose estimation problem between two consecutive LiDAR scans.

### Setup
Let scan_{k-1} and scan_k be two consecutive LiDAR scans in body frame. The relative
pose between them is:

T_rel = (R_rel, t_rel)

Where:
- R_rel = R_{k-1}^T R_k (relative rotation)
- t_rel = R_{k-1}^T (t_k - t_{k-1}) (relative translation)

Both R_rel and t_rel are derived from the IEKF state estimates at times k-1 and k.

### Point-to-plane registration
For each point p_cur in scan_k, transform it to the previous scan frame:

p_in_prev = R_rel * p_cur + t_rel

Find its nearest neighbors in scan_{k-1} and fit a plane (n, d) such that:

n^T q + d = 0 for all q in the local plane

The point-to-plane residual is:

r_i = n_i^T (R_rel * p_i + t_rel) + d_i

This is the same point-to-plane formulation as the EKF measurement model, but applied
between two scans rather than between a scan and the map.

## Step 2: Linearization and Jacobian

### Goal
Derive the Jacobian of the residual with respect to the relative pose error.

### Error parameterization
Define the relative pose error as:

- delta_t (R^3): translation error
- delta_theta (R^3): rotation error such that R_rel_true = R_rel * Exp(delta_theta)

### Jacobian derivation
The residual is r_i = n_i^T (R_rel * p_i + t_rel) + d_i.

Linearizing with respect to delta_t and delta_theta:

dr_i/d(delta_t) = n_i^T  (1x3)
dr_i/d(delta_theta) = -n_i^T R_rel [p_i]_x  (1x3)

Stack into a 1x6 row:

J_i = [n_i^T | -n_i^T R_rel [p_i]_x]

For N valid correspondences, stack into an N x 6 Jacobian matrix J.

## Step 3: Fisher Information Matrix and CRLB

### Goal
Compute the FIM and derive the CRLB on the relative pose covariance.

### FIM
Assuming the residuals are independent and zero-mean Gaussian with variance R_s2s:

FIM = (1/R_s2s) J^T J

### CRLB
The Cramer-Rao Lower Bound on any unbiased estimator of the relative pose is:

P_rel >= FIM^{-1} = R_s2s (J^T J)^{-1}

This is the minimum achievable covariance for any estimator using these measurements.

### Empirical noise estimation
Rather than assuming a fixed measurement variance, use the post-fit residual sum of
squares to empirically estimate R_s2s:

R_s2s = (sum of r_i^2) / (N - 6)

The (N - 6) accounts for the 6 degrees of freedom of the relative pose (degrees of
freedom in the residuals after fitting). This makes the noise estimate environment-
adaptive without manual tuning.

## Step 4: Eigenvalue Regularization

### Goal
Handle degenerate directions where J^T J is singular or nearly so.

### Problem
In geometrically degenerate environments (long corridors, flat walls), J^T J has
small eigenvalues in the unobservable directions. Direct inversion produces enormous
covariance values that explode the published uncertainty.

### Solution
Eigendecompose J^T J = V Lambda V^T. For each eigenvalue lambda_i:

- If lambda_i > epsilon: invert as 1/lambda_i
- If lambda_i <= epsilon: set inv_lambda_i = 0

This caps the covariance contribution from poorly observed directions. The
implementation uses epsilon = 1e-6.

P_rel = R_s2s * V * diag(inv_lambda) * V^T

The result: degenerate directions contribute zero uncertainty (rather than infinite),
while well-observed directions contribute the proper CRLB-bounded uncertainty.

### Trade-off discussion
This is conservative: it reports zero drift in degenerate directions, even though
drift may actually accumulate there. An alternative would be to report a very large
(but finite) covariance, but this would dominate the published uncertainty and make
it unusable. The conservative approach is acceptable because the FEJ-IEKF separately
prevents the filter from being overconfident in unobservable directions.

## Step 5: Accumulation Over Time

### Goal
Show how per-step CRLB drift covariance accumulates over a trajectory.

### Accumulation rule
At each time step, add the per-step relative pose CRLB to the running drift covariance:

P_drift_{k+1} = P_drift_k + Phi_k * P_rel_k * Phi_k^T

Where Phi_k accounts for the propagation of the previous drift estimate through the
relative pose. For small steps, Phi_k can be approximated as identity.

### Properties
- Monotonically growing: each step only adds, never subtracts
- Environment-adaptive: P_rel is small in feature-rich rooms, large in corridors
- Tuning-free: only depends on residual statistics and FIM eigenvalues
- Zero on stationary segments: relative motion is zero, residuals are zero

### Published covariance
The final published covariance is the sum of the IEKF filter covariance and the
accumulated drift covariance:

P_published = P_filter + P_drift

The filter covariance reflects local uncertainty (with FEJ for consistency), while
the drift covariance reflects accumulated error since the start of the trajectory.

## Step 6: Properties and Guarantees

### Goal
Formally state the properties of the scan-to-scan CRLB drift covariance.

### Property 1: Lower bound
The accumulated drift covariance is a lower bound on the actual error covariance,
because each P_rel_k is a CRLB (and the sum of lower bounds is a lower bound on the
sum). Real drift can exceed this estimate but cannot be smaller.

### Property 2: Achievability
The CRLB is achievable when residuals are Gaussian and the linearization is exact.
In practice, the empirical R_s2s estimate captures actual residual statistics, so
the bound is tight when registration converges well.

### Property 3: Environment adaptivity
In a feature-rich room, J^T J has high eigenvalues in all 6 DOF, so P_rel is small.
In a corridor, eigenvalues collapse in the along-corridor direction, but the
regularization keeps that contribution at zero. The drift grows mainly from the
well-observed directions, which is the correct behavior.

### Property 4: No parameters to tune
The only constants are the eigenvalue threshold (1e-6, well below any meaningful
value) and the maximum number of points used for the FIM computation (300, for
computational efficiency). Neither affects the qualitative behavior.

## Step 7: Connection with FEJ-IEKF

### Goal
Show how the two contributions complement each other.

### Roles
- FEJ-IEKF: prevents the filter covariance P_filter from artificially shrinking in
  unobservable directions due to linearization artifacts. Local consistency.
- Scan-to-scan CRLB: provides honest absolute uncertainty by accumulating per-step
  drift bounds. Global consistency.

### Combined published covariance
P_published = P_filter (FEJ-IEKF) + P_drift (CRLB accumulation)

Both contributions are physically meaningful:
- P_filter encodes "how much do I trust this local estimate given recent measurements"
- P_drift encodes "how much could I have drifted since the start"

Together they produce a covariance that is consistent locally (NEES within bounds)
and grows honestly with distance traveled.

### Validation strategy
- Mocap arena: NEES analysis with mocap ground truth, should be within chi-squared
  bounds for both linear and angular components
- University loops: drift covariance at the end should approximate actual loop
  closure error
- Degenerate environments: drift should grow faster in corridors than in rooms

## Implementation Reference

### Key files (feat/scan-to-scan branch)
- src/laserMapping.cpp: compute_scan_to_scan_covariance() function
- Configuration parameter: mapping.use_scan_to_scan_cov (boolean toggle)

### Key code structure
1. Maintain previous scan and its k-d tree (kdtree_prev_scan)
2. After each EKF update, compute relative pose to previous scan
3. For up to 300 points in current scan, find nearest neighbors in previous scan
4. Fit a plane and compute point-to-plane residual + Jacobian row
5. Stack J, compute J^T J (FIM)
6. Eigendecompose, regularize, invert
7. Empirically estimate R_s2s from residuals
8. Compute P_rel = R_s2s * V * diag(inv_lambda) * V^T
9. Accumulate into P_drift, add to P_filter for publishing
10. Update previous scan to current scan for next iteration

### Computational cost
- Negligible: 300 nearest-neighbor queries + a 6x6 eigendecomposition per scan
- Runs at LiDAR rate (10 Hz), not IMU rate
- Memory: one extra k-d tree for the previous scan

## References

- Cramer, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
- Rao, C. R. (1945). Information and the accuracy attainable in the estimation of
  statistical parameters. Bulletin of the Calcutta Mathematical Society.
- Censi, A. (2007). An accurate closed-form estimate of ICP's covariance. ICRA 2007.
- Brossard, M., Bonnabel, S., Barrau, A. (2020). A new approach to 3D ICP covariance
  estimation. IEEE Robotics and Automation Letters.

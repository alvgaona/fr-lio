# Future Improvements

## Covariance Drift Calibration

The forward-propagated covariance (`J * P * J^T + N`) is physically correct but reports near-zero values
after EKF convergence. This is accurate given the model — the EKF is genuinely confident after LiDAR
correction — but insufficient for downstream systems that rely on covariance, such as pose graph
optimizers (RTAB-Map, Cartographer) where near-zero covariance produces near-infinite information
weights, preventing loop closures from correcting accumulated drift.

### Linear Drift Noise Model

A linear process noise term added to the propagated covariance models unmodeled drift sources
(vibration, non-rigid mounting, map drift, environmental changes):

```
P_pos(i,i) += sigma2_pos * dt
P_rot(i,i) += sigma2_rot * dt
```

Where `sigma2_pos` (m^2/s) and `sigma2_rot` (rad^2/s) represent the rate at which odometry
uncertainty grows over time. This term is independent of the EKF — it only affects the published
covariance, not the SLAM estimation.

### Calibration Procedure

The drift noise parameters must be calibrated empirically from real trajectory data.

#### Method 1: Motion Capture Ground Truth (recommended for publication)

1. Fly trajectories with a motion capture system (Vicon, OptiTrack) recording ground truth
2. Compute position and rotation error between FAST-LIO and ground truth over sliding windows
   of duration `dt`
3. Plot `Var(error)` vs `dt` — the slope of the linear fit is the drift rate `sigma2`
4. Fit position and rotation drift rates separately

#### Method 2: Closed-Loop Trajectories (practical field calibration)

1. Fly loops that return to the start point
2. Measure the position and rotation gap between start and end poses
3. Repeat for loops of different durations T_1, T_2, ..., T_N
4. Fit the drift rate: `sigma2 = (1/N) * sum(||error_i||^2 / T_i)`

#### Method 3: Known Landmarks

1. Place targets at surveyed positions in the environment
2. Compare FAST-LIO pose estimates at those positions against the survey
3. Apply the same linear fitting procedure as Method 2

### Expected Output

Calibration should yield values such as:

```
sigma2_pos = 2.3e-4 m^2/s     (position drift rate)
sigma2_rot = 1.1e-4 rad^2/s   (rotation drift rate)
```

For the paper, report the RPE-vs-time plot demonstrating the linear relationship, which validates
the random walk drift model assumption.

### Implementation

The infrastructure for this is already in place in `laserMapping.cpp`. Adding the calibrated
drift noise requires only adding two constants to the forward propagation noise section:

```cpp
constexpr double prop_noise_pos = <calibrated_value>;
constexpr double prop_noise_rot = <calibrated_value>;

P_pose(i,i) += acc_noise + prop_noise_pos * dt;  // position diagonals
P_pose(i,i) += gyr_noise + prop_noise_rot * dt;  // rotation diagonals
```

## Adaptive Measurement Noise

Currently `LASER_POINT_COV` is a fixed constant — every point-to-plane residual is weighted equally
regardless of scan quality. In practice, scan quality varies significantly depending on the
environment:

- Feature-rich rooms produce small, consistent residuals (high confidence)
- Open areas with few surfaces produce large, scattered residuals (low confidence)
- Transitional zones (e.g., passing through a doorway) produce mixed residuals

### Approach

After computing point-to-plane residuals for a scan, analyze their distribution to scale the
measurement covariance:

1. Compute the mean and variance of the residual magnitudes for the current scan
2. Compare against a running baseline (e.g., exponential moving average of past residual stats)
3. Scale `LASER_POINT_COV` proportionally: `R_adaptive = R_base * (sigma_residual / sigma_baseline)`

When residuals are large or spread out, the inflated covariance tells the EKF to trust the
correction less. When residuals are tight, the EKF applies corrections more aggressively.

### Benefits

- Automatic robustness in geometrically degenerate environments without manual tuning
- Prevents EKF from applying confident but wrong corrections in poor scan conditions
- Complementary to degeneracy detection — adaptive noise handles gradual quality changes while
  degeneracy detection handles sudden structural loss of observability

### Implementation

The residuals are already computed in `h_share_model()`. The adaptive scaling can be applied
after the residual loop, before passing `ekfom_data` to the EKF update. No changes to the
filter structure are needed — only the measurement noise matrix R changes per scan.

## Adaptive Scan Accumulation

Replace the fixed N=10 scan accumulation with adaptive logic based on:

- Point density per scan (non-repetitive LiDAR produces uneven distributions)
- Geometric spread (ensure sufficient spatial coverage for registration)
- Motion intensity (accumulate fewer scans during fast motion to reduce distortion)

This would allow the system to automatically adjust the LiDAR update rate based on conditions,
improving robustness in both static and dynamic scenarios.

## Observability-Constrained IEKF (OC-IEKF)

FAST-LIO's IEKF suffers from a fundamental linearization artifact: the covariance along
unobservable directions (primarily global yaw) artificially shrinks over time. The EKF
becomes overconfident about states it cannot actually observe, because the Jacobians evaluated
at the estimated state (which contains errors) shift the null space of the observability
matrix away from its true structure.

The IEKF's iterations reduce this effect (by bringing the linearization point closer to the
true state) but do not eliminate it. Over long trajectories, yaw covariance collapses below
physically justified values, which:

- Prevents downstream systems (loop closure, GPS fusion) from correcting accumulated drift
- Produces inconsistent uncertainty estimates (NEES > 1)
- Gives false confidence in the state estimate

### Research Plan

#### 1. Observability Analysis

Derive the unobservable subspace for FAST-LIO's state representation:
- State: SO(3) rotation, position, velocity, gyro bias, accel bias, gravity (S2)
- Measurements: point-to-plane residuals from LiDAR scan matching
- Expected null space: global yaw (4D including position/yaw coupling)

Construct the observability matrix `O = [H; H*F; H*F^2; ...]` for the continuous-time system
and verify the null space dimension and structure.

#### 2. Demonstrate the Problem

Using mocap ground truth from the drone arena:
- Run standard FAST-LIO IEKF and log the full covariance matrix over time
- Plot yaw covariance vs time — show it shrinks monotonically
- Compute NEES (Normalized Estimation Error Squared) against ground truth — show it exceeds
  the chi-squared bound, proving inconsistency

#### 3. Implement Two Constrained Variants

**FEJ-IEKF (First-Estimates Jacobian):**
- Store the state estimate at the beginning of each EKF update cycle
- Evaluate Jacobians F and H at this first estimate for covariance propagation
- Let the IEKF iterations refine the state using unconstrained Jacobians
- Simple to implement: modify `update_iterated_dyn_share_modified()` to separate state
  and covariance Jacobians

**OC-IEKF (Observability-Constrained):**
- After computing Jacobians F and H, project out components that would inject false
  information into the unobservable subspace
- Enforce that the null space of the linearized observability matrix matches the true
  nonlinear null space at every update step
- More rigorous than FEJ but requires explicit null space computation

#### 4. Experimental Validation

Compare three variants (standard IEKF, FEJ-IEKF, OC-IEKF) on real drone flights:

**Consistency metrics:**
- NEES against mocap ground truth (should be within chi-squared bounds)
- Yaw covariance evolution over time (should not shrink for constrained variants)
- Covariance envelope plots (3-sigma bounds vs actual error)

**Accuracy metrics:**
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error) at multiple time horizons

**Fusion compatibility:**
- Feed each variant's output into a pose graph optimizer with artificial loop closures
- Show that constrained variants accept loop closure corrections while standard resists them

#### 5. Paper Structure

*"Observability-Constrained Iterated Kalman Filtering for LiDAR-Inertial Odometry on
Aerial Platforms"*

1. Introduction — motivation for consistent covariance in aerial LIO
2. Preliminaries — FAST-LIO IEKF formulation, observability theory
3. Observability analysis — null space derivation for LiDAR-inertial IEKF
4. Proposed methods — FEJ-IEKF and OC-IEKF formulations
5. Experimental results — consistency and accuracy on real drone flights
6. Discussion — FEJ vs OC trade-offs, computational cost, practical recommendations

### Key References

- Huang, Mourikis, Roumeliotis (2010) — OC-EKF for visual-inertial SLAM
- Huang, Mourikis, Roumeliotis (2009) — FEJ for EKF-SLAM consistency
- Xu, Cai, et al. (2022) — FAST-LIO2 iterated EKF formulation
- Hesch et al. (2014) — Observability-constrained VINS

### Implementation Notes

The core modification is in `src/FAST_LIO/include/ikd-Tree/IKFoM_toolkit/esekfom/esekfom.hpp`,
specifically `update_iterated_dyn_share_modified()`. For FEJ, store the initial state before
iteration and evaluate `h_x` (measurement Jacobian) at that state for the covariance update.
For OC, additionally compute the null space projection and apply it to both F and H before the
covariance update.

## Error Bound Analysis

Derive theoretical error bounds for the forward-propagated odometry as a function of:

- IMU noise parameters (accelerometer and gyroscope spectral densities)
- Propagation interval (time since last LiDAR correction)
- Number of accumulated scans
- Point cloud density and geometric distribution

This analysis would provide formal guarantees on the odometry quality between LiDAR corrections,
which is relevant for safety-critical applications like autonomous flight.

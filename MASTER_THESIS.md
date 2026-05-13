# Master's Thesis: FR-LIO

## Title

*"FR-LIO: Flight-Ready LiDAR-Inertial Odometry for Multirotor Platforms"*

## Core Claim

No existing LiDAR-inertial odometry system provides IMU-rate (200Hz) odometry output with
indoor-grade accuracy on a flying multirotor platform. Existing systems either:

- Publish odometry at LiDAR rate only (10-20Hz), forcing the flight controller to dead-reckon
  between updates (at 2 m/s, 10Hz means 0.2m between corrections vs 0.01m at 200Hz)
- Are designed for repetitive-scan LiDARs (Velodyne, Ouster) and fail or degrade with
  non-repetitive scanning patterns (Livox Mid-360)
- Are validated on handheld or ground platforms, not under flight vibration and real-time
  constraints of an onboard flight controller

FR-LIO fills this gap: accurate IMU-rate odometry, designed for non-repetitive LiDAR, validated
on a real multirotor platform.

## Related Work Gap Analysis

A literature survey table comparing aerial LIO systems should include:

| System | Odom Rate | LiDAR Type | Platform | Non-Repetitive Support |
|--------|-----------|------------|----------|----------------------|
| FAST-LIO2 | 10Hz (LiDAR rate) | Any | Handheld/ground | Yes |
| LIO-SAM | 10Hz (LiDAR rate) | Spinning only | Ground/handheld | No |
| Point-LIO | IMU-rate | Any | Handheld/ground | Yes |
| Faster-LIO | 10Hz (LiDAR rate) | Any | Handheld/ground | Yes |
| LINS | 10Hz (LiDAR rate) | Spinning only | Ground | No |
| LIO-EKF | TBD | TBD | TBD | TBD |
| **FR-LIO** | **200Hz (IMU rate)** | **Non-repetitive** | **Multirotor** | **Yes** |

Key gaps to highlight:
- No system validated on a flying platform with IMU-rate output
- LIO-SAM and LINS assume repetitive scanning (feature extraction relies on ring structure)
- Point-LIO claims IMU-rate but has not been validated on aerial platforms under vibration
- None address flight-specific robustness (stale anchor, velocity filtering for FCU, memory
  bounding for long flights)

**Point-LIO differentiation** (critical — it's the closest competitor):
- Point-LIO processes every point individually (per-point EKF update), FR-LIO keeps batch
  scan-matching and decouples odom publishing to IMU rate. Simpler architecture.
- Per-point processing has much higher computational cost — may not sustain real-time on
  embedded platforms like Orin NX. Must validate in Experiment 10.
- Point-LIO redesigns the entire filter; FR-LIO builds on proven FAST-LIO2 with minimal
  modifications, making it easier to maintain and tune.
- Point-LIO lacks flight-specific features: no stale anchor handling, no velocity filtering
  for FCU, no memory bounding, no degeneracy/drift detection.
- With non-repetitive LiDAR (Mid-360), FR-LIO's scan accumulation provides control over
  spatial coverage per update. Point-LIO has no equivalent lever.

**TODO**: Test LIO-EKF and fill in its row. Evaluate its compatibility with Mid-360.

## Thesis Structure

### Chapter 1 — Introduction
- Motivation: onboard LiDAR odometry for GPS-denied multirotor flight
- Gap: no existing system provides IMU-rate odometry with indoor accuracy on a flying platform
- Non-repetitive LiDAR challenge: most systems assume repetitive scanning patterns
- Problem formulation: state estimation problem (pose, velocity, biases) with IMU propagation
  model and point-to-plane measurement model
- Contributions summary

### Chapter 2 — Background and Related Work
- LiDAR-inertial odometry fundamentals (error-state IEKF, point-to-plane registration, ikd-Tree)
- FAST-LIO2 architecture overview
- Non-repetitive vs repetitive LiDAR scanning patterns and their impact on LIO
- ROS2 real-time considerations (executors, QoS)
- Related work survey with comparison table (see above)
- Clear statement of what gap FR-LIO fills

### Chapter 3 — System Architecture
- Hardware platform: Orin NX + Mid-360 + Pixhawk 4 Mini
- Computational constraints and sensor characteristics
- Non-repetitive scanning pattern of Mid-360 and why it matters (uneven point distribution,
  scan accumulation needed, no ring-based feature extraction)

### Chapter 4 — Real-Time Odometry at IMU Rate

#### 4.1 Problem Statement
- Odom rate degrades from 200Hz under load
- Why IMU-rate matters for flight: eliminates dead-reckoning gap between corrections,
  provides continuous feedback to FCU

#### 4.2 Method
- Executor isolation: separate IMU thread, never blocked by scan processing
- Mutex contention: lidar preprocessing outside lock
- Path serialization bottleneck: throttled to 10Hz

#### 4.3 Experiments

**Experiment 1 — Odom Rate Stability (Ablation Study)**

Goal: Prove constant 200Hz under all conditions and isolate each fix's contribution.

Environment: Mocap arena (9×9m).

Repetitions: 5 flights per variant.

Setup: Fly the same trajectory with each variant:
- (a) Upstream FAST-LIO2 (single executor, no modifications)
- (b) Separate executors only
- (c) Separate executors + mutex fix
- (d) Full FR-LIO (+ path throttle)

Metrics:
- Odom rate over time for each variant
- Rate histogram (mean, std, min)
- Show (a) degrades, (d) stays constant

**Experiment 2 — Accuracy vs Ground Truth**

Goal: Show IMU-rate odometry doesn't degrade accuracy compared to LiDAR-rate.

Environment: Mocap arena (9×9m) with OptiTrack/Vicon ground truth.

Repetitions: 3-5 flights per trajectory type.

Setup: Fly structured trajectories:
- Slow hover (0.5 m/s)
- Moderate flight (2 m/s)
- Aggressive maneuvers (fast yaw, quick stops)
- Altitude variation (floor to ceiling sweeps)
- Yaw rotations (spin in place)

Metrics:
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error) at 1s, 5s, 10s horizons
- Compare FR-LIO vs upstream FAST-LIO2 vs Point-LIO vs LIO-EKF
- Report with confidence intervals

### Chapter 5 — Robustness for Flight

#### 5.1 Stale Anchor Handling
- Problem: odom silently stops when EKF anchor becomes stale (dt > 0.5s)
- Solution: warn-and-continue instead of hard return

#### 5.2 Velocity and Orientation Filtering
- Low-pass filter for flight controller compatibility
- Configurable alpha parameter for noise vs lag trade-off

#### 5.3 Memory Bounding
- ikd-Tree cube sizing for long-duration flights
- QoS and queue tuning: SensorDataQoS for odom, reduced IMU queue

#### 5.4 Experiments

**Experiment 3 — Velocity Filter Evaluation**

Goal: Quantify filter quality vs lag trade-off.

Environment: Mocap arena (9×9m) with mocap-derived ground truth velocity.

Repetitions: 3 flights per alpha value.

Setup: Fly trajectories with known velocity profiles (hover → accelerate → constant velocity
→ stop). Vary `vel_filter_alpha` (0.02, 0.05, 0.1, 0.2, 1.0).

Metrics:
- Velocity RMSE against mocap-derived velocity
- Velocity delay (cross-correlation lag)
- Plot filtered vs ground truth velocity for each alpha
- Identify the sweet spot where noise is removed without excessive lag

**Experiment 4 — Memory and Computational Performance**

Goal: Show FR-LIO is viable for long flights.

Environment: University building (long trajectory rosbag) + mocap arena (long duration hover).

Setup: Replay long rosbags (10-20 minutes). Compare `cube_side_length` values
(50, 100, 200, 1000).

Metrics:
- RSS memory over time
- Per-scan processing time over time
- Odom rate over time
- Show 1000m cube causes memory growth and rate degradation
- Show bounded cube keeps both constant

**Experiment 5 — Stale Anchor Robustness**

Goal: Show odom survives lidar interruptions.

Environment: Mocap arena (9×9m).

Repetitions: 5 trials per condition.

Setup: Intentionally degrade lidar during flight:
- Block the sensor briefly (hand over lens for 1-2s)
- Or replay a bag and drop lidar messages artificially (skip 0.5s of scans every N seconds)

Metrics:
- Compare original behavior (hard return at dt > 0.5s) vs FR-LIO (warn and continue)
- Plot odom output — show original goes silent, FR-LIO continues
- Measure position drift during the gap vs mocap ground truth
- Show recovery accuracy after lidar resumes

### Chapter 6 — Covariance Consistency

#### 6.1 FEJ-IESKF (Filter-Level Consistency)
- Problem: standard IEKF re-linearizes Jacobians at each iteration and time step, injecting spurious
  observability into unobservable directions (global yaw + position). Covariance collapses to near-zero
  in those directions → overconfident filter, brittle to disturbances.
- Solution: First-Estimates Jacobian — freeze measurement Jacobian at the propagated state for Kalman
  gain and covariance update. IEKF iterations still re-linearize residuals for convergence.
- Two-level re-linearization unique to IESKF: (1) across time steps, (2) within one update's iterations.
  FEJ fixes both simultaneously.
- Point-set locking: freeze valid correspondences from iteration 0 to keep Jacobian dimensions stable.
- Related work: FEJ for VIO (Huang et al. 2008, Li & Mourikis 2013, FEJ2 2022). InEKF for LIO
  (Inv-LIO, Eq-LIO) requires full Lie group reformulation. No prior FEJ for IEKF-based LIO.
- Implementation: ~200 lines added to FAST-LIO's esekfom.hpp (feat/fej-iekf branch, ea8fdf4).

#### 6.2 Uncertainty-Aware Map (Map-Level Consistency)
- Problem: even with FEJ, covariance converges because scan-to-map matching treats the ikd-tree as
  ground truth. Drift is absorbed into map distortion, invisible to the filter. The covariance doesn't
  grow over time, making it useless for downstream SLAM systems.
- Insight (from VoxelMap, Yuan et al. 2022): map points have uncertainty inherited from the pose
  covariance at registration time. This should propagate into the measurement noise.
- Solution for ikd-tree: store diagonal of P_point in normal_x/y/z fields (curvature NOT available —
  used for timestamps). At measurement time project onto plane normal:
  - `P_point = J * P_pose * J^T` (Jacobian of `m = R * p_L + t` w.r.t. pose)
  - `sigma_reg = n^T * diag(P_point) * n` (diagonal approximation projected onto plane normal)
  - `R_i = LASER_POINT_COV + sigma_reg_i` (per-point measurement noise)
- Raises steady-state covariance (~3.5x in 1D analysis) and tightens on revisit (natural loop closure).
- **Limitation**: covariance still converges to a bounded steady state P_ss = sqrt(Q * r0). Does NOT
  grow with distance. After 10m or 1000m in a corridor, same P. The map always provides finite
  information, preventing unbounded growth.
- Prior art: VoxelMap stores full 6x6 plane covariance per voxel. Our approach is a lightweight diagonal
  per point in the ikd-tree — different architecture, same principle.

#### 6.2b Scan-to-Scan Drift Covariance (Global Consistency) — NOVEL CONTRIBUTION
- Problem: neither FEJ nor uncertainty-aware map produce covariance that grows with distance.
  Scan-to-map covariance converges because the map provides bounded information every scan.
  Proved in 1D: P_ss = sqrt(Q * r0), independent of distance. This is a fundamental limitation of
  ALL scan-to-map systems (FAST-LIO, VoxelMap, etc.) — unsolved in the literature.
- Considered and rejected approaches:
  - Fixed drift floor (alpha_min * distance): grows, but requires per-environment tuning
  - Adaptive Q from residuals: fails in feature-rich corridors (local consistency ≠ global consistency)
  - Uncertainty-aware map (VoxelMap-style per-point R): raises steady state, still converges
  - Scan-to-map FIM: anchored to drifted map, same problem as residuals
- Key insight: scan-to-scan registration sees only two consecutive point clouds — no map anchoring.
  The scan-to-scan Hessian honestly reflects geometric observability of relative motion.
- Solution: architectural separation of estimation and uncertainty quantification:
  - **Estimation**: scan-to-map (FAST-LIO IESKF, untouched, accurate)
  - **Uncertainty**: scan-to-scan CRLB accumulated over time (honest, growing)
  - Published covariance: P_published = P_filter + P_drift
- No published system does this. No tuning parameters. Environment-adaptive.

**Mathematical Framework**:

**Definition 1 — Scan-to-scan relative pose model**:
Given two consecutive downsampled scans S_{k-1} and S_k, the relative pose T_rel = (R_rel, t_rel)
relates body-frame points: p_{k-1} = R_rel * p_k + t_rel. The point-to-plane measurement model is:
  h_i = n_i^T * (R_rel * p_i + t_rel - q_i)
where n_i is the plane normal fitted to 5 nearest neighbors in S_{k-1}, p_i ∈ S_k, q_i ∈ S_{k-1}.

**Definition 2 — Scan-to-scan Jacobian**:
Perturbing T_rel by (δt, δθ) via R_rel → R_rel * Exp(δθ), t_rel → t_rel + δt:
  J_i = [∂h_i/∂δt, ∂h_i/∂δθ] = [n_i^T, -n_i^T * R_rel * [p_i]_x]   (1×6)
where [p_i]_x is the skew-symmetric matrix of p_i.

**Theorem 1 — Scan-to-scan CRLB**:
Under Gaussian measurement noise h_i ~ N(0, σ²), the Cramér-Rao Lower Bound on relative pose is:
  P_rel = σ² * (J^T J)^{-1} = σ² * FIM^{-1}
where FIM = Σ_i J_i^T J_i is the 6×6 Fisher Information Matrix and σ² is the measurement variance.
P_rel is the minimum achievable covariance for any unbiased estimator of T_rel from {h_i}.
(Follows from Censi 2007, extended to 3D point-to-plane by Prakhya 2015.)

**Proposition 1 — Empirical noise estimation**:
The measurement variance σ² is estimated from post-fit residuals:
  σ̂² = Σ_i r_i² / (N - 6)
where r_i are the point-to-plane residuals at the IMU-predicted relative pose and N is the number
of valid correspondences. This is the minimum variance unbiased estimator (standard least squares).
No tuning parameter — the residuals capture all error sources (sensor noise, surface roughness,
plane fit error from limited neighbors).

**Theorem 2 — Accumulated drift covariance**:
The end-to-end pose T_{0:K} = T_{0:1} ∘ T_{1:2} ∘ ... ∘ T_{K-1:K}. Under first-order covariance
propagation (valid for small inter-scan motion):
  P_drift(K) = Σ_{k=1}^{K} Ad_{T_{k:K}} * P_rel(k) * Ad_{T_{k:K}}^T
For small rotations and as a conservative lower bound:
  P_drift(K) ≈ Σ_{k=1}^{K} P_rel(k)
This grows monotonically and unboundedly with K (number of scans), unlike scan-to-map covariance.

**Theorem 3 — Scan-to-map covariance convergence (proof of limitation)**:
In a scan-to-map filter with process noise Q and per-scan measurement with effective noise R_eff:
  P_{k+1} = (P_k + Q) * R_eff / (P_k + Q + R_eff)
This converges to P_ss satisfying P_ss² + Q*P_ss - Q*R_eff = 0, giving:
  P_ss = (-Q + sqrt(Q² + 4*Q*R_eff)) / 2
which is independent of distance traveled. Even with per-point R inflation (VoxelMap), R_eff is
finite → P_ss is finite → covariance converges. Only R_eff → ∞ permits unbounded growth, which
requires removing all map constraints.

**Corollary — Separation principle**:
The scan-to-map estimate x̂ is optimal given the map. The scan-to-scan P_drift is a conservative
envelope accounting for map drift invisible to the filter. Adding P_drift to the published
covariance does not modify the filter state or its optimality — it only provides honest uncertainty
to downstream systems.

**Property — Environment adaptivity**:
FIM eigenvalues depend on local geometry:
- In corridors: along-corridor eigenvalue small → P_rel large in that direction → fast drift growth
- In cluttered rooms: all eigenvalues large → P_rel small → slow drift growth
- During fast motion: less scan overlap → fewer correspondences → weaker FIM → larger P_rel
- During rotation: new surface orientations → stronger rotation constraints → smaller P_rel_rot
No tuning parameter — the Hessian of the geometry determines everything.

**Novelty claim**: No published LIO system separates estimation from uncertainty quantification
using different registration strategies. All existing systems (FAST-LIO, VoxelMap, LIO-SAM,
Inv-LIO, Eq-LIO) use the same registration for both. This is the first principled, tuning-free,
environment-adaptive drift covariance for scan-to-map LiDAR-inertial odometry.

**Prior art to cite and differentiate**:
- Censi 2007 / Prakhya 2015: scan matching CRLB derivation (we build on this)
- Brossard et al. 2020: ICP covariance with initialization uncertainty (per-registration, not accumulated)
- LIO-SAM: Hessian covariance in factor graph (from scan-to-submap, not scan-to-scan; same registration for both)
- DCE (Sensors 2019): online drift covariance (requires external GNSS, not self-contained)
- VoxelMap: per-voxel R inflation (converges, proven in Theorem 3)

#### 6.3 Degeneracy Detection
- FIM eigendecomposition from measurement Jacobian (H^T H)
- 6 eigenvalues map to position and rotation DOFs
- Small eigenvalue = poorly constrained direction
- Condition number as a single scalar health indicator

#### 6.5 Drift Detection
- Old vs new map point residual comparison using point age
- Rising old-point residuals indicate map staleness
- Ratio of old/new mean residuals as drift indicator

#### 6.6 CRLB-Based Covariance Inflation
- Cramer-Rao Lower Bound from FIM eigenvalues
- Inflate published covariance along degenerate directions when EKF is overconfident
- Downstream systems (flight controller, pose graphs) get honest uncertainty

#### 6.7 Experiments

**Experiment 6 — FEJ-IESKF Covariance Consistency**

Goal: Show FEJ-IESKF produces more consistent (honest) covariance than standard IESKF.

Environment: Mocap arena (9×9m) with ground truth + long corridor (no ground truth).

Repetitions: 3-5 flights in arena.

Setup: Run both standard IESKF and FEJ-IESKF on the same data.

Metrics:
- NEES (Normalized Estimation Error Squared) against mocap ground truth — should be closer to
  chi-squared bounds with FEJ
- Covariance diagonal plots over time: FEJ should maintain higher steady-state values (especially Z)
- ATE comparison: should be similar or better (FEJ doesn't hurt accuracy)
- Corridor: qualitative covariance comparison — FEJ should show larger uncertainty along unconstrained
  directions

**Experiment 7 — Scan-to-Scan Drift Covariance**

Goal: Validate that scan-to-scan CRLB produces environment-adaptive, growing covariance that
predicts actual drift without tuning parameters.

Environment: Dungeon (flat walls), drone arena (9×9m), university corridor, closed-loop trajectory.

Setup: Compare variants on same data:
- (a) Standard IESKF (published P = filter P, converges)
- (b) FEJ-IESKF (published P = filter P, higher steady state)
- (c) FEJ-IESKF + scan-to-scan drift covariance (published P = filter P + P_drift, grows)

Metrics:
- **Covariance growth**: plot P_drift_pos and P_drift_rot vs time/distance for all environments.
  Show monotonic growth in (c), convergence in (a) and (b).
- **Environment adaptivity**: P_drift grows faster in dungeon (flat walls) than arena (rich geometry).
  P_drift_rot is larger in dungeon (parallel walls, weak yaw constraint) than arena.
  P_drift grows faster during fast flight than slow flight.
- **Drift prediction**: at landmarks along closed-loop path, compare actual position error to
  sqrt(P_drift_pos/3). Should be within 1-3 sigma. Preliminary result: 1.5 sigma in dungeon.
- **Estimate unchanged**: ATE identical between (a), (b), (c) — scan-to-scan only modifies
  published covariance, not the filter.
- **No tuning parameters**: same code, same settings across all environments. The Hessian adapts.

Key demonstrations:
1. Theorem 3 validated: (a) and (b) covariance converges regardless of distance
2. Scan-to-scan grows: (c) P_drift increases monotonically
3. Environment-adaptive: compare drift rates across arena / corridor / dungeon
4. Empirical R_s2s vs fixed LASER_POINT_COV: show residual-based R produces better calibration

**Experiment 8 — Degeneracy Detection**

Goal: Validate that FIM eigenvalues correctly identify geometrically degenerate directions.

Environment: University building (handheld rosbag), designed to include varied geometry.

Setup: Walk through environments with known degeneracy characteristics:
- Feature-rich room (no degeneracy expected — all eigenvalues high)
- Flat wall approach (1 DOF degenerate — translation normal to wall unobservable)
- Long featureless corridor (2+ DOF degenerate — along-corridor translation and yaw)
- Open area / atrium (multiple DOF degenerate)

Metrics:
- Plot 6 FIM eigenvalues over time, annotated with environment type
- Condition number over time — should spike in degenerate segments
- Identify which eigenvector corresponds to the expected degenerate direction
- Show eigenvalues recover when entering feature-rich areas

**Experiment 9 — Drift Detection**

Goal: Validate that the old/new residual ratio detects map staleness and correlates with
actual drift.

Environment: University building (handheld closed-loop rosbag). No ground truth needed.

Setup:
- Walk a closed loop through varied geometry (corridors, rooms, stairs)
- Include deliberately degenerate segments (long corridors)
- Stationary pause mid-trajectory (map ages, residuals should grow)

Metrics:
- Plot old/new residual ratio over time, annotated with environment type
- Correlation between drift indicator spikes and degenerate eigenvalues (from Experiment 8)
- Drift indicator should spike in corridors and during stationary periods
- Drift indicator should drop in feature-rich rooms
- Relate drift indicator behavior to final loop closure error — segments with high drift
  indicator should correspond to where error accumulated

**Experiment 10 — Covariance Inflation Validation**

Goal: Show CRLB-based inflation produces more honest covariance than raw EKF output.

Environment: Mocap arena (9×9m) with ground truth.

Repetitions: 3-5 flights.

Setup: Fly trajectories that include brief degenerate conditions (e.g., facing a single
flat wall). Compare covariance with and without CRLB inflation.

Metrics:
- NEES (Normalized Estimation Error Squared) over time for both variants
- NEES should be closer to expected chi-squared bounds with inflation
- Plot 3-sigma covariance envelope vs actual error — inflated version should contain the
  actual error more consistently
- Show inflation activates only during degenerate segments (not inflating when unnecessary)

### Chapter 7 — System Evaluation

Holistic experiments that span multiple contributions and validate the full system.

**Statistical methodology**:
- Arena flights (live): 3-5 independent flights per condition for confidence intervals
- Rosbag replay (deterministic): single recording per trajectory. For accuracy metrics,
  compute RPE over sliding windows to obtain a distribution from a single run.
- University loops: 3 separate recordings of varying length. Different geometry provides
  natural variation across runs.
- Computational metrics (memory, CPU, odom rate) are algorithm properties — single run is
  sufficient, no confidence interval needed.

**Experiment 11 — Long-Distance Trajectory (reuses rosbags from Experiments 8-9)**

Goal: Validate at scale beyond the mocap arena.

Environment: University building, handheld closed-loop trajectory.

Note: Handheld because indoor drone flight outside the arena is not permitted. Acknowledged
as a limitation — the system is validated airborne in the arena and at scale handheld.

Repetitions: 3 loops of varying length (short ~100m, medium ~300m, long ~500m+).

Setup: Walk a closed loop through the university (corridors, rooms, stairs). Start and end
at the same point. Record rosbag, process offline.

Metrics:
- Loop closure error (position gap between start and end)
- Drift as percentage of total distance traveled
- Qualitative map quality (visual inspection for misalignment)
- Memory and CPU usage over the full trajectory
- Odom rate stability over the full trajectory
- Compare FR-LIO vs FAST-LIO2 vs LIO-EKF on the same rosbags

**Experiment 12 — Cross-System Comparison**

Goal: Position FR-LIO against competing systems, with emphasis on Point-LIO (closest
competitor claiming IMU-rate) and non-repetitive LiDAR compatibility.

Environment: Mocap arena (with ground truth) + university rosbag. All on Orin NX hardware.

Setup: Run on the same Mid-360 rosbags:
- FR-LIO
- FAST-LIO2 (upstream)
- LIO-SAM
- Point-LIO
- LIO-EKF

Metrics:
- Which systems can process Mid-360 data at all (LIO-SAM likely cannot)
- ATE, RPE for systems that work (arena, with ground truth)
- Odom rate stability over time (critical for Point-LIO — does it sustain IMU-rate on Orin NX?)
- CPU usage and per-update processing time (Point-LIO's per-point update vs FR-LIO's batch)
- Peak and steady-state memory usage
- Loop closure error (university trajectory)

Point-LIO specific analysis:
- Can it sustain real-time on the Orin NX with Mid-360's point rate?
- How does per-point computational cost scale compared to FR-LIO's batch approach?
- Does it maintain stable odom rate under flight-like conditions or degrade?

Note: Document which systems fail with non-repetitive LiDAR and why (ring-based feature
extraction, scan format assumptions, etc.).

### Chapter 8 — Conclusions and Future Work
- Summary of contributions
- Limitations (degeneracy in extreme cases, outdoor long-range, no visual fusion)
- Future: OC-IEKF, semantic drift correction (ship inspection), adaptive scan accumulation, camera fusion

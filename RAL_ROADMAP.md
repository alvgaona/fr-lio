# RA-L Paper Roadmap

## Positioning

**Title**: "Self-Aware Uncertainty Estimation for LiDAR-Inertial Odometry: Online Degeneracy
Detection and Drift Monitoring"

**The problem**: LIO systems report near-zero covariance after EKF convergence, but the actual
estimation error grows over time due to map drift and is anisotropic due to geometric degeneracy.
Flight controllers and downstream SLAM systems that consume this covariance make decisions based
on overconfident uncertainty estimates.

**The gap**: No existing LIO system provides:
1. Mathematically correct covariance in the ROS odometry frame (upstream FAST-LIO2 has a
   pos/rot swap bug and wrong frame convention)
2. Online detection of geometric degeneracy (which directions are poorly observed)
3. Online detection of map drift (whether the map is still consistent)
4. Covariance inflation that reflects the actual information content per direction

LION (Zhang 2016) detected degeneracy but for LiDAR-only odometry without IMU coupling.
No ikd-tree based system exposes observability metrics or drift diagnostics.

**Venue**: RA-L (IEEE Robotics and Automation Letters) — theory-oriented, reviewers value
mathematical rigor, formal proofs, and thorough experimental validation.

## Contributions

### C1: Correct Covariance Propagation with Frame Rotation [IMPLEMENTED]

Full Jacobian-based covariance propagation `P_pose = J * P * J^T + N` with:
- 6×23 Jacobian mapping full EKF state to pose error
- Cumulative noise model consistent with the EKF's per-step noise
- Rotation of the rotation covariance block from body frame (IEKF right perturbation) to
  odom frame (ROS convention): `P_odom = T * P_body * T^T`, `T = diag(I, R)`
- Fixes upstream FAST-LIO2 pos/rot swap bug

**Theorems**: T1 (error bound), T3 (frame consistency)

### C2: Online Drift Detection via Map Age Partitioning [IMPLEMENTED]

Each ikd-tree point is stamped with its insertion time. Point-to-plane residuals are partitioned
by map point age. The ratio `ρ = r̄_old / r̄_new` detects map inconsistency caused by drift.

**Preliminary result**: ρ = 1.4 on a 120s trajectory with 3.2cm mean APE.

**Theorems**: T4 (drift lower bound), T5 (age ratio as drift indicator), T7 (map uncertainty)

### C3: Degeneracy Detection via FIM Eigenanalysis [IMPLEMENTED]

The Fisher Information Matrix `FIM = H_pose^T * H_pose` (6×6) is eigendecomposed at each scan.
Eigenvalues reveal per-direction observability; the condition number detects degeneracy.

**Preliminary result**: κ ≈ 115 in a mocap room, eigenvalue range 516-59079.

**Theorems**: T2 (observability for non-repetitive LiDAR), T6 (CRLB inflation), T9
(degeneracy-drift coupling)

### C4: CRLB-Based Covariance Inflation [IMPLEMENTED]

The Cramér-Rao Lower Bound `σ²_z / λ_i` sets the minimum achievable variance in each FIM
eigenvector direction. When the EKF covariance is smaller (overconfident), it is inflated
to the CRLB. No arbitrary tuning parameters.

**Theorems**: T6 (formal proof with PSD preservation), T10 (NEES validation)

## Theoretical Framework (PAPER.md)

| # | Theorem | Statement |
|---|---------|-----------|
| T1 | Error Bound | Position ≤ O(dt), rotation ≤ O(√dt) between scans |
| T2 | Observability | N accumulated scans required for rank-6 FIM |
| T3 | Frame Consistency | P_odom = T·P·T^T, T=diag(I,R) |
| T4 | Drift Lower Bound | Var(ε_K) ≥ K·σ²_η regardless of EKF P |
| T5 | Drift Detection | ρ = r̄_old/r̄_new grows with √(1 + σ²_drift·Δt/σ²_z) |
| T6 | CRLB Inflation | γ_i = max(0, σ²_z/λ_i - P_proj), preserves PSD |
| T7 | Map Uncertainty | Σ_map = G·P·G^T per point, age as proxy |
| T8 | Loop Closure | Revisitation detection via ρ spike |
| T9 | Drift-Degeneracy | Drift amplified by (1-K_∞)² in degenerate directions |
| T10 | NEES Test | E[NEES]=6 iff covariance is consistent |

## Required Experiments

### E1: NEES Consistency Analysis (most critical for RA-L)

**Goal**: Prove that our covariance is statistically consistent.

**Method**:
```
NEES_k = (x_true - x_est)^T · P⁻¹ · (x_true - x_est)
```

**Configurations to compare**:
1. Vanilla FAST-LIO2 (upstream covariance) → expect NEES ≫ 6
2. Our system without CRLB inflation → expect NEES > 6
3. Our system with CRLB inflation → expect NEES ≈ 6

**What to show**:
- NEES time series for all three configurations
- Average NEES with χ²(6) confidence intervals
- NEES breakdown by direction (position vs rotation)

### E2: Drift Detection Validation

**Goal**: Prove that the age ratio correlates with actual drift.

**Method**:
- Multiple trajectories of different lengths (30s, 120s, 300s)
- Return-to-start loop trajectory
- Corridor vs textured room

**What to show**:
- Age ratio vs APE scatter plot across trajectories
- Age ratio time series alongside APE time series
- Age ratio spike at revisitation
- Two-sample t-test on old vs new residual distributions

**Preliminary data**: ρ = 1.4 at 120s with APE = 3.2cm. Predicted ρ ≈ 1.23 from T5 formula.

### E3: Degeneracy Detection Validation

**Goal**: Prove that FIM condition number predicts accuracy degradation.

**Method**:
- Trajectories through geometrically diverse regions
- Corridor → room → open area transitions

**What to show**:
- Condition number time series alongside APE time series
- Eigenvector visualization (arrow showing unobservable direction)
- APE growth rate in high-κ vs low-κ segments
- Correlation between κ and age ratio growth rate (T9 prediction)

### E4: Baseline Comparison

Compare covariance quality against:

| System | Covariance quality | Notes |
|--------|-------------------|-------|
| FAST-LIO2 upstream | pos/rot swapped, wrong frame | Our baseline |
| Point-LIO | Not propagated at IMU rate | Different architecture |
| LIO-EKF | Classical EKF covariance | No degeneracy awareness |

**Metrics**: NEES, covariance anisotropy ratio, APE.

### E5: Ablation Study

**Goal**: Show the contribution of each component.

| Configuration | NEES | Notes |
|---------------|------|-------|
| Upstream covariance | ≫ 6 | Baseline (broken) |
| + Frame rotation fix | > 6 | T3 only |
| + Cumulative noise | > 6 | T3 + noise model |
| + CRLB inflation | ≈ 6 | T3 + T6 (full system) |

### E6: Accuracy (shared with ICUAS)

**Setup**: Same mocap trajectories as ICUAS paper.

**Metrics**: APE, RPE at multiple time deltas.

Reference: "System integration details are presented in [ICUAS paper]."

## Paper Structure

1. **Introduction**: The overconfidence problem in LIO, why it matters for UAVs and SLAM
2. **Related Work**: FAST-LIO2, Point-LIO, LION, EKF consistency analysis literature
   (Huang & Dissanayake 2007, Bar-Shalom et al.)
3. **Preliminaries**: IEKF error state, right perturbation, EKF covariance semantics
4. **Covariance Propagation**: Jacobian derivation (T1), frame rotation (T3), CRLB (T6)
5. **Online Drift Detection**: Map age partitioning (T5), drift lower bound (T4),
   map uncertainty (T7), degeneracy-drift coupling (T9)
6. **Degeneracy Detection**: FIM eigenanalysis (T2), condition number, eigenvector
   interpretation
7. **Experiments**: E1-E6
8. **Discussion**: Limitations (drift not corrected, CRLB is single-scan), future work
   (loop closure T8, drift-aware inflation)
9. **Conclusions**

## Key References

- FAST-LIO2 (Xu et al., IEEE T-RO 2022) — base system
- Point-LIO (He et al., Adv. Intell. Syst. 2023) — per-point alternative
- LION (Zhang et al., ICRA 2016) — degeneracy detection
- Huang & Dissanayake (IJRR 2007) — EKF-SLAM consistency
- Bar-Shalom et al. (2001) — Estimation with Applications to Tracking
- Censi (ICRA 2007) — Cramér-Rao bound for point cloud registration
- RTLIO (Bai et al., Sensors 2021) — IMU forward propagation for UAVs
- LIO-EKF (Vizzo et al., ICRA 2024) — near-IMU rate classical EKF

## Implementation Status

All code contributions are implemented and committed.

| Component | Commit | Branch |
|-----------|--------|--------|
| Covariance frame rotation | 924d7ee | main |
| Drift detection | 211dd8c | main |
| Degeneracy detection | 211dd8c | main |
| CRLB covariance inflation | 48f21ee | main |
| FEJ-IEKF | ea8fdf4 | feat/fej-iekf |

### FEJ-IEKF Preliminary Results (2026-03-20)

Tested on dungeon (9x9x12m) and corridor datasets with Mid-360 on multirotor:
- **Position covariance**: Standard IEKF drives all axes to ~0 (overconfident, even during Z drift).
  FEJ-IEKF keeps covariance bounded (0.00006-0.00016) and responsive to geometry changes.
- **RPY covariance**: Similar in dungeon (rich geometry constrains rotation). Difference expected in
  degenerate scenarios (long corridors, straight-line flight).
- **Trajectory accuracy**: Identical between IEKF and FEJ-IEKF — no accuracy degradation.
- **Config**: toggled via `mapping.use_fej: true` in YAML config.

## Next Steps

### Theory (required for RA-L submission)
1. [ ] Derive observability matrix for point-to-plane LIO (no prior work exists)
2. [ ] Derive closed-form null space (expected 4D: global yaw + position-yaw coupling)
3. [ ] Prove standard IESKF re-linearization breaks null space dimension
4. [ ] Prove FEJ-IESKF preserves null space (both across-time and within-update levels)
5. [ ] Define "effective Jacobian" of IESKF and formalize the two-level re-linearization argument

### Code (can do now)
6. [ ] Build NEES computation node (uses /odom + /ground_truth/odom + covariance)
7. [ ] Add latency instrumentation for comparison
8. [ ] Add covariance diagonal logging to mat_out.txt for offline analysis

### Data Collection (requires hardware)
3. [ ] Record multiple mocap trajectories of different lengths
4. [ ] Record return-to-start loop trajectory
5. [ ] Record trajectory through diverse geometry (corridor + room)
6. [ ] Run NEES analysis: vanilla vs frame rotation vs CRLB
7. [ ] Validate drift detector: age ratio vs APE across bags
8. [ ] Validate degeneracy detector: κ vs APE in diverse environments

### Baselines
9. [ ] Run vanilla FAST-LIO2 on same bags for NEES comparison
10. [ ] Run Point-LIO and LIO-EKF for covariance comparison

### Writing
11. [ ] Write paper draft
12. [ ] Create covariance visualization figures
13. [ ] Create NEES comparison plots

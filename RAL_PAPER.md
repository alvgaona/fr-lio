# RA-L Paper: Observability-Constrained IEKF for LiDAR-Inertial Odometry

## Working Title

*"Observability-Constrained Iterated Kalman Filtering with CRLB-Based Covariance Inflation
for LiDAR-Inertial Odometry"*

## Novel Contributions

1. **FEJ-IESKF for LiDAR-inertial odometry** — first application of First-Estimates Jacobian
   to an iterated error-state Kalman filter with point-to-plane measurements. Includes the
   first formal observability analysis for point-to-plane LIO, and the first FEJ formulation
   for an iterated filter (addressing both across-time and within-update re-linearization).
2. **Scan-to-scan drift covariance** — first system to architecturally separate estimation
   (scan-to-map) from uncertainty quantification (scan-to-scan CRLB). Produces environment-
   adaptive, growing published covariance with zero tuning parameters. Proved that all scan-to-map
   covariance converges (Theorem 3), and that accumulated scan-to-scan CRLB is the principled
   solution. Uses empirical residual variance (no fixed noise parameter).
3. **Online drift detection** — old vs new map point residual comparison as a runtime health
   indicator, complementary to eigenvalue-based degeneracy detection.

## Related Work and Differentiation

### OC-EKF / FEJ in State Estimation (VIO)

- Huang, Mourikis, Roumeliotis (2008/2009) — FEJ-EKF for SLAM consistency. Proved that evaluating
  Jacobians at latest estimates causes the observable subspace to have higher dimension than the
  true nonlinear system, shrinking covariance in unobservable directions.
- Huang, Mourikis, Roumeliotis (2010) — OC-EKF for visual-inertial SLAM (MSCKF). IJRR.
- Li & Mourikis (2013) — High-precision, consistent EKF-based VIO. Applied FEJ to MSCKF. IJRR.
- Hesch et al. (2014) — Observability-constrained VINS.
- Chen, Yang, Geneva, Huang (2022) — FEJ2: compensates for linearization errors from poor first
  estimates. ICRA 2022. Implemented in OpenVINS.
- Jia et al. (2022) — FEJ-VIRO: FEJ for visual-inertial-ranging (UWB). IROS 2022.
- T-ESKF (2025) — Transformed ESKF achieving consistency without FEJ via linear time-varying
  transformation. RA-L 2025.
- **Gap**: ALL of the above are VIO/SLAM. None applied to LIO with point-to-plane IEKF.

### Invariant EKF for LIO (competing consistency approach)

- **Inv-LIO** (Xia et al., IEEE T-ASE 2023) — right-invariant EKF on SE_2(3) for tightly-coupled
  LIO. Naturally preserves observability from Lie group structure. Requires full filter redesign.
- **Eq-LIO** (Tao et al., 2024, arxiv 2409.06948) — equivariant filter for LIO with semi-direct
  product group symmetry. Theoretically proven consistency. Outperforms FAST-LIO2 in benchmarks.
- **Invariant-DLIO** (IEEE 2025) — InEKF for direct LIO, >50Hz on low-cost CPUs.
- **Differentiation**: InEKF approaches require reformulating the entire state/error model on Lie
  groups — architecturally invasive. FEJ-IEKF can be retrofitted to existing IEKF systems (like
  FAST-LIO) with minimal code changes (~200 lines). Complementary, not competing.

### Efficient LIO Systems Building on FAST-LIO's IEKF (no consistency treatment)

- **Super-LIO** (Wang et al., RA-L 2026, arxiv 2509.05723) — replaces ikd-tree with OctVox
  compact voxel map and HKNN search. ~73% faster per-frame. Uses FAST-LIO's IEKF unmodified.
  No observability or covariance consistency treatment. Co-authored by HKU MARS Lab member.
- **I2EKF-LO** (IROS 2024) — dual-iteration EKF for LiDAR odometry. No FEJ treatment.
- **LIO-EKF** (Vizzo et al., ICRA 2024) — classical EKF-based LIO. No FEJ or consistency.

These systems demonstrate that FAST-LIO's IEKF remains the dominant paradigm for real-time LIO,
and none address the observability inconsistency we fix with FEJ.

### FEJ for LiDAR-Inertial (closest work)

- **[A Consistency-Improved LiDAR-Inertial Bundle Adjustment](https://arxiv.org/abs/2602.06380)**
  (Feb 2026) — applies FEJ to LiDAR-inertial bundle adjustment with stereographic projection
  for feature parameterization. Optimization-based (MAP), not filter-based (IEKF). Different
  formulation: BA optimizes over a window of states, IEKF updates incrementally. Our work
  addresses the filter case, which is the dominant paradigm in real-time LIO (FAST-LIO2,
  Point-LIO, etc.).

### Degeneracy-Aware LiDAR-Inertial Systems

- **[DALI-SLAM](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000413)**
  (2025) — detects degeneracy via Jacobian analysis, uses remapping strategy in ESKF update.
  Suppresses degenerate directions. Does not address covariance consistency.

- **[LODESTAR](https://arxiv.org/abs/2511.09142)** (2025) — degeneracy-aware LIO with adaptive
  Schmidt-Kalman filter. Classifies states as active/fixed based on degeneracy level. More
  complex filter design. Does not apply FEJ or address observability-constrained covariance.

- **[OR-LIM](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003745)** (2024)
  — observability-aware surfel mapping with ESKF. Analyzes observability per frame based on
  surfel normal directions. Focused on mapping consistency, not filter consistency.

- **[LION](https://arxiv.org/abs/2102.03443)** (2021) — LiDAR-inertial observability-aware
  navigator for vision-denied environments. Earlier work, does not address IEKF consistency.

### LiDAR Odometry on UAVs

- **SUPER MAV** (Gao et al., Science Robotics 2024) — Mid-360 on 280mm quad, 20+ m/s flights.
- **Swarm-LIO2** (HKU MARS Lab, IEEE T-RO 2024) — decentralized LIO for UAV swarms.
- **Point-LIO** (HKU MARS Lab, 2023) — per-point updates at 4-8 kHz on ARM.
- **FAST-LIVO2** (2024) — LiDAR-inertial-visual onboard autonomous UAV navigation.

### Our Differentiation

| Approach | Filter Type | FEJ/OC | Consistency | Degeneracy | Cov Inflation |
|----------|------------|--------|-------------|------------|---------------|
| Inv-LIO | InEKF | N/A | Lie group | No | No |
| Eq-LIO | Equivariant | N/A | Lie group | No | No |
| DALI-SLAM | ESKF | No | No | Jacobian remapping | No |
| LODESTAR | Schmidt-KF | No | No | State classification | No |
| OR-LIM | ESKF | No | No | Surfel observability | No |
| [1] (2026) | BA (MAP) | FEJ | Yes (BA) | No | No |
| **Ours** | **IEKF** | **FEJ** | **Yes** | **FIM eigendecomp** | **CRLB-based** |

## Mathematical Structure

### 1. System Formulation
- State: x = [R, p, v, bg, ba, g] ∈ SO(3) × R^15
- IMU propagation model (continuous-time)
- Point-to-plane measurement model: h_i(x) = n_i^T (R p_i^L + p - q_i) = 0
- Measurement Jacobian H derivation w.r.t. error state

### 2. Observability Analysis (novel — no prior work for point-to-plane LIO)

**Step 1: Continuous-time system**
- State: x = [R, p, v, bg, ba] (ignore gravity/extrinsics — observable, don't affect null space)
- Process model: IMU propagation (Ṙ = R[ω-bg]×, ṗ = v, v̇ = R(a-ba) + g, ḃg = 0, ḃa = 0)
- Measurement model: h_i(x) = n_i^T * (R * p_i^L + p) + d_i (scalar per point)

**Step 2: Measurement Jacobian H**
- ∂h_i/∂δθ = -n_i^T * R * [p_i^L]× (rotation perturbation)
- ∂h_i/∂δp = n_i^T (position perturbation)
- ∂h_i/∂δv = 0, ∂h_i/∂δbg = 0, ∂h_i/∂δba = 0

**Step 3: Process Jacobian F** — linearize continuous-time dynamics around true state

**Step 4: Observability matrix** — O = [H; H*F; H*F²; ...] until rank stabilizes

**Step 5: Null space derivation** — evaluate O at true state
- Expected 4D null space: global yaw (1D) + position-yaw coupling (3D)
- Derive explicit closed-form N = [n_yaw, n_px, n_py, n_pz]

**Step 6: Standard IESKF breaks observability**
- Evaluate O at estimated state (different linearization point per iteration and per time step)
- Show rank(O_estimated) > rank(O_true) → spurious information injected

**Step 7: FEJ-IESKF preserves observability**
- Evaluate O with Jacobians frozen at first estimate
- Prove null(O_FEJ) = null(O_true)

**IESKF-specific contribution** (novel beyond Huang et al.):
- The IESKF has TWO levels of re-linearization that break observability:
  1. Across time steps (classic FEJ problem, same as standard EKF)
  2. Within one update (IESKF iterates re-evaluate H at x_0, x_1, x_2...)
- FEJ fixes both simultaneously: frozen H at propagated state used for all iterations
  and for the covariance update
- Define the "effective Jacobian" of the IESKF update (enters covariance after convergence)
- Prove that FEJ-IESKF effective Jacobian preserves the null space

### 3. FEJ-IESKF Formulation
- Store state estimate x_0 at the beginning of each update cycle (propagated state)
- Evaluate measurement Jacobian H at x_0 for Kalman gain and covariance update
- Let IESKF iterations refine state using re-linearized residuals (convergence preserved)
- Point-set locking: freeze the set of valid correspondences from iteration 0
- Prove: with FEJ, dim(null(O_FEJ)) = dim(null(O_true))
- Computational cost analysis: negligible overhead (one stored matrix, one precomputed HTH)

### 4. FIM and Degeneracy Detection
- Fisher Information Matrix: FIM = H^T R^{-1} H (6×6 pose block)
- Eigendecomposition: FIM = V Λ V^T
- λ_i small → direction v_i poorly constrained
- Condition number κ = λ_max / λ_min as scalar health indicator

### 5. CRLB-Based Covariance Inflation
- Cramer-Rao Lower Bound per direction: σ²_CRLB(i) = R / λ_i
- For each eigenvector v_i:
  - Compute projected EKF variance: σ²_EKF(i) = v_i^T P v_i
  - If σ²_EKF(i) < σ²_CRLB(i): inflate P += (σ²_CRLB(i) - σ²_EKF(i)) v_i v_i^T
- Guarantees published covariance is never smaller than physically achievable minimum
- Does not modify the EKF state — only the published covariance

### 6. Drift Detection
- Partition point-to-plane residuals by map point age
- Old points (age > threshold): residual mean r_old
- New points (age ≤ threshold): residual mean r_new
- Drift ratio: η = r_old / r_new
- η >> 1 indicates map staleness (old geometry no longer consistent with current pose)
- Mathematical justification: under no drift, E[r_old] ≈ E[r_new]; under drift,
  old points accumulate systematic registration error

## Paper Structure

1. **Introduction** — motivation for consistent covariance in LIO, gap in IEKF literature
2. **Related Work** — OC-EKF/FEJ history, degeneracy-aware LIO systems, differentiation table
3. **Preliminaries** — FAST-LIO2 IEKF formulation, point-to-plane measurement model
4. **Observability Analysis** — null space derivation, proof of false observability in IEKF
5. **Proposed Methods**
   - 5.1 FEJ-IEKF formulation and consistency proof
   - 5.2 FIM-based degeneracy detection
   - 5.3 CRLB-based covariance inflation
   - 5.4 Online drift detection
6. **Experimental Validation**
   - Covariance consistency (NEES against mocap ground truth)
   - Degeneracy detection in varied geometry
   - Drift detection correlation with loop closure error
   - Accuracy comparison (ATE, RPE) — show FEJ-IEKF matches or improves standard IEKF
   - Computational cost analysis
7. **Conclusion**

## Experimental Plan

Reuses data and experiments from the Master's thesis:
- **Mocap arena flights** (Experiments 6, 8 from thesis): NEES analysis, covariance envelope plots
- **University loops** (Experiments 6, 7, 9 from thesis): degeneracy eigenvalues, drift indicator,
  loop closure error correlation
- **Additional**: compare standard IEKF vs FEJ-IEKF vs FEJ-IEKF+CRLB on public datasets
  (Hilti, NTU VIRAL) for reproducibility

## Target Venue

IEEE Robotics and Automation Letters (RA-L), with ICRA/IROS presentation option.

## Uncertainty-Aware Map (Potential Additional Contribution)

### Problem

FAST-LIO treats map points as deterministic (zero uncertainty). Covariance converges even with FEJ-IESKF
because the map anchors everything — drift is absorbed into map distortion, invisible to the filter.

### Prior Art — VoxelMap Family

**This idea is NOT novel as standalone.** VoxelMap (Yuan et al., RA-L 2022) propagates pose covariance at
insertion time into per-voxel plane covariance and uses it to inflate measurement noise in the IEKF.

| System | Year | Approach |
|---|---|---|
| VoxelMap [24] | 2022 | Pose → plane covariance, adaptive R per voxel |
| VoxelMap++ [25] | 2023 | Mergeable voxels, same uncertainty model |
| PV-LIO [26] | 2023 | VoxelMap uncertainty + FAST-LIO2 IEKF |
| LOG-LIO2 [27] | 2024 | Per-point uncertainty with incidence angle, O(1) LUFA propagation |
| AKF-LIO [28] | 2025 | Empirical per-voxel R from innovation residuals |
| LIO-GVM [29] | 2024 | Gaussian voxel map, distribution divergence |

### What IS Novel

Combining FEJ-IESKF with uncertainty-aware map for complete filter+map consistency:
- FEJ fixes filter-level inconsistency (prevents false observability)
- Map uncertainty fixes map-level inconsistency (prevents covariance anchoring)
- No published work combines both approaches
- Our formulation: per-point scalar `R_i = R_sensor + n^T J P_pose J^T n` in ikd-tree (lightweight
  vs VoxelMap's full 6x6 plane covariance)

### Critical Insight: Uncertainty-Aware Map Still Converges

Per-point R inflation raises the covariance steady state but does NOT produce distance-dependent growth.
In 1D: P_ss = sqrt(Q * r0), independent of distance traveled. The map always provides finite information
(R_i is finite), preventing unbounded growth. After 10m or 1000m, same steady-state P.

This is a fundamental limitation of ANY scan-to-map approach: as long as you match against the map, the
map constrains you. Drift is invisible because the map and state drift together — residuals stay small.

### Distance-Based Drift Floor (Complementary)

The only way to get truly growing covariance: add explicit drift process noise proportional to distance:
`Q_drift = alpha_min * delta_d * I_pos`, calibrated from loop closure experiments.

This is 3 lines of code and solves the problem that uncertainty-aware map cannot. For downstream SLAM
integration (semantic loop closure), this is what actually matters.

### Decision

Keep RA-L focused on FEJ-IESKF + observability analysis (strong standalone contribution). The uncertainty-
aware map and drift floor are thesis material — they require honest discussion of limitations that would
dilute the FEJ paper's clean message.

## Future Work: OC-IEKF (Separate Paper)

OC-IEKF (full observability-constrained IEKF) explicitly computes the null space of the true
nonlinear system and projects Jacobians to preserve it. This is more rigorous than FEJ, which
only ensures consistency by fixing the linearization point.

OC-IEKF is a separate paper, not part of this one:
- It requires proving the null space projection preserves IEKF convergence properties
- The practical improvement over FEJ in visual-inertial systems was marginal (Huang et al.)
- It needs to demonstrably outperform FEJ-IEKF to justify the added complexity

**Strategy**: publish FEJ-IEKF + CRLB first (this paper). If experiments later show OC-IEKF
gives measurable improvement over FEJ-IEKF, publish as a follow-up. If the difference is
marginal, the FEJ paper already fills the literature gap and OC-IEKF becomes a negative
result (still publishable but lower impact).

### OC-IEKF Paper Plan

**Target venue**: IEEE Transactions on Automatic Control (TAC) or RA-L (theoretical focus).

**Title**: *"Observability-Constrained Iterated Extended Kalman Filtering for Range-Inertial
State Estimation"*

**Key contributions**:
- General OC-IEKF framework with formal proof of null space preservation under iteration
- Proof of convergence: IEKF iterations still converge with the null space projection
- Two instantiations demonstrating generality:
  - 3D LiDAR (point-to-plane): null space is global yaw (4D with position-yaw coupling)
  - 2D laser (point-to-line): null space includes global yaw + Z-axis (planar motion)
- Conditions under which OC-IEKF outperforms FEJ (formalized):
  - Poor initialization (first estimate far from truth)
  - Large inter-scan motion (aggressive maneuvers)
  - Long trajectories with accumulated drift
  - Degenerate-to-non-degenerate transitions

**Validation**: simulation only (no hardware dependency, fully reproducible)
- Simulated 3D LiDAR + IMU and 2D laser + IMU with configurable noise
- Environments: feature-rich box, corridor, open plane (controlled degeneracy)
- Monte Carlo NEES analysis (hundreds of runs per condition) — gold standard for consistency
- Compare: standard IEKF vs FEJ-IEKF vs OC-IEKF across all scenarios
- Sweep initialization error, motion aggressiveness, degeneracy level
- Show OC-IEKF wins specifically in bad-init and high-motion scenarios
- 2D/3D duality demonstrates the framework handles different null space structures

**Publication sequence**: ICUAS (system) → RA-L (FEJ + CRLB) → TAC/RA-L (OC-IEKF theory).

## Key References

- [1] A Consistency-Improved LiDAR-Inertial Bundle Adjustment, arXiv 2602.06380, Feb 2026
- [2] DALI-SLAM, ISPRS J. Photogrammetry and Remote Sensing, 2025
- [3] LODESTAR, arXiv 2511.09142, 2025
- [4] OR-LIM, ISPRS J. Photogrammetry and Remote Sensing, 2024
- [5] LION, Experimental Robotics (ISER), 2021
- [6] Huang, Mourikis, Roumeliotis — FEJ-EKF, ISER 2008/2009
- [7] Huang, Mourikis, Roumeliotis — OC-EKF, IJRR 2010
- [8] Hesch et al. — OC-VINS, IJRR 2014
- [9] Xu, Cai et al. — FAST-LIO2, TIE 2022
- [10] LIO-EKF (Vizzo et al.), arXiv 2311.09887, ICRA 2024
- [11] Li & Mourikis — Consistent EKF-based VIO, IJRR 2013
- [12] Chen, Yang, Geneva, Huang — FEJ2, ICRA 2022
- [13] Jia et al. — FEJ-VIRO, IROS 2022
- [14] Inv-LIO (Xia et al.), IEEE T-ASE 2023
- [15] Eq-LIO (Tao et al.), arXiv 2409.06948, 2024
- [16] Invariant-DLIO, IEEE 2025
- [17] SUPER MAV (Gao et al.), Science Robotics 2024
- [18] Swarm-LIO2, IEEE T-RO 2024
- [19] Point-LIO, 2023
- [20] T-ESKF, RA-L 2025, arXiv 2510.23359
- [21] OpenVINS (Geneva et al.), 2019 — reference FEJ implementation for VIO
- [22] Super-LIO (Wang et al.), RA-L 2026, arXiv 2509.05723 — efficient LIO, FAST-LIO IEKF unmodified
- [23] I2EKF-LO, IROS 2024 — dual-iteration EKF for LiDAR odometry
- [24] VoxelMap (Yuan et al.), RA-L 2022, arXiv 2109.07082 — probabilistic voxel mapping with pose covariance
- [25] VoxelMap++ (Yuan et al.), 2023, arXiv 2308.02799 — mergeable voxels
- [26] PV-LIO (Tsoi), 2023 — VoxelMap uncertainty + FAST-LIO2 IEKF
- [27] LOG-LIO2, 2024, arXiv 2405.01316 — per-point uncertainty with incidence angle, LUFA
- [28] AKF-LIO (Xie et al.), 2025, arXiv 2503.06891 — adaptive Kalman filter for Gaussian map LIO
- [29] LIO-GVM (Ji et al.), RA-L 2024, arXiv 2306.17436 — Gaussian voxel map for LIO
- [30] GICP (Segal et al.), RSS 2009 — per-point covariance weighting in registration (precursor)
- [31] Voxel-SLAM, 2024, arXiv 2410.08935 — VoxelMap pipeline for complete SLAM

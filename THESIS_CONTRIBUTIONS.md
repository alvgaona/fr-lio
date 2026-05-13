# Master's Thesis Contributions Summary

**Working title:** *Flight-Ready LiDAR-Inertial Odometry and SLAM: Principled
Uncertainty, Self-Correcting Maps, and Real-Time Engineering for UAV Autonomy*

**Author:** Alvaro Gonzalez
**Compiled:** 2026-04-20

---

## Contribution 1: Real-Time Flight-Ready LIO Engineering

System-level modifications to FAST-LIO2 enabling safe multirotor deployment on
Jetson Orin NX with Livox Mid-360:

- **200 Hz IMU-rate odometry** via dedicated executor thread (separates IMU
  callback from scan processing)
- **Sliding cube box-deletion** bounds map memory for long flights
- **Velocity publication in body frame** for downstream flight control
- **Race-condition fix** in scan/IMU buffer synchronization (`sync_packages()`
  mutex locking)
- **Path publication decoupled from estimate rate** to prevent serialization
  bottleneck
- **Lidar preprocessing moved outside `mtx_buffer` lock** for minimal contention

Result: FAST-LIO2 that can actually fly a drone at full rate, validated on hardware.

---

## Contribution 2: Per-Point Covariance for Measurement Noise

Replaces FAST-LIO2's hand-tuned scalar `LASER_POINT_COV = 0.001` with a
data-driven per-point variance derived from the local k-NN neighborhood already
used for plane fitting (GICP-style anisotropy):

```
R_i = sigma_range^2 + n_i^T Sigma_p_i n_i
```

### Results (no-LC IESKF RMSE)

| Environment | Baseline (scalar) | Per-point Sigma | Improvement |
|---|---|---|---|
| Cube room | 0.576 m | 0.448 m | **+22.3%** |
| Square corridor | 0.651 m | 0.450 m | **+30.8%** |

### Implementation

Pre-scaling trick: scale `h_x` rows and `h` entries by `1/sqrt(R_i)`, pass
`R=1.0` to the IKFoM filter. Mathematically equivalent to per-measurement R
without modifying the IKFoM toolkit. Drop-in integration.

### Related work positioning

- **UA-LIO (IEEE TIM 2025):** per-point k-NN covariance, eigenvalue-normalized
  to (1,1,epsilon), distribution-to-distribution cost. Closest competitor.
- **iG-LIO (RA-L 2024):** per-voxel covariance, full GICP cost.
- **LOG-LIO2 (arXiv 2024):** per-point covariance from physical sensor model
  (range, bearing, incident angle, roughness) — not k-NN based.

**This work's differentiation:** preserves FAST-LIO2's weighted point-to-plane
cost (drop-in via pre-scaling, no IKFoM modification); no eigenvalue
normalization (uses actual k-NN spatial magnitudes); lighter per-correspondence
cost than UA-LIO's Cholesky.

---

## Contribution 3: Unified Per-Point Sigma Across Measurement + Drift Cost

The same `Sigma_p` that weights the IESKF measurement noise is also used in the
scan-to-scan CRLB FIM weighting, replacing the scalar empirical `R_s2s`:

```
FIM = sum_i (1 / (sigma_range^2 + n_i^T Sigma_p_i n_i)) * J_i^T J_i
P_rel = inv(FIM)
```

**Unified principle:** one derivation from k-NN spatial covariance eliminates
two hand-tuned constants (`LASER_POINT_COV` in IESKF, scalar `R_s2s` in CRLB).
Unlike UA-LIO/iG-LIO which only use their covariance in the measurement update,
this work applies it consistently to both cost terms.

---

## Contribution 4: Honest Published Covariance for Flight-Safe Autonomy

The scan-to-scan CRLB drift accumulation inflates the published `P_drift` to
reflect registration uncertainty over time, providing downstream consumers
(flight controllers, factor graph SLAM, multi-sensor fusion) with calibrated
sigma — derived only from sensor specs, no per-environment tuning.

### Key framing for the thesis (pre-empts reviewer question)

> LC corrects the past trajectory; calibrated forward sigma propagation tells
> the autonomy stack how much to trust the current pose **right now**,
> continuously between LC events. Both are needed for safe autonomy — they
> serve different operational questions.

### Calibration analysis (honest)

- Raw filter: overconfident ~5 orders of magnitude (NEES ≈ 400k)
- CRLB + bias-walk floor: NEES ≈ 12 (pos) / ≈ 3 (rot)
- Position overconfidence is a fundamental limitation of per-edge
  independent-noise models when IMU bias drift is temporally correlated —
  noted in the limitations section

---

## Contribution 5: Source-Pose-Tagged Map Correction

After loop closure computes corrected trajectory poses, every map point
transforms by the Delta of its source keyframe. Each map point carries the
index of the keyframe that inserted it; `correct_map(original, corrected)`
transforms per-point and rebuilds the spatial index.

### Results (mean point-to-GT-wall distance)

| Environment | Pre-correction | Post-correction | Improvement |
|---|---|---|---|
| Cube room | 235 mm | 87 mm | **+63%** |
| Square corridor | 276 mm | 40 mm | **+86%** |

Validated in Python sim with both positive (corridor) and negative (large cube,
no drift) controls.

---

## Contribution 6: Shadow Global Map Architecture for LiDAR-Inertial SLAM

Novel architectural pattern: working `OnlineMap` stays small and cube-pruned
(fast registration, bounded memory); a separate `GlobalMap` absorbs points
evicted from the working map (with their source-pose tags) and supports the
same per-source-pose Delta correction.

### Results

| Environment | Working map | Global map (shadow) | Coverage ratio |
|---|---|---|---|
| Cube | 8,254 pts; 87 mm post-corr | **52,438 pts; 44 mm post-corr** | **6.4x** |
| Square corridor | 1,978 pts; 40 mm post-corr | **16,964 pts; 49 mm post-corr** | **8.6x** |

The shadow architecture preserves the entire traversed environment for LC
correction without sacrificing working-memory bounds — a third alternative to:

- LIO-SAM's keyframe-graph + global map rebuild
- RTAB-Map's submap approach

**This is the strongest standalone novelty.** Promotes the system from LIO to
**LiDAR-inertial SLAM** — the thesis contribution should be framed as SLAM,
competing against LIO-SAM rather than against FAST-LIO2.

---

## Contribution 7: C++ Production Implementation

All Python-validated contributions ported to FAST-LIO2 C++ on `feat/scan-to-scan`
branch (commit `9f97540`):

- Per-point Sigma in `h_share_model` (replaces `LASER_POINT_COV`)
- Per-point Sigma in `compute_scan_to_scan_covariance` (replaces empirical
  `R_s2s`)
- New helper `compute_neighborhood_cov()` in `common_lib.h`
- Config flags `mapping.use_perpoint_cov`, `mapping.point_range_noise_std` in
  `mid360.yaml`
- Backward-compatible (default off; single flag enables)
- Built and verified on Jetson Orin NX

**Pending:** shadow GlobalMap C++ port (architecture validated in Python, ready
for ikd-Tree integration).

---

## Experimental Rigor

Multi-configuration comparison harness (10+ configs across 2 environments):

- Demonstrated empirically that CRLB-weighted pose graph LC **does not beat
  hand-tuned fixed weights for ATE** — honest negative result
- Demonstrated that **per-point IESKF + LC is the first config to beat fixed
  weights** (+10% in cube room)
- Analysis of optimization-vs-calibration tension and LC-vs-chain-sigma
  dominance ratio
- False-LC robustness sweep showing edge-weighting alone is insufficient for
  outlier resilience (motivation for robust kernels as future work)

---

## Honest Limitations

- Published sigma is a Cramer-Rao lower bound on chain drift derivable from
  sensor noise alone; does not include map-aging drift in degenerate
  environments
- No per-deployment calibration (design choice) means published sigma is
  optimistic by 2-4x in position under severe drift — document, let downstream
  consumers add safety margins
- Shadow map memory grows with environment coverage; multi-km deployments need
  submap pivoting (future work)

---

## Headline numbers for the abstract

- **+22-31%** no-LC IESKF RMSE from per-point Sigma
- **+10%** LC ATE vs hand-tuned fixed-weight LC (cube room)
- **+63-86%** map-to-GT-wall distance improvement via source-pose correction
- **6-9x spatial coverage** with shadow global map
- **Two hand-tuned magic numbers** (`LASER_POINT_COV`, `R_s2s`) eliminated by
  one principle
- **Trajectory AND map** corrected post-LC, no per-deployment tuning
- **200 Hz IMU-rate odometry** enabling real-time UAV flight on Jetson Orin NX

---

## Future Work

### Large-scale deployment (multi-km flights)

Three layered scaling extensions (in order of typical need):

1. **Submap pivoting** (~10 km scale): per-submap Delta instead of per-point;
   standard pattern in LIO-SAM, RTAB-Map
2. **Surfel / plane compression** (orthogonal): 10-100x memory reduction for
   structured indoor / urban environments
3. **Out-of-core submaps** (unbounded scale): far-from-sensor submaps
   serialized to disk, load-on-demand

### Other threads

- C++ shadow GlobalMap port (production architecture; ~1-2 weeks)
- Real-data validation on Mid-360 rosbag with mocap GT
- Per-point Sigma-weighted map correction (Sigma as per-point trust measure)
- Multi-session SLAM with persistent global map

---

## Thesis narrative in one sentence

> From hand-tuned scalar noise and uncorrectable drifted maps to data-driven
> per-point uncertainty, source-pose-tagged map correction, and shadow-tree
> SLAM architecture — all without per-deployment calibration, all flight-ready
> on real hardware.

---

## Files and commits

**Python sim** (`src/FAST_LIO/scripts/`):

- `sim_iekf_3d.py` — per-point Sigma in find_correspondences, iekf_update,
  compute_scan_to_scan_covariance; OnlineMap + GlobalMap
- `sim_square_corridor_no_lc.py` — IESKF runner with sliding cube, bias-walk,
  shadow map
- `sim_square_corridor_compare.py` / `sim_cube_circle_compare.py` — 11-config
  LC comparison harness
- `sim_perpoint_diagnosis.py` — standalone per-point Sigma validation
  (+22-31% no-LC RMSE)
- `sim_map_correction.py` — end-to-end LC + map correction + shadow map
- `sim_empirical_edge_noise.py` — per-edge Mahalanobis diagnostic
- `sim_falselc_robustness.py` — false-LC sweep

**C++ port** (`feat/scan-to-scan` branch):

- commit `9f97540`: per-point covariance for IESKF + scan-to-scan CRLB
- `src/laserMapping.cpp`, `include/common_lib.h`, `config/mid360.yaml`

**Python map correction** (`feat/correct-map` branch):

- commit `775f41f`: source-pose-tagged map correction
- commit `2b79168`: shadow GlobalMap for Option A architecture validation

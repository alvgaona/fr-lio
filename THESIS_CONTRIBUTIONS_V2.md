# Master's Thesis — Contributions (Revised Post-Validation)

**Working title:** Flight-Ready Uncertainty-Aware LiDAR-Inertial SLAM
with Self-Correcting Maps for Aerial Autonomy

**Revised:** 2026-04-20, after real-data ablation on long feature-rich
corridor bag (104 m, Mid-360).

---

## Overall thesis claim

> **FAST-LIO2 is excellent as a short-range odometry system but does not
> solve the memory-bounded, correctable-map, multi-session SLAM problem.**
> This thesis extends the FAST-LIO2 family into that regime with
> principled uncertainty weighting, a correctable shadow-map architecture,
> and real-time loop closure + pose-graph optimization — all while
> preserving the flight-ready engineering properties of the original.

The claim is *not* that this work outperforms FAST-LIO2 where FAST-LIO2
already excels. It is that this work *enables a regime* where FAST-LIO2
cannot operate: bounded-memory, long-traversal, globally-consistent,
correctable-map aerial autonomy.

---

## Empirical foundation

Ablation on a 104 m indoor corridor bag (Livox Mid-360), five
configurations, same sensor input, shutdown-metric capture:

| Config | Cube | drift_pct_map | Z (end) | Notes |
|---|---|---|---|---|
| A′ (stock FAST-LIO2, native cube) | 300 | 0.12% | +0.04 m | Reference — environment fits cube |
| B′ (+per-point Σ + Huber, native cube) | 300 | 0.14% | +0.05 m | No benefit in well-fed regime |
| A (baseline, bounded cube) | 50 | 2.87% | −2.85 m | Cube-bounded regression |
| B (+per-point Σ + Huber, bounded cube) | 50 | 2.72% | −2.20 m | Robustness mechanism |
| C (full SLAM stack, bounded cube) | 50 | **1.29%** | **+0.08 m** | Full contribution |

Full results with timing breakdowns and LC gate statistics:
`src/FAST_LIO/LONG_CORRIDOR_RESULTS.md`.

---

## Contribution 1: Flight-Ready FAST-LIO2 Engineering

Systems-level modifications to FAST-LIO2 enabling safe deployment on
Jetson Orin NX + Livox Mid-360 for multirotor flight:

- 200 Hz IMU-rate odometry via a dedicated executor thread that decouples
  the IMU callback from LiDAR scan processing
- Sliding cube box-deletion for bounded working-map memory in long flights
- Velocity publication in body frame for downstream controllers
- Race-condition fix in `sync_packages` buffer synchronization
- Path publication decoupled from estimate rate to prevent serialization
  bottleneck
- Lidar preprocessing moved outside `mtx_buffer` lock

**Status:** shipped on `main` branch. Not a validation contribution per se;
this is the engineering baseline the rest of the thesis builds on.

---

## Contribution 2: Per-Point Σ + Huber Kernel for Bounded-Regime Robustness

Replaces FAST-LIO2's hand-tuned scalar `LASER_POINT_COV` with a data-driven
per-point variance from the same k-NN neighborhood used for plane fitting:

```
R_i = σ_range² + n_iᵀ Σ_p_i n_i
```

Combined with a Huber M-estimator on standardized residuals (cap at
`|r|/√R_i = k`) to prevent divergence on drifted-map revisits where the
per-point Σ alone becomes overconfident.

Implementation: pre-scaling of IKFoM `h_x` rows and `h` entries by
`1/√R_i`, mathematically equivalent to per-measurement `R` without
modifying the IKFoM toolkit.

### Validation

**Empirical finding:** this is a **robustness mechanism for marginal
regimes**, not a free upgrade.

| Regime | Config comparison | Drift change | Effect |
|---|---|---|---|
| Native cube (well-fed filter) | A′ → B′ | 0.12% → 0.14% | None (within noise) |
| Bounded cube (marginal filter) | A → B | 2.87% → 2.72% | −5% (modest) |
| Bounded cube + revisits | B → C (via Huber) | N/A | Prevents divergence in dungeon tests |

**Key finding from dungeon tests:** without Huber, per-point Σ diverged
at revisits in feature-poor environments due to the "drifted self-
consistent map" failure mode (small Σ_p → high measurement weight →
filter pulls toward drifted map). Huber caps the effective weight and
restores stability.

**Interpretation for the thesis:** Contribution 2 extends the
**minimum viable cube size** for FAST-LIO2, not the accuracy at any
fixed cube. The value appears only when the filter is under constraint
pressure.

---

## Contribution 3: Source-Pose-Tagged Map Correction

Each point inserted into the working ikd-Tree carries the index of the
keyframe that inserted it, stored in the `normal_x` field of the PCL
point payload. On loop closure:

1. iSAM2 returns corrected keyframe poses
2. Each map point transforms by `Δ_k = corrected[k] · original[k]⁻¹`
   of its source keyframe
3. ikd-Tree rebuilt with transformed points

Unified with the published covariance story (Contribution 4) by a single
k-NN Σ derivation — one principle replaces both hand-tuned `LASER_POINT_COV`
and the scalar `R_s2s` in scan-to-scan CRLB.

### Validation

- **Unit tests:** `test/test_map_correction.cpp`, 9/9 passing. Covers
  identity correction, per-keyframe Δ, out-of-range source_idx,
  rotation+translation, voxel-key exactness.
- **Live rosbag:** Z-axis endpoint in map frame recovered from −1.26 m
  (filter drift, sensor "in the floor") to +0.08 m (physically plausible,
  sensor at floor level). 3 corrections fired, each transforming
  ~50k–77k shadow points.
- **Multi-LC consistency fix:** initial implementation had a frame-
  accumulation bug on the second and subsequent LC events (evicted
  points entered the shadow in odom frame while older points were in
  map frame). Fixed by pre-transforming evictions through `T_map_odom`
  before absorbing. See commit `90cc24f`.

---

## Contribution 4: Shadow Global Map Architecture

A second ikd-Tree, separate from the working tree, absorbs points
evicted from the working tree (sliding cube box-deletion) with voxel
deduplication. The shadow tree grows with environment coverage while
the working tree stays memory-bounded.

Unlike FAST-LIO2's `pcl_wait_save` (an unbounded chronological
accumulator that is never queried and never corrected), the shadow
tree is:
- Queryable like a normal ikd-Tree
- Voxel-deduplicated (so revisited regions do not bloat memory)
- Correctable (receives `correct_map` transforms)
- Published to downstream (`/cloud_shadow_map`, periodic)

### Validation

- Real-data rosbag: 64,471 peak points after 104 m traversal,
  87.5% voxel dedup on evictions (indicating heavy revisit structure)
- Total evictions: 517,460 points. Unique absorptions: 64,471.
  (1 – 64471/517460 ≈ 87.5% dedup.)

**Position versus LIO-SAM and RTAB-Map:**

- LIO-SAM maintains keyframe-submaps and rebuilds a global map from
  scratch after each LC
- RTAB-Map uses submap-level corrections
- This work: single global ikd-Tree with per-point source-pose tagging,
  correctable in-place (no rebuild-from-scratch), O(log n) queryable

---

## Contribution 5: Real-Time Loop Closure + iSAM2 Pose-Graph Optimization

Real-time LC detection and PGO running on a dedicated background thread
with bounded-queue drop-oldest backpressure, never stalling the main
registration loop.

### Components

- Keyframe database (`include/lc_keyframe_db.hpp`): thread-safe store
  with pose + downsampled scan + `source_idx`, spatial index rebuilt
  every 10 adds
- Candidate detection: radius + time-gap filter (LIO-SAM-style)
- Verification: GICP between current and candidate downsampled scans,
  fitness-gated
- PGO: GTSAM iSAM2 with `BetweenFactor<Pose3>` for odom edges
  (derived from FAST-LIO pose differences) and LC edges (from GICP)
- Correction trigger: fires `correct_map` when max keyframe pose delta
  exceeds thresholds; coalesces bursts with an in-flight flag

### False-positive defense

Two principled gates catch perceptual aliasing in corridor geometry:

1. **`max_rel_t_m`:** reject if GICP's refined `|t_rel|` exceeds threshold.
   Corridor walls look locally similar from different positions →
   GICP can converge on wall-to-wall matches at wildly different
   physical locations.
2. **PGO-consistency sanity gate:** reject if PGO's `max_dpos` exceeds
   `3 × |t_rel|` and > 1 m. A true revisit with `|t_rel| = X` induces
   PGO corrections of order X; a false LC slipped past per-correspondence
   gates manifests as 10-20× larger PGO deltas.

### Validation

- 104 m corridor bag, 8 candidates evaluated, 6 accepted by GICP,
  3 rejected by sanity gate, 3 corrections actually triggered
- Drift reduction within bounded-cube regime: A → C = **−55%**
  (2.87% → 1.29% of path length)
- Z-axis recovery: −2.85 m → +0.08 m (Z drift from "sensor in ground"
  to "sensor on floor")

### Real-time compliance

All on async LC thread; main loop (10 Hz, 100 ms budget) unaffected.

| Operation | Mean (ms) | Budget (ms) | Headroom |
|---|---|---|---|
| GICP per LC | 23.92 | 100 | 4.2× |
| iSAM2 incremental update | 1.10 | 50 | 45× |
| correct_map tree rebuild | 14.97 | 300 | 20× |

Dropped keyframes: 0 across all test runs.

---

## Contribution 6: Empirical Characterization of Operating Regimes

A five-configuration ablation study on the same bag distinguishes
regime-dependent contribution value from regime-independent claims.

### Regime-by-regime positioning

| Regime | Best system | Rationale |
|---|---|---|
| Short bag, unbounded memory, no revisit requirement | **Stock FAST-LIO2 (A′)** | 0.12% drift, simplest, no LC false-positive risk |
| Short bag, well-fed filter | **FAST-LIO2 (A′) ≈ our B′** | Per-point Σ offers no benefit when filter has ample context |
| Bounded cube (Jetson UAV, long flight), no revisit | **Our B (partial benefit)** | Per-point Σ extends minimum cube size modestly |
| Bounded cube + revisits | **Our C (full stack)** | LC + correction recovers 55% of drift, restores Z-axis |
| Degenerate geometry (flat-wall dungeon) | **Known failure** | Geometric SLAM fundamentally cannot localize along corridor axis |

### Thesis positioning vs. prior work

- **vs FAST-LIO2:** orthogonal at native cube; enables bounded-cube regime
- **vs LIO-SAM:** same SLAM regime; our in-place shadow correction is
  cheaper than keyframe-submap rebuild; our per-point Σ and Huber
  robustness are distinct contributions
- **vs UA-LIO / LOG-LIO2:** closest peers on per-point covariance, but
  theirs are evaluated only on single-cube configurations; the
  regime analysis here is novel

---

## Contribution 7: C++ Production Implementation and Metrics Infrastructure

All algorithmic contributions ported from the Python sim to the
production FAST-LIO2 C++ codebase:

- `feat/correct-map` branch, 12+ commits
- GTSAM 4.2 integrated via conda-forge (`pixi.toml`)
- Pure-math helpers in `include/map_correction.hpp` (unit-tested in
  `test/test_map_correction.cpp`, 9 tests)
- Structured shutdown-metric dump with path length, drift in odom and
  map frames, LC gate breakdown, and timing aggregates
- Configurable via `config/fast_lio.yaml` — all features flag-gated and
  default-off (strict backward compatibility with stock FAST-LIO2)

---

## Honest limitations (for thesis limitations section)

1. **Along-corridor degeneracy:** no geometric LiDAR system can observe
   drift along a featureless corridor axis. Per-point Σ + Huber
   delays but does not eliminate this failure mode.

2. **Feature-poor dungeon environments:** in purely flat-wall geometry,
   even the full stack diverges. LC detection suffers from perceptual
   aliasing, and our false-positive gates catch the worst cases but
   cannot distinguish a true revisit from a false one when geometry is
   genuinely indistinguishable. Matches known limitations of all
   geometric-only LiDAR SLAM (LIO-SAM, UA-LIO, LOG-LIO2, iG-LIO).

3. **One-way (non-loop-closing) trajectories:** LC never fires,
   drift accumulates unbounded. Multi-modal fusion (GPS, UWB, visual)
   is the only answer — out of scope here.

4. **Shadow-map correction requires traversal exceeding cube half-
   width:** on short bags in small rooms, the shadow stays empty and
   map-point correction is a no-op (though trajectory correction still
   applies).

5. **Published drift as percent of path length is an indirect metric:**
   without ground truth, we cannot distinguish filter drift from
   physical endpoint offset. The Z-axis recovery test (sensor physically
   on floor) provides a qualitative sanity check where the quantitative
   comparison is ambiguous.

---

## Headline numbers for abstract

- **−55%** endpoint drift (map frame) on 104 m corridor when working
  cube is memory-bounded (2.87% → 1.29% of path length)
- **Z-axis recovery** from −2.85 m (filter drifted into ground) to
  +0.08 m (sensor at floor level) via LC + map correction
- **Real-time compliant:** main 10 Hz loop unaffected by async
  LC+PGO+correction; mean GICP 24 ms, iSAM2 1.1 ms, correct_map 15 ms
- **False-positive defense:** 3 of 8 LC candidates rejected via
  principled gates (GICP fitness, |t_rel|, PGO-vs-GICP consistency)
- **Memory-bounded:** 50 m cube produces 64k-point shadow map with
  87.5% voxel-dedup on revisits (vs. unbounded baseline ikd-Tree)

---

## Thesis chapter structure suggestion

1. **Introduction** — problem statement, regime analysis
2. **Background** — FAST-LIO2, LIO-SAM, iSAM2, per-point covariance literature
3. **Flight-ready engineering** (Contribution 1)
4. **Per-point Σ + Huber** (Contribution 2) — principled weighting for
   marginal regimes
5. **Correctable shadow-map architecture** (Contributions 3+4) —
   source-pose tagging, shadow global map
6. **Real-time LC + PGO** (Contribution 5) — GTSAM iSAM2, false-positive
   defense
7. **Empirical evaluation** (Contribution 6) — 5-config ablation on real
   data, regime analysis, honest limitations
8. **Conclusion** — regime-aware positioning, future work (visual
   fusion, UWB anchors, multi-session)

---

## Python sim contributions (for thesis scientific contribution, not C++)

- `sim_iekf_3d.py`, `sim_square_corridor_*` — 3D IESKF with source-pose
  tagging, shadow global map, pose-graph optimization
- Multi-configuration LC comparison harness (10+ configs, 2 environments)
- Empirical edge-noise diagnostic
- False-LC robustness sweep
- Map correction + shadow map validation (+63%–86% map-to-GT wall
  distance improvement in sim)

These informed the C++ implementation but are not themselves the C++
contribution. The thesis should present them as the "design validation"
step that precedes the C++ production port.

---

## One-sentence contribution statement

> This thesis extends the FAST-LIO2 family to the memory-bounded,
> correctable-map, multi-session SLAM regime by combining principled
> per-point uncertainty weighting, a voxel-deduplicated shadow global
> map with source-pose-tagged corrections, and real-time loop closure
> via GTSAM iSAM2 — validated on real data with a 55% drift reduction
> in the bounded-cube regime, full real-time compliance, and honest
> characterization of remaining failure modes in geometrically
> degenerate environments.

# ICUAS Paper Roadmap

## Positioning

**Title**: "Flight-Ready High-Rate LiDAR-Inertial Odometry for UAVs with Non-Repetitive
Solid-State LiDAR"

**The problem**: The Livox Mid-360 is increasingly adopted on small UAVs for its 360-degree
dome FoV, low weight, and low cost. However, no existing open-source LIO system provides
complete, flight-ready integration: FAST-LIO2 outputs at 10Hz (too slow for flight control),
Point-LIO achieves kHz rates but has not been validated with Mid-360's non-repetitive scanning
pattern, and no system provides end-to-end integration with PX4 flight controllers.

**The gap**: RTLIO (2021) introduced IMU forward propagation for UAV control but uses Velodyne
spinning LiDARs. Point-LIO (2023) processes point-by-point at 4-8kHz but uses a different IMU
model. LIO-EKF (2024) achieves near-IMU rate but lacks Livox support. None address the scan
accumulation problem specific to non-repetitive LiDARs or demonstrate autonomous flight.

**Venue**: ICUAS (International Conference on Unmanned Aircraft Systems) — systems-oriented,
reviewers value practical demonstrations and flight experiments.

## Contributions

### C1: Scan Accumulation for Non-Repetitive LiDARs [IMPLEMENTED]

The Livox Mid-360 at 100Hz produces wildly uneven point distributions per scan (48-8790 points
observed). The accumulator node merges N consecutive scans while preserving per-point timestamps
for motion undistortion.

**Key insight**: Non-repetitive LiDARs require scan accumulation for observability — spinning
LiDARs don't need this because each scan covers the full FoV.

### C2: Unified IMU Forward Propagation Architecture [IMPLEMENTED]

Single-publisher design where the IMU callback is the sole source of odometry, TF, and path.
The scan processing only updates the anchor state. This provides:
- 200Hz odometry output (vs 10Hz from vanilla FAST-LIO2)
- No duplicate timestamps (eliminates evo speed computation failures)
- Sub-millisecond latency for the forward propagation step

### C3: Complete Flight-Ready System Integration [IMPLEMENTED]

End-to-end: Livox Mid-360 → scan accumulation → FAST-LIO2 EKF → IMU forward propagation
→ 200Hz odometry → PX4 flight controller. Including:
- REP 105 frame compliance (odom → imu_link → base_link)
- Static TF for IMU-to-base_link offset
- Proper `use_sim_time` support for rosbag replay
- Mocap ground truth converter for evaluation

## Required Experiments

### E1: Accuracy — Mocap Ground Truth Comparison

**Setup**: Drone with Livox Mid-360 + Pixhawk 4 mini in a motion capture room.

**Trajectories**:
- Hover (static accuracy baseline)
- Slow figure-8 (~0.5 m/s)
- Fast figure-8 (~2.0 m/s)
- Aggressive maneuvers (step inputs, rapid yaw)
- Long duration hover (5+ minutes, drift characterization)

**Metrics** (use `evo`):
- APE (Absolute Pose Error) — overall accuracy
- RPE (Relative Pose Error) — local consistency

**Preliminary result**: 3.2cm mean APE on a 120s trajectory.

**Commands**:
```bash
evo_ape bag2 <bag> /ground_truth/odom /odom -a --sync -p --save_results ape.zip
evo_rpe bag2 <bag> /ground_truth/odom /odom -a --sync --delta 1 --delta_unit s -p
```

### E2: Latency Measurement

**Goal**: Measure end-to-end latency from IMU timestamp to odometry publication.

**Method**: Record wall-clock time at odometry publish, compare with IMU message timestamp.

**Expected**: < 1ms for forward propagation, ~30-50ms for EKF correction (non-blocking).

**What to show**: Latency histogram, comparison with vanilla FAST-LIO2 (100ms latency).

### E3: Rate Comparison

**Goal**: Demonstrate 200Hz vs 10Hz odometry and its impact on control.

**Method**:
- Record odometry at both rates
- Compare control performance (position tracking error, attitude oscillation)
- Show odometry rate histogram

**What to show**: Rate histogram, step response comparison, position tracking RMSE.

### E4: Scan Accumulation Ablation

**Goal**: Show that scan accumulation is necessary for Mid-360.

**Method**: Run with N = 1, 5, 10, 20 accumulated scans.

**What to show**:
- EKF failure rate vs N
- APE vs N (accuracy improves then plateaus)
- Effective odometry rate vs N
- Point count distribution per scan

### E5: Baseline Comparison

Compare against systems on the same rosbag data:

| System     | Why                                     |
|------------|-----------------------------------------|
| FAST-LIO2  | Upstream baseline (10Hz output)         |
| Point-LIO  | Highest-rate alternative (4-8kHz)       |
| LIO-EKF    | Near-IMU rate classical EKF             |

**Metrics**: APE, RPE, output rate, CPU usage.

**Note**: Document which baselines support Mid-360 natively — the gap itself supports the
contribution.

### E6: Real Autonomous Flight (strongest evidence)

**Goal**: Closed-loop autonomous flight using this system as the sole odometry source.

**Scenarios**:
- Waypoint following (square, figure-8)
- Position hold
- Transition between feature-rich and feature-poor regions

**What to show**: Flight video, trajectory plot, position tracking error.

## Paper Structure

1. **Introduction**: Solid-state LiDAR on drones, the rate gap problem, Mid-360 challenges
2. **Related Work**: FAST-LIO2, Point-LIO, RTLIO, LIO-EKF (organized by approach to rate)
3. **System Overview**: Block diagram from LiDAR to flight controller
4. **Scan Accumulation**: Non-repetitive pattern problem, timestamp preservation
5. **IMU Forward Propagation**: Architecture, single-publisher design, error bound
6. **System Integration**: REP 105 frames, PX4 interface, launch configuration
7. **Experiments**: E1-E6
8. **Conclusions**

## Key References

- FAST-LIO2 (Xu et al., IEEE T-RO 2022)
- Point-LIO (He et al., Adv. Intell. Syst. 2023)
- RTLIO (Bai et al., Sensors 2021)
- LIO-EKF (Vizzo et al., ICRA 2024)
- PX4 Autopilot (Meier et al., 2015)

## Implementation Status

All code contributions are implemented and committed.

## Next Steps

### Data Collection (requires hardware)
1. [ ] Record multiple mocap trajectories (hover, figure-8, aggressive, long)
2. [ ] Run `evo_ape` and `evo_rpe` for accuracy numbers
3. [ ] Add latency instrumentation and measure
4. [ ] Run scan accumulation ablation (N = 1, 5, 10, 20)
5. [ ] Install and run Point-LIO and LIO-EKF on the same bags
6. [ ] Demonstrate autonomous flight with PX4
7. [ ] Record flight video

### Writing
8. [ ] Write paper draft
9. [ ] Create system block diagram figure
10. [ ] Create timing diagram figure

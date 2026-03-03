# Changes from Upstream FAST-LIO (ROS2 branch)

This document describes all modifications made to the upstream
[FAST-LIO ROS2 branch](https://github.com/hku-mars/FAST_LIO/tree/ROS2) for integration with a drone
flight controller (Pixhawk 4 mini) using a Livox Mid360 LiDAR.

## Hardware Setup

- **LiDAR**: Livox Mid360 (non-repetitive scanning, ~200Hz IMU, 100Hz point cloud)
- **Flight Controller**: Pixhawk 4 mini, mounted ~12cm below the LiDAR
- **TF tree**: `odom -> imu_link -> base_link` (Livox IMU frame -> Pixhawk frame)

## FAST-LIO EKF State and Process Model

Understanding the upstream mathematics is essential context for the changes below.

### State Manifold

FAST-LIO uses an iterated Error-State Extended Kalman Filter (iESEKF) operating on a 23-DOF state
manifold defined in `include/use-ikfom.hpp` (lines 12-21):

```
x = (pos, rot, offset_R_L_I, offset_T_L_I, vel, bg, ba, grav)
```

| Symbol           | Type  | DOF | Description                           |
|------------------|-------|-----|---------------------------------------|
| `pos`            | R^3   | 3   | Position in world frame               |
| `rot`            | SO(3) | 3   | Rotation from body (IMU) to world     |
| `offset_R_L_I`  | SO(3) | 3   | LiDAR-to-IMU rotation extrinsic       |
| `offset_T_L_I`  | R^3   | 3   | LiDAR-to-IMU translation extrinsic    |
| `vel`            | R^3   | 3   | Linear velocity in world frame        |
| `bg`             | R^3   | 3   | Gyroscope bias                        |
| `ba`             | R^3   | 3   | Accelerometer bias                    |
| `grav`           | S(2)  | 2   | Gravity vector on unit sphere * 9.81  |

Total: 23 DOF (24 state dimensions, but `grav` on S(2) has 2 DOF).

### Continuous-Time Process Model

The process model `get_f()` in `include/use-ikfom.hpp` (lines 47-59) defines the continuous-time
dynamics:

```
dp/dt = v
dR/dt = R * [omega - bg]_x        (where [.]_x is the skew-symmetric matrix)
dv/dt = R * (a_imu - ba) + g
dbg/dt = 0
dba/dt = 0
```

Where:
- `p` is position in world frame
- `R` is rotation matrix from body to world (SO(3))
- `v` is velocity in world frame
- `a_imu` is the raw accelerometer measurement (in body frame, m/s^2)
- `omega` is the raw gyroscope measurement (in body frame, rad/s)
- `bg`, `ba` are gyroscope and accelerometer biases
- `g` is the gravity vector (approximately [0, 0, -9.81]^T in world frame)

### Discrete-Time Integration (in `UndistortPcl`)

`IMU_Processing.hpp` (lines 251-289) discretizes the process model between consecutive IMU samples
using midpoint averaging:

```
omega_avg = 0.5 * (omega_k + omega_{k+1})
a_avg     = 0.5 * (a_k + a_{k+1})
a_avg     = a_avg * G_m_s2 / ||mean_acc||       (unit normalization, see Section 7)
```

Then the EKF predict step (`kf_state.predict(dt, Q, in)`) internally applies:

```
R_{k+1} = R_k * Exp((omega_avg - bg) * dt)
v_{k+1} = v_k + (R_k * (a_avg - ba) + g) * dt
p_{k+1} = p_k + v_k * dt + 0.5 * (R_k * (a_avg - ba) + g) * dt^2
```

### Exponential Map (Rodrigues Formula)

The `Exp(omega, dt)` function in `include/so3_math.h` (lines 37-58) computes the rotation matrix
corresponding to rotating by angular velocity `omega` for duration `dt`:

```
theta = ||omega|| * dt
k = omega / ||omega||              (unit rotation axis)
K = [k]_x                          (skew-symmetric matrix of k)

Exp(omega, dt) = I + sin(theta) * K + (1 - cos(theta)) * K^2
```

This is the Rodrigues rotation formula. For small angles (`theta < 1e-7`), it returns the identity
matrix to avoid numerical instability.

The skew-symmetric matrix `[k]_x` for vector `k = (k1, k2, k3)` is:

```
        [  0  -k3   k2 ]
[k]_x = [  k3   0  -k1 ]
        [ -k2   k1   0  ]
```

## Summary of Changes

### 1. IMU Forward Propagation for 200Hz Odometry

**Problem**: FAST-LIO only publishes odometry at the LiDAR scan rate (10Hz), but the flight controller
needs ~200Hz for stable control.

**Solution**: Added lightweight IMU forward propagation in the IMU callback. Each new IMU sample
propagates the last EKF-corrected state forward and immediately publishes the predicted pose.

**Architecture**:
```
100Hz polling timer:  sync_packages() waits for complete LiDAR+IMU data
                      -> EKF correction -> snapshot anchor state (x_anchor, P_anchor, t_anchor)
IMU callback (200Hz): read anchor -> propagate with new IMU -> publish odometry + TF + path
```

The timer runs at 100Hz but `sync_packages()` only proceeds when a full LiDAR scan (with covering
IMU data) is available, making the effective correction rate equal to the LiDAR scan rate (~10Hz
with accumulation). The IMU callback is the **sole publisher** of odometry, TF, and path — the scan
processing only updates the anchor state. This eliminates duplicate timestamps that occurred when
both the scan processing and IMU callback published to `/odom`.

**Anchor state**: After each EKF correction, the full corrected state is snapshotted into an
`ImuPropState` struct protected by a mutex:

```cpp
struct ImuPropState {
    V3D pos, vel, bg, ba, grav;
    M3D rot;
    Eigen::Matrix<double, 23, 23> P;
    double timestamp;
    bool valid;
};
```

The full 23x23 EKF covariance `P` is stored (not just the 6x6 pose block) because the forward
propagation Jacobian maps from the full error state to the 6-DOF pose error.

**Forward propagation math**: For each incoming IMU sample at time `t_imu` with measurements
`(a_imu, omega_imu)`, compute `dt = t_imu - t_anchor` and propagate:

```
omega = omega_imu - bg_anchor                           (bias-corrected angular velocity)
a_body = a_imu - ba_anchor                              (bias-corrected acceleration in body frame)
a_world = R_anchor * a_body + g_anchor                  (acceleration in world frame)

p_prop = p_anchor + v_anchor * dt + 0.5 * a_world * dt^2
v_prop = v_anchor + a_world * dt
R_prop = R_anchor * Exp(omega, dt)
```

This is mathematically equivalent to the EKF process model (Section above) but applied as a single
large step from the anchor state rather than incrementally between consecutive IMU samples. The
single-step approach is valid because:
1. Biases are treated as constant (they only change at EKF correction)
2. The anchor rotation `R_anchor` is used throughout (no intermediate rotation updates)
3. For the ~5ms between IMU samples, the approximation error is negligible

**Velocity output in body frame**: The odometry message requires velocity in body (child) frame.
Since the EKF estimates velocity in world frame:

```
v_body = R_prop^T * v_prop
```

Where `R_prop^T` is the transpose (= inverse for rotation matrices) of the propagated body-to-world
rotation, transforming world-frame velocity into body-frame velocity.

**Angular velocity output**: Directly from bias-corrected gyroscope:

```
omega_body = omega_imu - bg_anchor
```

This is already in body frame since gyroscope measurements are body-frame quantities.

**Why angular velocity is not an EKF state variable**: A natural question is whether the EKF should
estimate angular velocity rather than using the raw gyroscope. The answer is no, for several reasons:

1. The gyroscope is a **direct sensor** for angular velocity. Unlike position (which must be integrated
   from accelerometers or matched against LiDAR), angular velocity is directly observed at 200Hz with
   low noise. The EKF's contribution is estimating the gyroscope bias `bg`, which it already does.

2. Adding `omega` to the state (3 extra DOF) would require a **process model for angular
   acceleration**. The rigid-body dynamics are:

   ```
   d_omega/dt = I^{-1} * (tau - omega x (I * omega))
   ```

   Where `I` is the inertia tensor and `tau` is the applied torque vector. Neither is available to
   FAST-LIO (they depend on motor commands, aerodynamic forces, and mass properties). Without a good
   model, the EKF would resort to a random-walk or constant-velocity assumption, which would merely
   smooth the gyroscope readings and introduce unnecessary lag.

3. The bias-corrected gyroscope `omega_imu - bg` is already the **minimum-variance estimate** of
   angular velocity given the current EKF state. Any further filtering would trade latency for
   marginal noise reduction — a bad tradeoff for a flight controller that needs responsive attitude
   feedback.

4. Each IMU sample uses its own **live gyroscope reading** (not a stale anchor value), so angular
   velocity updates at the full 200Hz IMU rate with zero additional latency.

**Covariance propagation**: Each forward-propagated odometry message carries a **propagated** pose
covariance that grows between LiDAR corrections, reflecting the true increasing uncertainty of the
IMU-only prediction.

#### EKF Covariance Structure

The EKF maintains a 23x23 covariance matrix `P` over the error state. The DOF indices are:

```
pos: 0-2,  rot: 3-5,  offset_R: 6-8,  offset_T: 9-11,  vel: 12-14,  bg: 15-17,  ba: 18-20,  grav: 21-22
```

After each LiDAR correction, the full 23x23 `P` is snapshotted into the anchor state. For each
subsequent IMU sample at time `t_imu`, with `dt = t_imu - t_anchor`, we compute the propagated 6x6
pose covariance.

#### Propagation Derivation

The forward-propagated pose is a function of the anchor state and the current IMU measurement:

```
p_prop = p + v*dt + 0.5*(R*(a_imu - ba) + g)*dt²
R_prop = R * Exp((omega_imu - bg)*dt)
```

To propagate uncertainty, we linearize around the anchor state. Define the error state perturbation
`δx = [δp, δθ, δR_L, δT_L, δv, δbg, δba, δg]` (23 DOF). The propagated pose error
`[δp_prop, δθ_prop]` is related to `δx` by the Jacobian `J ∈ R^{6x23}`:

```
[δp_prop]       [δp  ]
[       ] = J * [δθ  ]
[δθ_prop]       [... ]
                [δba ]
                [δg  ]
```

The non-zero blocks of J are derived by differentiating the propagation equations:

**Position error** (rows 0-2 of J):

```
∂p_prop/∂δp = I₃                                           (J[0:3, 0:3])
∂p_prop/∂δθ = -R * [a']_× * dt²/2                          (J[0:3, 3:6])
∂p_prop/∂δv = I₃ * dt                                      (J[0:3, 12:15])
∂p_prop/∂δba = -R * dt²/2                                  (J[0:3, 18:21])
```

Where `a' = a_imu - ba` is the bias-corrected acceleration in body frame, and `[a']_×` is its
skew-symmetric matrix. The `∂p_prop/∂δθ` term arises because a rotation error `δθ` rotates the
specific force vector `R*a'`, producing a cross-product: `δ(R*a') = R*[a']_× * δθ` (using the
identity `δR*v = R*[v]_× * δθ` for small-angle rotation perturbations).

**Rotation error** (rows 3-5 of J):

```
∂θ_prop/∂δθ = I₃                                           (J[3:6, 3:6])
∂θ_prop/∂δbg = -I₃ * dt                                    (J[3:6, 15:18])
```

The rotation propagation `R_prop = R * Exp((ω - bg)*dt)` is perturbed by anchor rotation error
`δθ` (which passes through directly) and bias error `δbg` (which scales linearly with dt). The
right Jacobian `J_r((ω-bg)*dt)` is approximated as `I₃` for the small angles typical of
`||ω||*dt ≈ 0.005 rad`.

**Zero blocks**: Columns corresponding to `offset_R_L_I` (6-8), `offset_T_L_I` (9-11), and
`grav` (21-22) are set to zero. The LiDAR-IMU extrinsics do not affect the IMU-frame pose
propagation. The gravity term `∂p_prop/∂δg` requires the S(2) manifold Jacobian (`S2_Mx`);
after initialization, gravity covariance is negligible (the direction is well-estimated from the
static initialization phase), so this approximation introduces no practical error.

#### Propagated Covariance Computation

The 6x6 propagated pose covariance is:

```
P_pose(dt) = J(dt) * P_anchor * J(dt)^T + N(dt)
```

Where `P_anchor` is the full 23x23 EKF covariance from the last LiDAR correction, and `N(dt)` is
the cumulative IMU measurement noise contribution modeled consistently with the EKF's per-step
noise:

```
n_samples = round(dt / 0.005)              (number of 200Hz IMU samples in interval)
sample_dt = dt / n_samples                 (actual per-sample interval)

N_pos = acc_cov * sample_dt * dt³ / 3      (position noise, added to diagonal 0:3)
N_rot = n_samples * sample_dt² * gyr_cov   (rotation noise, added to diagonal 3:6)
```

This cumulative model sums `n_samples` independent per-step noise contributions rather than
computing a single-step noise over the full `dt`. The position noise accumulates as `O(dt³)` (from
double-integrating `n_samples` independent acceleration noise samples) and the rotation noise as
`O(dt)` (from single-integrating `n_samples` independent gyro noise samples).

The values `acc_cov` and `gyr_cov` are configured as `mapping.acc_cov` and `mapping.gyr_cov`.

#### Why This Matters for Flight Control

Without propagation, the controller sees constant uncertainty for ~100ms (between LiDAR corrections),
then a discontinuous jump. This is problematic for:

1. **Sensor fusion**: If the flight controller's internal EKF (e.g., PX4 EKF2) fuses external
   odometry with other sources (barometer, GPS, flow), it needs accurate covariance to weight the
   sources correctly. A frozen covariance overestimates confidence in the IMU-propagated estimate.

2. **Adaptive control**: Controllers that modulate gains based on state uncertainty (MPC, adaptive
   PID) benefit from knowing that uncertainty grows between corrections.

3. **Fault detection**: Monitoring covariance growth rate enables detection of degraded LiDAR
   conditions (few features, high motion blur) where the correction rate may be insufficient.

With propagation, the covariance correctly reflects the system behavior:

```
t = 0ms   (correction):   P small     (EKF-corrected, tight)
t = 5ms   (IMU sample):   P slightly larger
t = 50ms  (IMU sample):   P moderately larger
t = 95ms  (IMU sample):   P largest   (maximum open-loop uncertainty)
t = 100ms (correction):   P snaps back (EKF correction resets drift)
```

Position covariance grows as O(dt⁴) (dominated by the `I₃*dt` velocity Jacobian amplifying velocity
uncertainty: `P_vel * dt²`). Rotation covariance grows as O(dt²) (dominated by the `-I₃*dt` bias
Jacobian amplifying gyro bias uncertainty: `P_bg * dt²`).

#### Computational Cost

Per IMU sample: one 6x23 * 23x23 matrix multiply (3174 FMAs) + one 6x23 * 23x6 multiply (828 FMAs)
= ~4000 FLOPs at 200Hz = **0.8 MFLOPS**. Negligible compared to the EKF update (~23³ ≈ 12000 FLOPs
per iteration, 3 iterations, plus point matching).

#### Output Remapping and Frame Rotation

The upstream FAST-LIO code swaps position and rotation blocks when writing to the ROS covariance
array using `k = i < 3 ? i + 3 : i - 3`. This is a bug — the ROS `PoseWithCovariance` message
uses `[x, y, z, rot_x, rot_y, rot_z]` ordering, which matches our `P_pose` ordering
`[pos(0:2), rot(3:5)]` directly. We write `P_pose(i, j)` without remapping.

The IEKF uses right perturbation (`R_true = R_est * Exp(δθ)`), so the rotation error `δθ` is in
the body frame. The ROS convention expects `pose.covariance` in the `header.frame_id` (odom)
frame. We rotate the rotation block and cross terms to odom frame:

```
δθ_odom = R * δθ_body

P_odom = T * P_pose * T^T,   where T = diag(I₃, R)
```

In code:
```cpp
M3D R = anchor.rot;
P_pose.block<3,3>(3, 3) = R * P_pose.block<3,3>(3, 3) * R.transpose();
P_pose.block<3,3>(0, 3) = P_pose.block<3,3>(0, 3) * R.transpose();
P_pose.block<3,3>(3, 0) = R * P_pose.block<3,3>(3, 0);
```

The isotropic cumulative noise (`σ²I`) is invariant under rotation, so it can be added before
or after the frame rotation.

**Edge cases handled**:
- Before EKF initialization: `anchor.valid == false`, early return
- Stale data: `dt > 0.5s`, skip (EKF stuck or data gap)
- Negative dt: IMU timestamp before anchor, skip
- Quaternion normalization after rotation matrix conversion

**Files modified**:
- `src/laserMapping.cpp`:
  - Added `ImuPropState` struct and globals (after line 144)
  - Moved `imu_cbk` from free function into `LaserMappingNode` class to access publishers
  - After each EKF update, snapshots corrected state into `fwd_prop_anchor` (mutex-protected)
  - In `imu_cbk`, propagates from anchor and publishes odometry, TF, and path
  - Emptied `publish_odometry()` — IMU callback is the sole publisher of odom and TF
  - Increased IMU subscription queue from 10 to 200 to prevent message drops when scan
    processing blocks the single-threaded executor

### 2. LiDAR Scan Accumulator Node

**Problem**: The Livox Mid360 publishes point clouds at 100Hz. At this rate, the non-repetitive scanning
pattern produces wildly uneven point distributions per 10ms window (observed range: 48 to 8790 points
per scan), causing frequent "No Effective Points" EKF failures when a scan has insufficient points for
the iterated closest-point matching.

The LiDAR driver rate cannot be changed when playing from a rosbag.

**Solution**: Added a standalone `lidar_accumulator` ROS2 node that subscribes to 100Hz
`livox_ros_driver2/msg/CustomMsg` messages, accumulates N consecutive scans (default: 10), and
publishes the combined cloud at 10Hz.

**Timestamp handling**: The `CustomMsg` format stores a `timebase` (uint64, nanoseconds) per message
and a per-point `offset_time` (uint32, nanoseconds relative to timebase). When accumulating:

```
For message k (k = 0..N-1):
    time_offset_ns = timebase_k - timebase_0
    For each point p in message k:
        p.offset_time += time_offset_ns
```

This preserves the per-point timing needed for FAST-LIO's motion undistortion (`UndistortPcl()`),
which uses point timestamps to compensate for ego-motion during the scan.

**Files added**:
- `src/lidar_accumulator.cpp`: Standalone ROS2 node with configurable parameters

**Files modified**:
- `CMakeLists.txt`: Added build target for `lidar_accumulator` with `Python3::Python` linkage
  (required for transitive dependency resolution on macOS)
- `config/mid360.yaml`: Changed `lid_topic` to `/livox/lidar_accumulated`
- `launch/mapping_custom.launch.py`: Launches `lidar_accumulator` before `fastlio_mapping`

### 3. Velocity Publishing in Odometry

**Problem**: Upstream FAST-LIO never fills the `twist` field in the odometry message. The flight
controller needs both linear and angular velocity.

**Solution**: Added velocity to the odometry output.

**Linear velocity**: The EKF state `vel` is estimated in world frame. The odometry `twist` field
requires velocity in the child frame (body/IMU frame). The transform uses the rotation quaternion
conjugate (equivalent to transpose for rotation matrices):

```
v_world = state_point.vel                    (EKF estimated, world frame)
R_bw = state_point.rot                       (body-to-world rotation, SO(3))

v_body = R_bw^{-1} * v_world = R_bw^T * v_world = conj(q_bw) * v_world
```

In Eigen: `state_point.rot.conjugate() * state_point.vel`

**Angular velocity**: The gyroscope measures angular velocity in body frame. The EKF estimates
gyroscope bias `bg`. The corrected angular velocity is:

```
omega_body = omega_measured - bg
```

With a guard against empty IMU buffer (which caused a segfault on Jetson when `Measures.imu.back()`
was called on an empty deque).

**Files modified**:
- `src/laserMapping.cpp`: `publish_odometry()` function and IMU forward propagation callback

### 4. Frame Renaming (REP 105 Compliance)

**Problem**: FAST-LIO uses non-standard frame names (`camera_init`, `body`) that don't follow
[REP 105](https://www.ros.org/reps/rep-0105.html) conventions.

**Solution**: Renamed all frame references:
- `camera_init` -> `odom` (the fixed world frame for odometry)
- `body` -> `imu_link` (the IMU sensor frame, rigidly attached to the LiDAR)

The `odom` frame is the origin of FAST-LIO's odometry estimate. It corresponds to the IMU position at
initialization time. The `imu_link` frame moves with the sensor.

**Files modified**:
- `src/laserMapping.cpp`: All TF broadcasts, odometry messages, point cloud headers, and path messages

### 5. Topic Renaming

**Problem**: FAST-LIO publishes odometry on `/Odometry`, a non-standard topic name.

**Solution**: Renamed to `/odom` to follow ROS conventions.

**Files modified**:
- `src/laserMapping.cpp`: Publisher topic name

### 6. Custom Launch File with Static Transform

**Problem**: Need a TF from `imu_link` (Livox IMU) to `base_link` (Pixhawk) for the flight controller.

**Solution**: Created `mapping_custom.launch.py` with:
- Static transform publisher: `imu_link -> base_link` with translation (0, 0, -0.12m)
- LiDAR accumulator node
- All original launch arguments preserved

The transform is pure translation (no rotation) because the Pixhawk is mounted directly below the
LiDAR with aligned axes. The -0.12m Z offset means base_link is 12cm below imu_link.

**Files added**:
- `launch/mapping_custom.launch.py`

### 7. IMU Acceleration Unit Fix

**Problem**: The Livox Mid360 driver publishes IMU acceleration in units of g (~1.0 at rest), but
FAST-LIO expects m/s^2 (~9.81 at rest).

This caused a critical issue in `UndistortPcl()` (line 260 of `IMU_Processing.hpp`):

```
acc_avr = acc_avr * G_m_s2 / mean_acc.norm()
```

This rescaling is meant to normalize sensor units. When the sensor reports in g's,
`mean_acc.norm() ~ 1.0`, so the rescaling becomes `acc * 9.81 / 1.0 = acc * 9.81`, which amplifies
accelerations by ~9.81x and causes immediate state divergence.

When the sensor correctly reports in m/s^2, `mean_acc.norm() ~ 9.81`, so the rescaling becomes
`acc * 9.81 / 9.81 = acc * 1.0`, a no-op as intended.

**Solution**: Added `*= G_m_s2` (9.81) scaling in `imu_cbk` to convert from g's to m/s^2 before
buffering. After this fix, `mean_acc.norm() ~ 9.81` and the rescaling in `UndistortPcl()` becomes a
harmless identity operation.

```
a_scaled = a_raw * 9.81        (g -> m/s^2)
```

**Files modified**:
- `src/laserMapping.cpp`: `imu_cbk()` function

### 8. IMU Initialization Window

**Problem**: FAST-LIO initializes by accumulating `MAX_INI_COUNT` IMU samples to estimate the initial
gravity direction and sensor biases. At 200Hz IMU rate, the original `MAX_INI_COUNT = 10` only
accumulated 50ms of data, leading to unreliable gravity estimation and initial state divergence.

The initialization computes:

```
mean_acc = (1/N) * sum(a_i)                             (gravity direction estimate)
mean_gyr = (1/N) * sum(omega_i)                         (gyro bias estimate)
```

A short window makes these estimates noisy, especially in the presence of vibration.

**Solution**: Increased `MAX_INI_COUNT` to 100 (0.5 seconds at 200Hz), providing a much more stable
gravity and bias estimate.

**Files modified**:
- `src/IMU_Processing.hpp`: `MAX_INI_COUNT` constant

### 9. Point Cloud Filter Parameters

**Problem**: With the Mid360's non-repetitive pattern, aggressive downsampling left too few points for
the iterative closest-point matching in the EKF update step.

FAST-LIO applies two levels of point reduction:
1. `point_filter_num`: Keep every Nth point from the raw scan
2. `filter_size_surf`: Voxel grid downsampling with this voxel edge length (meters)

**Solution**: Adjusted parameters in `config/mid360.yaml`:
- `point_filter_num`: 3 -> 1 (keep all raw points)
- `filter_size_surf`: 0.5 -> 0.15 (finer 15cm voxel grid, retaining more spatial detail)
- `scan_rate`: 100 -> 10 (matches the 10Hz accumulated scan rate)

### 10. Parameter Logging at Startup

Added `RCLCPP_INFO` calls to print all FAST-LIO parameters on node startup for easier debugging and
configuration verification.

**Files modified**:
- `src/laserMapping.cpp`: After parameter loading in `LaserMappingNode` constructor

### 11. Mocap Ground Truth Converter Node

**Problem**: For trajectory evaluation with `evo`, we need ground truth from a motion capture system
published as standard ROS odometry and path messages.

**Solution**: Added a standalone `mocap_converter` node that subscribes to
`mocap4r2_msgs/msg/RigidBodies` and publishes:
- `nav_msgs/Odometry` on `/ground_truth/odom`
- `nav_msgs/Path` on `/ground_truth/path` (accumulated poses)

The node matches rigid bodies by name (not array index) via the `rigid_body_name` parameter. This
avoids confusion between the mocap system's internal ID and the array position.

**Parameters**:

| Parameter          | Type   | Default               | Description                   |
|--------------------|--------|-----------------------|-------------------------------|
| `rigid_body_name`  | string | `""`                  | Name to match (empty = first) |
| `mocap_topic`      | string | `/mocap/rigid_bodies`  | Input topic                   |
| `odom_frame`       | string | `map`                 | Parent frame for ground truth |

**Files added**:
- `src/mocap_converter.cpp`

**Files modified**:
- `CMakeLists.txt`: Added `mocap4r2_msgs` dependency and `mocap_converter` build target
- `launch/mapping_custom.launch.py`: Added `mocap_converter` node with `rigid_body_name` launch
  argument (default `"91"`, forced to string type via `ParameterValue` to prevent YAML integer
  parsing)

### 12. Minor Changes

- **C++17**: Changed from C++14 to C++17 (`CMakeLists.txt`)
- **PCD saving**: Disabled by default (`config/mid360.yaml`)
- **RViz config**: Updated frame names, topic names, and display settings (`rviz/fastlio.rviz`)
- **Empty IMU guard**: Added check for empty `Measures.imu` deque before calling `.back()` to
  prevent segfault (`src/laserMapping.cpp`)

# Real-Time Engineering for Flight: Plan

Plan for the engineering contributions that make the LiDAR-inertial odometry system
suitable for onboard flight on multirotor platforms. These are the contributions
covered in Chapter 5 of the thesis.

## Motivation

A LiDAR-inertial odometry algorithm that achieves high accuracy on a benchmark dataset
is not necessarily suitable for closed-loop flight control. Flight controllers impose
specific requirements that benchmarks rarely test:

1. Pose feedback at high rate (hundreds of Hz) for stable control loops
2. Velocity estimates at the same rate for damping and feedforward control
3. Low-noise signals that don't excite control oscillations
4. No interruptions, even during transient sensor degradation
5. Bounded computational resources for long-duration missions

This chapter addresses each of these requirements through specific modifications to
the FAST-LIO2 base system, without altering the core estimation algorithm.

## Section 1: IMU-Rate Odometry Publishing

### Goal
Provide pose, velocity, and orientation estimates at the IMU rate (200 Hz on the
target platform) instead of the LiDAR rate (10 Hz from accumulated scans).

### Why this matters
At 2 m/s flight speed, a 10 Hz update rate means the controller receives a new pose
every 0.2 m. At 200 Hz, the gap shrinks to 0.01 m. The flight controller no longer
needs to dead-reckon between updates, which reduces latency and improves stability,
especially during aggressive maneuvers.

### Approach
The IEKF correction can only happen when a new accumulated LiDAR scan arrives (10 Hz).
Between corrections, the state is propagated forward using IMU integration. Instead
of publishing only at correction events, we publish the propagated state at every
IMU sample.

Forward propagation between corrections:
- p(t) = p_anchor + v_anchor * dt + 0.5 * a_world * dt^2
- v(t) = v_anchor + a_world * dt
- R(t) = R_anchor * Exp(omega_body * dt)

Where (p_anchor, v_anchor, R_anchor) is the state at the last EKF correction, and
a_world = R_anchor * (a_m - b_a) + g, omega_body = omega_m - b_g.

### Velocity computation
Linear velocity in body frame:
v_body = R^T * v_world

Angular velocity in body frame: directly from the bias-corrected gyroscope:
omega_body = omega_m - b_g

These are published in the odometry message twist field at IMU rate.

## Section 2: Real-Time Architecture

### Goal
Sustain constant 200 Hz odometry under all conditions, regardless of EKF processing
load.

### Problem
A single-threaded executor running both the IMU callback and the IEKF update will
inevitably drop IMU messages when the EKF processing exceeds the IMU period. The
result is a gradual or sudden degradation of odometry rate from 200 Hz down to 168 Hz
or lower.

### Solution: Executor Isolation
Use two separate single-threaded ROS 2 executors:
- IMU executor: runs only the IMU callback. Dedicated thread, never blocked.
- Main executor: runs the EKF timer callback, LiDAR callback, and publishers.

The IMU callback is registered to its own MutuallyExclusive callback group, which is
added to the IMU executor only. The default callback group goes to the main executor.

### Mutex Contention
Even with separate executors, the IMU thread can be blocked if it tries to acquire
a mutex held by the main thread. The main bottleneck is the LiDAR preprocessing,
which is computationally heavy and was originally inside the buffer mutex.

Solution: move preprocessing outside the mutex. The mutex is only held for the
buffer push operation, which is a few microseconds.

### Path Serialization
Publishing the nav_msgs::Path message at 200 Hz becomes a serialization bottleneck
as the path grows. After 30 seconds at 2 m/s, the path contains 6000 poses, and
serializing this message at every IMU callback consumes significant CPU.

Solution: throttle the path publication to 10 Hz. The path is for visualization
only, not for control.

### Result
With these three changes (executor isolation, mutex relocation, path throttling),
the odometry rate stays constant at 200 Hz indefinitely, with no dropouts even
under sustained load.

## Section 3: Velocity and Orientation Filtering

### Goal
Provide smooth velocity and orientation signals to the flight controller without
introducing excessive lag.

### Problem
The forward-propagated velocity is computed by integrating the accelerometer over
the time since the last EKF correction. The accelerometer is noisy, and this noise
appears directly in the velocity output. Plotting the velocity shows high-frequency
jitter on the order of 0.2 m/s, which would excite oscillations in a tightly tuned
velocity controller.

The orientation suffers from a similar issue: the gyroscope noise is integrated,
producing a noisy quaternion output. Yaw is particularly affected.

### Solution: First-Order Low-Pass Filter
Apply an exponential moving average filter to the published velocity, angular
velocity, and orientation:

For linear/angular velocity (vector):
v_filtered = alpha * v_raw + (1 - alpha) * v_filtered_prev

For orientation (quaternion): use SLERP between the previous filtered quaternion
and the current raw quaternion:
q_filtered = q_filtered_prev.slerp(alpha, q_raw)

The alpha parameter controls the trade-off between noise reduction and lag:
- alpha = 1.0: no filtering, raw output
- alpha = 0.05: heavy smoothing, 1.6 Hz cutoff at 200 Hz update rate
- alpha = 0.1: moderate smoothing, 3.2 Hz cutoff

### Why filter the body-frame velocity (not world-frame)
Filtering must happen after the rotation from world to body frame, not before. If
we filter v_world and then rotate by the current R, the rotation matrix injects
gyroscope noise back into the filtered output. By filtering v_body directly, the
rotation is applied once and the filter handles all noise sources together.

### Quaternion filter via SLERP
Component-wise quaternion filtering followed by normalization is an approximation
that breaks for large angular changes. SLERP (spherical linear interpolation)
handles the manifold structure correctly. At 200 Hz, the angular change between
samples is small enough that both methods give similar results, but SLERP is the
mathematically correct choice and Eigen provides it natively.

### Tuning
For drone applications with maximum 2 m/s flight speed, alpha = 0.05 provides:
- Velocity RMSE reduction by an order of magnitude
- Lag of approximately 0.2 s (acceptable for the control bandwidth)
- No visible oscillation in the controller response

## Section 4: Stale Anchor Handling

### Goal
Maintain odometry output even when LiDAR scans are temporarily unavailable.

### Problem
The forward propagation depends on the last EKF correction (the "anchor") to set
the integration starting point. If LiDAR scans stop arriving, the anchor timestamp
freezes. The original implementation had a hard guard: if the time since the last
anchor exceeded 0.5 s, the IMU callback returned without publishing odometry. This
caused silent odometry death during transient LiDAR interruptions.

### Solution
Replace the hard return with a warn-and-continue. The forward propagation continues
to integrate IMU measurements from the stale anchor, producing increasingly drifted
odometry but never going silent. A warning is logged once per 2 seconds to indicate
the anchor is stale.

### Trade-off
Drifting odometry is dangerous, but no odometry is more dangerous for a flight
controller. The drift accumulates only during the gap (typically less than 1-2 s for
transient issues) and is corrected as soon as a new LiDAR scan arrives. For longer
outages, the controller can use other criteria (failsafe, hover) but at least the
odometry stream remains alive.

## Section 5: Memory Bounding for Long Missions

### Goal
Prevent memory growth and processing time degradation during long-duration flights.

### Problem
The ikd-Tree map grows unboundedly as new scans are added. Over a 10-minute flight,
memory grew from 1.3 GB to 5.2 GB and per-scan processing time increased proportionally.
This eventually causes the EKF to fall behind real-time, dropping odometry rate.

### Solution: Bounded Local Map
The ikd-Tree already supports a local map cube — points outside the cube are pruned
as the drone moves. The default cube_side_length of 1000 m is effectively unbounded
for indoor flight. Reducing it to 100 m (50 m radius from the current position)
limits memory and processing time without affecting accuracy.

For indoor flight, all relevant features are within 30 m, so 50 m radius is
generous. The map is pruned automatically as the drone moves, keeping the tree size
roughly constant.

### Other memory sources
Additional contributions to long-term memory growth:
- Path message: throttled and could be capped to last N poses
- Debug log files: continuously appended, but disk-bound rather than RAM-bound
- Pre-allocated debug arrays: 55 MB allocated even when logging is disabled

These are smaller contributors and are addressed individually.

## Section 6: QoS and Queue Tuning

### Goal
Match ROS 2 QoS settings to the timing characteristics of each topic.

### IMU subscription
- Queue depth reduced from 200 to 10. Holding 200 IMU messages (1 second of data)
  serves no purpose: if the callback falls behind, dropping old messages is
  preferable to processing stale data.

### Odometry publication
- Use SensorDataQoS (best-effort, volatile) instead of the default reliable QoS.
- Best-effort is appropriate for high-rate sensor data: missing one message is
  acceptable, but waiting for retransmission introduces latency.
- Volatile means new subscribers do not receive past messages, reducing serialization
  overhead.

## Section 7: Experimental Validation

### Goal
Validate each engineering contribution with quantitative measurements.

### Experiments
- Odometry rate stability under load (Experiment 1 in thesis plan)
- Accuracy preservation vs upstream FAST-LIO2 (Experiment 2)
- Velocity filter trade-off curve (Experiment 3)
- Memory and computational performance (Experiment 4)
- Stale anchor recovery (Experiment 5)

These experiments are described in detail in MASTER_THESIS.md.

## References

- ROS 2 design document: Executors and callback groups.
- PX4 documentation: Position estimator requirements.
- FAST-LIO2: Xu, W., Cai, Y., He, D., Lin, J., Zhang, F. (2022). IEEE T-RO.
- Solà, J. (2017). Quaternion kinematics for the error-state Kalman filter. arXiv.

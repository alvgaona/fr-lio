"""Stage 2: Trajectory + IMU simulator.

Defines a parametric ground-truth trajectory in 3D, computes the analytical
derivatives to generate clean IMU measurements, adds realistic bias and noise,
and writes the result to a NumPy file for later use by an EKF.

Also generates LiDAR scans from the trajectory poses using the environment
defined in sim_environment_3d.py.

Run: python sim_imu_trajectory.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sim_environment_3d import (
    PLANES, OBSTACLES, ROOM_X, ROOM_Y, ROOM_Z, ENVIRONMENT,
    cast_ray, mid360_rosette_directions,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(7)

DURATION = 50.0 if ENVIRONMENT == "long_corridor" else 20.0
IMU_RATE = 200.0
LIDAR_RATE = 10.0
# Real Livox Mid-360 accumulated scan has ~20000 raw points at 10Hz.
# FAST-LIO downsamples with filter_size_surf=0.15 m, leaving a few hundred
# effective points. We use 5000 here as a middle ground that mimics the
# effective information per scan after preprocessing.
N_RAYS_PER_SCAN = 5000
RANGE_NOISE_STD = 0.02
MAX_RANGE = 30.0

GRAVITY = np.array([0.0, 0.0, -9.81])

# Match FAST-LIO config (config/fast_lio.yaml):
#   acc_cov = gyr_cov = 0.1 are continuous-time spectral densities (variance per second)
#   b_acc_cov = b_gyr_cov = 0.0001 are bias random-walk spectral densities
#
# Discrete-time std per IMU sample: sigma_d = sqrt(cov_cont * dt_imu)
# dt_imu = 1/200 = 0.005
# gyr: sqrt(0.1 * 0.005) ~ 0.0224 rad/s  (~1.3 deg/s) — realistic for MEMS IMU
# acc: sqrt(0.1 * 0.005) ~ 0.0224 m/s^2 — realistic for MEMS IMU
DT_IMU_SIM = 1.0 / IMU_RATE
GYR_NOISE_STD = np.sqrt(0.1 * DT_IMU_SIM)
ACC_NOISE_STD = np.sqrt(0.1 * DT_IMU_SIM)
GYR_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_SIM)
ACC_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_SIM)

INITIAL_GYR_BIAS = np.array([0.001, -0.002, 0.0015])
INITIAL_ACC_BIAS = np.array([0.02, -0.01, 0.015])

if ENVIRONMENT == "corridor":
    TRAJ_CENTER = np.array([15.0, 1.5, 1.5])
    TRAJ_X_AMP = 10.0
    TRAJ_X_FREQ = 2 * np.pi / 20.0
    TRAJ_Z_AMP = 0.2
    TRAJ_Z_FREQ = 2 * np.pi / 7.0
elif ENVIRONMENT == "wall":
    # Hover and drift slowly parallel to the wall (y axis). The wall is at
    # x=5 and the sensor is at x=2. Small vertical oscillation and slow
    # lateral sweep to exercise the scan-to-scan registration while keeping
    # the wall in view.
    TRAJ_CENTER = np.array([2.0, 25.0, 25.0])
    TRAJ_Y_AMP = 1.5
    TRAJ_Y_FREQ = 2 * np.pi / 15.0
    TRAJ_Z_AMP = 0.3
    TRAJ_Z_FREQ = 2 * np.pi / 9.0
elif ENVIRONMENT == "hover":
    # Stationary hover inside the cube room. Sanity-check the scan-to-scan
    # CRLB: with no relative motion, P_drift should stay near zero because
    # every scan is geometrically identical to the previous one.
    TRAJ_CENTER = np.array([5.0, 5.0, 5.0])
elif ENVIRONMENT == "room_corridor":
    # Piecewise trajectory: hover in the room, transition into the
    # corridor, traverse to the end, and come back. This demonstrates
    # environment-adaptive P_drift growth: slow in the room, faster in
    # the corridor where yaw is weakly observable.
    TRAJ_CENTER = np.array([5.0, 5.0, 1.5])
elif ENVIRONMENT == "long_corridor":
    # Straight traversal of a long corridor. Tests long-distance
    # drift behavior. The drone flies at a constant velocity from
    # near the start to ~100 m along the corridor.
    TRAJ_CENTER = np.array([5.0, 1.5, 1.5])
    LONG_CORRIDOR_SPEED = 2.0
    TRAJ_Z_AMP = 0.1
    TRAJ_Z_FREQ = 2 * np.pi / 7.0
else:
    TRAJ_CENTER = np.array([5.0, 5.0, 1.5])
    TRAJ_RADIUS = 2.5
    TRAJ_OMEGA = 2 * np.pi / 10.0
    TRAJ_Z_AMP = 0.3
    TRAJ_Z_FREQ = 2 * np.pi / 7.0


def trajectory_at(t):
    """Return ground-truth (R, p, v, omega_world, a_world) at time t.

    Cube room: horizontal circle with a small vertical sinusoid, yaw aligned
    to the tangent of the motion.

    Corridor: back-and-forth along the corridor axis with a small vertical
    sinusoid. Yaw stays constant (facing forward along x).

    Hover: completely stationary. Used to verify that the scan-to-scan CRLB
    drift stays at zero when there is no relative motion.
    """
    cx, cy, cz = TRAJ_CENTER

    if ENVIRONMENT == "hover":
        p = np.array([cx, cy, cz])
        v = np.zeros(3)
        a_world = np.zeros(3)
        R = np.eye(3)
        omega_world = np.zeros(3)
        return R, p, v, omega_world, a_world

    if ENVIRONMENT == "room_corridor":
        # Smooth piecewise trajectory over DURATION (20s):
        #   0 - 4 s: small loop inside the room (feature-rich)
        #   4 - 8 s: fly from room into corridor entrance
        #   8 - 14 s: traverse corridor to the far end
        #   14 - 20 s: return back to room
        # Uses a single smooth parametric path so the derivatives are
        # continuous.
        room_cx, room_cy, room_cz = 5.0, 5.0, 1.5
        cor_entry_x, cor_entry_y = 10.5, 5.0
        cor_end_x = 28.0
        phase_period = 20.0
        tau = (t % phase_period) / phase_period  # 0 to 1
        omega_phase = 2 * np.pi / phase_period

        # Use a smooth parametric motion: x progresses along a sinusoid
        # that returns to start, while y stays near the corridor midline.
        # This gives a single smooth oscillation from room to corridor
        # end and back, perfectly periodic.
        x_amp = (cor_end_x - room_cx) / 2
        x_mid = (cor_end_x + room_cx) / 2
        x = x_mid - x_amp * np.cos(omega_phase * t)
        vx = x_amp * omega_phase * np.sin(omega_phase * t)
        ax = x_amp * omega_phase * omega_phase * np.cos(omega_phase * t)

        # y gently drifts into the corridor midline as x increases past
        # the doorway, then back to the room center
        y_amp = (cor_entry_y - room_cy)  # typically 0 since same height
        y_modulation = (x - room_cx) / (cor_end_x - room_cx)
        y = room_cy
        vy = 0.0
        ay = 0.0

        # Small vertical oscillation
        z_amp = 0.2
        z_freq = 2 * np.pi / 7.0
        z = room_cz + z_amp * np.sin(z_freq * t)
        vz = z_amp * z_freq * np.cos(z_freq * t)
        az = -z_amp * z_freq * z_freq * np.sin(z_freq * t)

        p = np.array([x, y, z])
        v = np.array([vx, vy, vz])
        a_world = np.array([ax, ay, az])
        R = np.eye(3)
        omega_world = np.zeros(3)
        return R, p, v, omega_world, a_world

    if ENVIRONMENT == "corridor":
        wx = TRAJ_X_FREQ
        p = np.array([
            cx + TRAJ_X_AMP * np.sin(wx * t),
            cy,
            cz + TRAJ_Z_AMP * np.sin(TRAJ_Z_FREQ * t),
        ])
        v = np.array([
            TRAJ_X_AMP * wx * np.cos(wx * t),
            0.0,
            TRAJ_Z_AMP * TRAJ_Z_FREQ * np.cos(TRAJ_Z_FREQ * t),
        ])
        a_world = np.array([
            -TRAJ_X_AMP * wx * wx * np.sin(wx * t),
            0.0,
            -TRAJ_Z_AMP * TRAJ_Z_FREQ * TRAJ_Z_FREQ * np.sin(TRAJ_Z_FREQ * t),
        ])
        R = np.eye(3)
        omega_world = np.array([0.0, 0.0, 0.0])
        return R, p, v, omega_world, a_world

    if ENVIRONMENT == "long_corridor":
        # Straight-line traversal at constant velocity.
        p = np.array([
            cx + LONG_CORRIDOR_SPEED * t,
            cy,
            cz + TRAJ_Z_AMP * np.sin(TRAJ_Z_FREQ * t),
        ])
        v = np.array([
            LONG_CORRIDOR_SPEED,
            0.0,
            TRAJ_Z_AMP * TRAJ_Z_FREQ * np.cos(TRAJ_Z_FREQ * t),
        ])
        a_world = np.array([
            0.0,
            0.0,
            -TRAJ_Z_AMP * TRAJ_Z_FREQ * TRAJ_Z_FREQ * np.sin(TRAJ_Z_FREQ * t),
        ])
        R = np.eye(3)
        omega_world = np.zeros(3)
        return R, p, v, omega_world, a_world

    if ENVIRONMENT == "wall":
        wy = TRAJ_Y_FREQ
        p = np.array([
            cx,
            cy + TRAJ_Y_AMP * np.sin(wy * t),
            cz + TRAJ_Z_AMP * np.sin(TRAJ_Z_FREQ * t),
        ])
        v = np.array([
            0.0,
            TRAJ_Y_AMP * wy * np.cos(wy * t),
            TRAJ_Z_AMP * TRAJ_Z_FREQ * np.cos(TRAJ_Z_FREQ * t),
        ])
        a_world = np.array([
            0.0,
            -TRAJ_Y_AMP * wy * wy * np.sin(wy * t),
            -TRAJ_Z_AMP * TRAJ_Z_FREQ * TRAJ_Z_FREQ * np.sin(TRAJ_Z_FREQ * t),
        ])
        R = np.eye(3)
        omega_world = np.array([0.0, 0.0, 0.0])
        return R, p, v, omega_world, a_world

    w = TRAJ_OMEGA
    p = np.array([
        cx + TRAJ_RADIUS * np.cos(w * t),
        cy + TRAJ_RADIUS * np.sin(w * t),
        cz + TRAJ_Z_AMP * np.sin(TRAJ_Z_FREQ * t),
    ])
    v = np.array([
        -TRAJ_RADIUS * w * np.sin(w * t),
        TRAJ_RADIUS * w * np.cos(w * t),
        TRAJ_Z_AMP * TRAJ_Z_FREQ * np.cos(TRAJ_Z_FREQ * t),
    ])
    a_world = np.array([
        -TRAJ_RADIUS * w * w * np.cos(w * t),
        -TRAJ_RADIUS * w * w * np.sin(w * t),
        -TRAJ_Z_AMP * TRAJ_Z_FREQ * TRAJ_Z_FREQ * np.sin(TRAJ_Z_FREQ * t),
    ])
    yaw = w * t + np.pi / 2
    cy_, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy_, -sy, 0.0],
        [sy, cy_, 0.0],
        [0.0, 0.0, 1.0],
    ])
    omega_world = np.array([0.0, 0.0, w])
    return R, p, v, omega_world, a_world


def simulate_imu(t, prev_gyr_bias, prev_acc_bias):
    """Simulate one IMU sample at time t.

    Returns (gyro_meas, acc_meas, new_gyr_bias, new_acc_bias).
    """
    R, _, _, omega_world, a_world = trajectory_at(t)
    omega_body = R.T @ omega_world
    a_body_specific = R.T @ (a_world - GRAVITY)

    gyr_bias = prev_gyr_bias + np.random.normal(0, GYR_BIAS_WALK_STD, 3)
    acc_bias = prev_acc_bias + np.random.normal(0, ACC_BIAS_WALK_STD, 3)

    gyro_meas = omega_body + gyr_bias + np.random.normal(0, GYR_NOISE_STD, 3)
    acc_meas = a_body_specific + acc_bias + np.random.normal(0, ACC_NOISE_STD, 3)

    return gyro_meas, acc_meas, gyr_bias, acc_bias


def simulate_lidar_scan(R, p, n_rays):
    """Simulate one LiDAR scan from pose (R, p). Returns body-frame points."""
    body_directions = mid360_rosette_directions(0, n_rays)
    body_points = []
    for d_body in body_directions:
        d_world = R @ d_body
        r = cast_ray(p, d_world)
        if r < MAX_RANGE:
            r_noisy = r + np.random.normal(0, RANGE_NOISE_STD)
            body_points.append(r_noisy * d_body)
    return np.array(body_points)


def run_simulation():
    dt_imu = 1.0 / IMU_RATE
    dt_lidar = 1.0 / LIDAR_RATE
    n_imu = int(DURATION * IMU_RATE)

    imu_data = []
    gyr_bias = INITIAL_GYR_BIAS.copy()
    acc_bias = INITIAL_ACC_BIAS.copy()

    for i in range(n_imu):
        t = i * dt_imu
        gyro, acc, gyr_bias, acc_bias = simulate_imu(t, gyr_bias, acc_bias)
        imu_data.append({"t": t, "gyro": gyro, "acc": acc,
                         "gyr_bias": gyr_bias.copy(), "acc_bias": acc_bias.copy()})

    lidar_data = []
    n_lidar = int(DURATION * LIDAR_RATE)
    for i in range(n_lidar):
        t = i * dt_lidar
        R, p, _, _, _ = trajectory_at(t)
        scan = simulate_lidar_scan(R, p, N_RAYS_PER_SCAN)
        lidar_data.append({"t": t, "R": R, "p": p, "points_body": scan})

    return imu_data, lidar_data


def plot_results(imu_data, lidar_data):
    t_imu = np.array([d["t"] for d in imu_data])
    gyro = np.array([d["gyro"] for d in imu_data])
    acc = np.array([d["acc"] for d in imu_data])

    t_lidar = np.array([d["t"] for d in lidar_data])
    p_lidar = np.array([d["p"] for d in lidar_data])

    accumulated_world = []
    for d in lidar_data:
        if len(d["points_body"]) == 0:
            continue
        world_pts = (d["R"] @ d["points_body"].T).T + d["p"]
        accumulated_world.append(world_pts)
    accumulated_world = np.vstack(accumulated_world)

    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(t_imu, gyro[:, 0], label="x", linewidth=0.8)
    ax1.plot(t_imu, gyro[:, 1], label="y", linewidth=0.8)
    ax1.plot(t_imu, gyro[:, 2], label="z", linewidth=0.8)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("gyro (rad/s)")
    ax1.set_title("Gyroscope (body frame)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_imu, acc[:, 0], label="x", linewidth=0.8)
    ax2.plot(t_imu, acc[:, 1], label="y", linewidth=0.8)
    ax2.plot(t_imu, acc[:, 2], label="z", linewidth=0.8)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("acc (m/s^2)")
    ax2.set_title("Accelerometer (body frame, with gravity)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(p_lidar[:, 0], p_lidar[:, 1], "b-", linewidth=1.2)
    ax3.plot([0, ROOM_X, ROOM_X, 0, 0], [0, 0, ROOM_Y, ROOM_Y, 0], color="#888888", linewidth=1.2)
    for obs in OBSTACLES:
        cx, cy, _ = obs["center"]
        sx, sy, _ = obs["size"]
        ax3.plot(
            [cx - sx / 2, cx + sx / 2, cx + sx / 2, cx - sx / 2, cx - sx / 2],
            [cy - sy / 2, cy - sy / 2, cy + sy / 2, cy + sy / 2, cy - sy / 2],
            color="#cc6600", linewidth=1.0,
        )
    ax3.set_xlim(-0.5, ROOM_X + 0.5)
    ax3.set_ylim(-0.5, ROOM_Y + 0.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title("Trajectory (top-down)")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax4.plot(p_lidar[:, 0], p_lidar[:, 1], p_lidar[:, 2], "b-", linewidth=1.5)
    ax4.set_xlim(0, ROOM_X)
    ax4.set_ylim(0, ROOM_Y)
    ax4.set_zlim(0, ROOM_Z)
    ax4.set_xlabel("x (m)")
    ax4.set_ylabel("y (m)")
    ax4.set_zlabel("z (m)")
    ax4.set_title("Trajectory (3D)")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(accumulated_world[:, 0], accumulated_world[:, 1], s=0.3, c="black", alpha=0.4)
    ax5.plot(p_lidar[:, 0], p_lidar[:, 1], "r-", linewidth=1.0, label="trajectory")
    ax5.plot([0, ROOM_X, ROOM_X, 0, 0], [0, 0, ROOM_Y, ROOM_Y, 0], color="#888888", linewidth=1.2)
    ax5.set_xlim(-0.5, ROOM_X + 0.5)
    ax5.set_ylim(-0.5, ROOM_Y + 0.5)
    ax5.set_aspect("equal")
    ax5.set_xlabel("x (m)")
    ax5.set_ylabel("y (m)")
    ax5.set_title("Accumulated point cloud (top-down)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    gyr_bias_arr = np.array([d["gyr_bias"] for d in imu_data])
    acc_bias_arr = np.array([d["acc_bias"] for d in imu_data])
    ax6.plot(t_imu, gyr_bias_arr[:, 0], label="gyr bias x", linewidth=0.8)
    ax6.plot(t_imu, gyr_bias_arr[:, 1], label="gyr bias y", linewidth=0.8)
    ax6.plot(t_imu, gyr_bias_arr[:, 2], label="gyr bias z", linewidth=0.8)
    ax6.set_xlabel("t (s)")
    ax6.set_ylabel("bias (rad/s)")
    ax6.set_title("Gyro bias drift (random walk)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_imu_trajectory.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_imu_trajectory.png")


def save_data(imu_data, lidar_data):
    """Save the simulated data to a numpy file for use by an EKF."""
    imu_arr = np.array([
        (d["t"], *d["gyro"], *d["acc"], *d["gyr_bias"], *d["acc_bias"])
        for d in imu_data
    ])

    np.save(f"{OUT_DIR}/sim_imu.npy", imu_arr)
    print(f"Saved {OUT_DIR}/sim_imu.npy ({len(imu_arr)} samples)")

    lidar_save = []
    for d in lidar_data:
        lidar_save.append({
            "t": d["t"],
            "R": d["R"],
            "p": d["p"],
            "points_body": d["points_body"],
        })
    np.save(f"{OUT_DIR}/sim_lidar.npy", np.array(lidar_save, dtype=object), allow_pickle=True)
    print(f"Saved {OUT_DIR}/sim_lidar.npy ({len(lidar_save)} scans)")


print("Running trajectory + IMU simulation...")
imu_data, lidar_data = run_simulation()
print(f"Generated {len(imu_data)} IMU samples and {len(lidar_data)} LiDAR scans")
plot_results(imu_data, lidar_data)
save_data(imu_data, lidar_data)

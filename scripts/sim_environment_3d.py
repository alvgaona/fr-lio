"""Stage 1: 3D environment + synthetic Mid-360-like LiDAR.

Defines a 3D room as a set of planes, ray-casts from a fixed sensor pose, and
generates a non-repetitive scanning pattern that approximates the Livox Mid-360
rosette.

Run: python sim_environment_3d.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

import os
ENVIRONMENT = os.environ.get("SIM_ENV", "cube")

if ENVIRONMENT == "corridor":
    ROOM_X = 30.0
    ROOM_Y = 3.0
    ROOM_Z = 3.0
    SENSOR_POS = np.array([5.0, 1.5, 1.5])
elif ENVIRONMENT == "long_corridor":
    # 1000 m long, 3 m wide, 3 m tall corridor. Tests long-distance
    # drift behavior and whether the scan-to-scan CRLB bound tracks the
    # actual accumulated error.
    ROOM_X = 1000.0
    ROOM_Y = 3.0
    ROOM_Z = 3.0
    SENSOR_POS = np.array([5.0, 1.5, 1.5])
elif ENVIRONMENT == "wall":
    # Large open space with a single wall. The bounding planes exist at
    # MAX_RANGE so rays escape without hitting anything in most directions,
    # leaving only the single wall at x=5 to provide geometric constraints.
    ROOM_X = 50.0
    ROOM_Y = 50.0
    ROOM_Z = 50.0
    SENSOR_POS = np.array([2.0, 25.0, 25.0])
elif ENVIRONMENT == "room_corridor":
    # A 10x10x3 m room connected to a 20x3x3 m corridor along +x.
    # The connection is an open door between x=10 and x=13 (the corridor
    # entrance) at y in [3.5, 6.5]. The room occupies x in [0, 10] and
    # the corridor x in [10, 30] with y in [3.5, 6.5].
    ROOM_X = 30.0
    ROOM_Y = 10.0
    ROOM_Z = 3.0
    SENSOR_POS = np.array([5.0, 5.0, 1.5])
else:
    # Default: cube room (also used by the "hover" environment since hover
    # only differs in the trajectory, not the geometry).
    ROOM_SIZE = 10.0
    ROOM_X = ROOM_SIZE
    ROOM_Y = ROOM_SIZE
    ROOM_Z = ROOM_SIZE
    SENSOR_POS = np.array([3.0, 4.0, 5.0])

SENSOR_ROT = np.eye(3)

N_RAYS_PER_SCAN = 1000
N_SCANS_TO_ACCUMULATE = 10
RANGE_NOISE_STD = 0.02
MAX_RANGE = 30.0


def make_room_planes(x, y, z):
    """Return a list of (point_on_plane, outward_normal, plane_bounds) tuples.

    bounds is (umin, umax, vmin, vmax) in the local 2D coordinates of the plane,
    used for finite-extent intersection tests.
    """
    return [
        (np.array([0.0, y / 2, z / 2]), np.array([1.0, 0.0, 0.0]), (-y / 2, y / 2, -z / 2, z / 2)),
        (np.array([x, y / 2, z / 2]), np.array([-1.0, 0.0, 0.0]), (-y / 2, y / 2, -z / 2, z / 2)),
        (np.array([x / 2, 0.0, z / 2]), np.array([0.0, 1.0, 0.0]), (-x / 2, x / 2, -z / 2, z / 2)),
        (np.array([x / 2, y, z / 2]), np.array([0.0, -1.0, 0.0]), (-x / 2, x / 2, -z / 2, z / 2)),
        (np.array([x / 2, y / 2, 0.0]), np.array([0.0, 0.0, 1.0]), (-x / 2, x / 2, -y / 2, y / 2)),
        (np.array([x / 2, y / 2, z]), np.array([0.0, 0.0, -1.0]), (-x / 2, x / 2, -y / 2, y / 2)),
    ]


def make_box_planes(center, size):
    """Return six finite planes forming a box centered at center with given size.

    center: (cx, cy, cz)
    size: (sx, sy, sz) full extents (not half-extents)
    """
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    return [
        (np.array([cx - hx, cy, cz]), np.array([-1.0, 0.0, 0.0]), (-hy, hy, -hz, hz)),
        (np.array([cx + hx, cy, cz]), np.array([1.0, 0.0, 0.0]), (-hy, hy, -hz, hz)),
        (np.array([cx, cy - hy, cz]), np.array([0.0, -1.0, 0.0]), (-hx, hx, -hz, hz)),
        (np.array([cx, cy + hy, cz]), np.array([0.0, 1.0, 0.0]), (-hx, hx, -hz, hz)),
        (np.array([cx, cy, cz - hz]), np.array([0.0, 0.0, -1.0]), (-hx, hx, -hy, hy)),
        (np.array([cx, cy, cz + hz]), np.array([0.0, 0.0, 1.0]), (-hx, hx, -hy, hy)),
    ]


if ENVIRONMENT == "wall":
    # Single-wall environment: one essentially infinite vertical wall.
    # The wall is made very large (200x200 m) so that within the sensor's
    # FOV and MAX_RANGE, only the wall plane is ever hit and no edge
    # information leaks in. This approximates the ideal 1-DOF-observable
    # case: the only geometric constraint is the wall-normal direction.
    WALL_CENTER = np.array([5.0, 25.0, 25.0])
    WALL_NORMAL = np.array([-1.0, 0.0, 0.0])
    WALL_HALF_Y = 200.0
    WALL_HALF_Z = 200.0
    PLANES = [
        (WALL_CENTER, WALL_NORMAL, (-WALL_HALF_Y, WALL_HALF_Y, -WALL_HALF_Z, WALL_HALF_Z)),
    ]
    OBSTACLES = []
elif ENVIRONMENT == "room_corridor":
    # Composite environment: a 10x10x3 m room connected to a 20x3x3 m
    # corridor along +x. The corridor entrance is an open doorway at
    # x=10, y in [3.5, 6.5].
    #
    # Layout (top-down):
    #
    #   y=10 +--------+
    #        |        |
    #        |  ROOM  |
    #        |        +---------------------+  y=6.5
    #        |                              |
    #        |             CORRIDOR         |
    #        |                              |
    #        |        +---------------------+  y=3.5
    #        |        |
    #        |        |
    #   y=0  +--------+
    #        x=0   x=10                    x=30
    FLOOR_Z = 0.0
    CEIL_Z = 3.0
    ROOM_HALF_X = 5.0
    ROOM_HALF_Y = 5.0
    ROOM_HALF_Z = 1.5
    COR_START_X = 10.0
    COR_END_X = 30.0
    COR_Y_LOW = 3.5
    COR_Y_HIGH = 6.5
    COR_CENTER_X = (COR_START_X + COR_END_X) / 2
    COR_HALF_X = (COR_END_X - COR_START_X) / 2
    COR_HALF_Y = (COR_Y_HIGH - COR_Y_LOW) / 2
    COR_CENTER_Y = (COR_Y_LOW + COR_Y_HIGH) / 2
    COR_HALF_Z = 1.5

    PLANES = [
        # Room walls (square, 10x10, centered at (5, 5))
        # Left wall x=0
        (np.array([0.0, 5.0, 1.5]), np.array([1.0, 0.0, 0.0]),
         (-ROOM_HALF_Y, ROOM_HALF_Y, -ROOM_HALF_Z, ROOM_HALF_Z)),
        # Bottom wall y=0
        (np.array([5.0, 0.0, 1.5]), np.array([0.0, 1.0, 0.0]),
         (-ROOM_HALF_X, ROOM_HALF_X, -ROOM_HALF_Z, ROOM_HALF_Z)),
        # Top wall y=10
        (np.array([5.0, 10.0, 1.5]), np.array([0.0, -1.0, 0.0]),
         (-ROOM_HALF_X, ROOM_HALF_X, -ROOM_HALF_Z, ROOM_HALF_Z)),
        # Right wall x=10 (split into two pieces around the doorway)
        # Piece below doorway: y in [0, 3.5]
        (np.array([10.0, 1.75, 1.5]), np.array([-1.0, 0.0, 0.0]),
         (-1.75, 1.75, -ROOM_HALF_Z, ROOM_HALF_Z)),
        # Piece above doorway: y in [6.5, 10]
        (np.array([10.0, 8.25, 1.5]), np.array([-1.0, 0.0, 0.0]),
         (-1.75, 1.75, -ROOM_HALF_Z, ROOM_HALF_Z)),
        # Room floor z=0 (over full 10x10 room)
        (np.array([5.0, 5.0, 0.0]), np.array([0.0, 0.0, 1.0]),
         (-ROOM_HALF_X, ROOM_HALF_X, -ROOM_HALF_Y, ROOM_HALF_Y)),
        # Room ceiling z=3
        (np.array([5.0, 5.0, 3.0]), np.array([0.0, 0.0, -1.0]),
         (-ROOM_HALF_X, ROOM_HALF_X, -ROOM_HALF_Y, ROOM_HALF_Y)),
        # Corridor walls
        # Bottom wall y=3.5, x in [10, 30]
        (np.array([COR_CENTER_X, COR_Y_LOW, 1.5]), np.array([0.0, 1.0, 0.0]),
         (-COR_HALF_X, COR_HALF_X, -COR_HALF_Z, COR_HALF_Z)),
        # Top wall y=6.5, x in [10, 30]
        (np.array([COR_CENTER_X, COR_Y_HIGH, 1.5]), np.array([0.0, -1.0, 0.0]),
         (-COR_HALF_X, COR_HALF_X, -COR_HALF_Z, COR_HALF_Z)),
        # End wall x=30, y in [3.5, 6.5]
        (np.array([COR_END_X, COR_CENTER_Y, 1.5]), np.array([-1.0, 0.0, 0.0]),
         (-COR_HALF_Y, COR_HALF_Y, -COR_HALF_Z, COR_HALF_Z)),
        # Corridor floor
        (np.array([COR_CENTER_X, COR_CENTER_Y, 0.0]), np.array([0.0, 0.0, 1.0]),
         (-COR_HALF_X, COR_HALF_X, -COR_HALF_Y, COR_HALF_Y)),
        # Corridor ceiling
        (np.array([COR_CENTER_X, COR_CENTER_Y, 3.0]), np.array([0.0, 0.0, -1.0]),
         (-COR_HALF_X, COR_HALF_X, -COR_HALF_Y, COR_HALF_Y)),
    ]
    OBSTACLES = [
        {"center": (2.5, 2.5, 0.5), "size": (1.0, 1.0, 1.0)},
        {"center": (7.0, 7.5, 0.6), "size": (0.8, 0.8, 1.2)},
        {"center": (18.0, 5.0, 0.6), "size": (0.4, 0.4, 1.2)},
        {"center": (25.0, 5.0, 0.7), "size": (0.4, 0.4, 1.4)},
    ]
    for obs in OBSTACLES:
        PLANES.extend(make_box_planes(obs["center"], obs["size"]))
else:
    PLANES = make_room_planes(ROOM_X, ROOM_Y, ROOM_Z)

    if ENVIRONMENT == "corridor":
        OBSTACLES = [
            {"center": (8.0, 1.5, 0.6), "size": (0.5, 0.4, 1.2)},
            {"center": (15.0, 1.5, 0.5), "size": (0.6, 0.4, 1.0)},
            {"center": (22.0, 1.5, 0.7), "size": (0.5, 0.4, 1.4)},
        ]
    elif ENVIRONMENT == "long_corridor":
        # A handful of pillars at regular intervals give some structure
        # without making the corridor "feature-rich". They ensure the
        # scan-to-scan registration has non-empty FIM everywhere.
        OBSTACLES = [
            {"center": (x, 1.5, 0.7), "size": (0.4, 0.4, 1.4)}
            for x in range(20, 1000, 40)
        ]
    else:
        OBSTACLES = [
            {"center": (6.5, 2.0, 0.4), "size": (1.5, 0.8, 0.8)},
            {"center": (2.0, 6.0, 0.45), "size": (1.2, 1.2, 0.9)},
            {"center": (8.0, 6.5, 1.0), "size": (0.4, 0.4, 2.0)},
            {"center": (5.0, 1.0, 0.6), "size": (2.0, 0.5, 1.2)},
        ]
    for obs in OBSTACLES:
        PLANES.extend(make_box_planes(obs["center"], obs["size"]))


def ray_plane_intersect(origin, direction, plane_point, plane_normal, bounds):
    """Return distance to plane intersection or None if no valid hit.

    Bounds check uses two basis vectors orthogonal to the plane normal to
    project the hit point into local 2D coordinates.
    """
    denom = np.dot(plane_normal, direction)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(plane_normal, plane_point - origin) / denom
    if t <= 0.0:
        return None

    hit = origin + t * direction

    if abs(plane_normal[0]) > 0.9:
        u_axis = np.array([0.0, 1.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])
    elif abs(plane_normal[1]) > 0.9:
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 0.0, 1.0])
    else:
        u_axis = np.array([1.0, 0.0, 0.0])
        v_axis = np.array([0.0, 1.0, 0.0])

    rel = hit - plane_point
    u = np.dot(rel, u_axis)
    v = np.dot(rel, v_axis)
    umin, umax, vmin, vmax = bounds
    if umin <= u <= umax and vmin <= v <= vmax:
        return t
    return None


def cast_ray(origin, direction):
    closest = MAX_RANGE
    for plane_point, plane_normal, bounds in PLANES:
        t = ray_plane_intersect(origin, direction, plane_point, plane_normal, bounds)
        if t is not None and t < closest:
            closest = t
    return closest


def mid360_rosette_directions(scan_index, n_rays):
    """Sample ray directions uniformly within the Livox Mid-360 FOV.

    The Livox Mid-360 has:
    - Horizontal FOV: 360 degrees
    - Vertical FOV: 59 degrees, asymmetric from -7 to +52 degrees
      (mostly upward-looking, dome-shaped)

    For each ray, azimuth is sampled uniformly in [0, 2*pi] and the sine of
    elevation is sampled uniformly in [sin(v_min), sin(v_max)] to produce
    equal density per unit area on the sphere.

    The scan_index argument is unused (each scan is an independent random
    sample), but kept for API compatibility.
    """
    del scan_index
    azimuth = np.random.uniform(0.0, 2 * np.pi, n_rays)

    v_min = np.radians(-7.0)
    v_max = np.radians(52.0)
    sin_e = np.random.uniform(np.sin(v_min), np.sin(v_max), n_rays)
    elevation = np.arcsin(sin_e)

    cos_e = np.cos(elevation)
    dx = cos_e * np.cos(azimuth)
    dy = cos_e * np.sin(azimuth)
    dz = np.sin(elevation)
    return np.stack([dx, dy, dz], axis=1)


def simulate_scan(origin, rotation, scan_index, n_rays):
    """Simulate one scan from origin with body orientation rotation.

    Returns world-frame points.
    """
    body_directions = mid360_rosette_directions(scan_index, n_rays)
    points = []
    for d_body in body_directions:
        d_world = rotation @ d_body
        r = cast_ray(origin, d_world)
        if r < MAX_RANGE:
            r_noisy = r + np.random.normal(0, RANGE_NOISE_STD)
            points.append(origin + r_noisy * d_world)
    return np.array(points)


def plot_3d(points, title, output_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="black", alpha=0.5)
    ax.scatter(*SENSOR_POS, color="red", s=60, marker="x", label="sensor")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim(0, ROOM_X)
    ax.set_ylim(0, ROOM_Y)
    ax.set_zlim(0, ROOM_Z)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path}")


def plot_topdown(points, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(points[:, 0], points[:, 1], s=1, c="black", alpha=0.5)
    ax.scatter(SENSOR_POS[0], SENSOR_POS[1], color="red", s=60, marker="x", label="sensor")
    ax.plot([0, ROOM_X, ROOM_X, 0, 0], [0, 0, ROOM_Y, ROOM_Y, 0], color="#888888", linewidth=1.5)
    for obs in OBSTACLES:
        cx, cy, _ = obs["center"]
        sx, sy, _ = obs["size"]
        ax.plot(
            [cx - sx / 2, cx + sx / 2, cx + sx / 2, cx - sx / 2, cx - sx / 2],
            [cy - sy / 2, cy - sy / 2, cy + sy / 2, cy + sy / 2, cy - sy / 2],
            color="#cc6600", linewidth=1.2,
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-0.5, ROOM_X + 0.5)
    ax.set_ylim(-0.5, ROOM_Y + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved {output_path}")


single = simulate_scan(SENSOR_POS, SENSOR_ROT, scan_index=0, n_rays=N_RAYS_PER_SCAN)
print(f"Single scan: {len(single)} points")
plot_3d(single, "Single non-repetitive scan (3D)", f"{OUT_DIR}/sim3d_single_3d.png")
plot_topdown(single, "Single non-repetitive scan (top-down)", f"{OUT_DIR}/sim3d_single_topdown.png")

accumulated = []
for i in range(N_SCANS_TO_ACCUMULATE):
    scan = simulate_scan(SENSOR_POS, SENSOR_ROT, scan_index=i, n_rays=N_RAYS_PER_SCAN)
    if len(scan) > 0:
        accumulated.append(scan)
accumulated = np.vstack(accumulated)
print(f"Accumulated ({N_SCANS_TO_ACCUMULATE} scans): {len(accumulated)} points")
plot_3d(accumulated, f"Accumulated scans ({N_SCANS_TO_ACCUMULATE} frames)",
        f"{OUT_DIR}/sim3d_accumulated_3d.png")
plot_topdown(accumulated, f"Accumulated scans ({N_SCANS_TO_ACCUMULATE} frames)",
             f"{OUT_DIR}/sim3d_accumulated_topdown.png")

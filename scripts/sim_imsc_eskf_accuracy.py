"""Accuracy comparison: IMSC-ESKF vs standard IESKF.

Runs two filters on the same data:
1. Standard IESKF: scan-to-map with online map (like FAST-LIO2)
2. IMSC-ESKF: sliding window of poses, windowed map where every
   scan-to-map match is a multi-state constraint (H nonzero for
   both current pose and source pose)

Focus is on trajectory accuracy — drift covariance is not analyzed here.

Run: pixi run python sim_imsc_eskf_accuracy.py
"""

import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim_environment_3d import ROOM_X, ROOM_Y
from sim_imu_trajectory import (
    GRAVITY, IMU_RATE, INITIAL_GYR_BIAS, INITIAL_ACC_BIAS, trajectory_at,
)

OUT_DIR = os.environ.get(
    "SIM_OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_data"),
)
os.makedirs(OUT_DIR, exist_ok=True)

DT_IMU = 1.0 / IMU_RATE

DT_IMU_REF = 1.0 / IMU_RATE
GYR_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
ACC_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
GYR_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)
ACC_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)

LIDAR_POINT_VAR = 0.001
MAX_PLANE_DIST = 0.5
MAX_IEKF_ITERS = 3
CONVERGENCE_THRESH = 1e-4
MAX_POINTS_PER_SCAN = 1000

NN_K = 5
PLANE_FIT_THRESH = 0.05
MAP_VOXEL_SIZE = 0.2

WINDOW_SIZE = 20

N_IMU = 15


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def rodrigues_exp(omega):
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3) + skew(omega)
    axis = omega / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rodrigues_log(R):
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return angle / (2 * np.sin(angle)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def fit_plane(neighbors):
    centroid = np.mean(neighbors, axis=0)
    centered = neighbors - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    if eigvals[0] < 0 or eigvals[1] < eigvals[0] * 3:
        return None
    d = -np.dot(normal, centroid)
    residuals = np.abs(neighbors @ normal + d)
    if np.max(residuals) > PLANE_FIT_THRESH:
        return None
    return normal, d


class OnlineMap:
    def __init__(self):
        self.points = []
        self.voxel_set = set()
        self.tree = None
        self._dirty = False

    def add_points(self, world_points):
        for p in world_points:
            key = tuple(np.round(p / MAP_VOXEL_SIZE).astype(int))
            if key not in self.voxel_set:
                self.voxel_set.add(key)
                self.points.append(p.copy())
        self._dirty = True

    def build_tree(self):
        if self._dirty and len(self.points) > 0:
            self.tree = cKDTree(np.array(self.points))
            self._dirty = False

    def __len__(self):
        return len(self.points)


class WindowedMap:
    def __init__(self):
        self.body_points = []
        self.pose_indices = []
        self.voxel_set = set()

    def add_points(self, body_points, pose_idx, R, p):
        for pb in body_points:
            pw = R @ pb + p
            key = tuple(np.round(pw / MAP_VOXEL_SIZE).astype(int))
            if key not in self.voxel_set:
                self.voxel_set.add(key)
                self.body_points.append(pb.copy())
                self.pose_indices.append(pose_idx)

    def remove_oldest_pose(self):
        keep = [i for i, idx in enumerate(self.pose_indices) if idx > 0]
        self.body_points = [self.body_points[i] for i in keep]
        self.pose_indices = [self.pose_indices[i] - 1 for i in keep]
        self.voxel_set.clear()

    def get_world_points_and_indices(self, window_R, window_p):
        n = len(self.body_points)
        world = np.zeros((n, 3))
        for i in range(n):
            pidx = self.pose_indices[i]
            world[i] = window_R[pidx] @ self.body_points[i] + window_p[pidx]
        return world, self.pose_indices

    def __len__(self):
        return len(self.body_points)


# ============================================================
# Standard IESKF (baseline, from sim_iekf_3d.py)
# ============================================================

class State15:
    def __init__(self, R, p, v, bg, ba):
        self.R = R.copy()
        self.p = p.copy()
        self.v = v.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()

    def copy(self):
        return State15(self.R, self.p, self.v, self.bg, self.ba)

    def boxplus(self, dx):
        return State15(
            self.R @ rodrigues_exp(dx[0:3]),
            self.p + dx[3:6],
            self.v + dx[6:9],
            self.bg + dx[9:12],
            self.ba + dx[12:15],
        )

    def boxminus(self, other):
        return np.concatenate([
            rodrigues_log(other.R.T @ self.R),
            self.p - other.p, self.v - other.v,
            self.bg - other.bg, self.ba - other.ba,
        ])


def predict15(state, P, gyro, acc, dt):
    omega = gyro - state.bg
    a_body = acc - state.ba
    a_world = state.R @ a_body + GRAVITY
    new_state = State15(
        state.R @ rodrigues_exp(omega * dt),
        state.p + state.v * dt + 0.5 * a_world * dt * dt,
        state.v + a_world * dt,
        state.bg, state.ba,
    )
    F = np.eye(N_IMU)
    F[0:3, 0:3] = rodrigues_exp(-omega * dt)
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    F[6:9, 0:3] = -state.R @ skew(a_body) * dt
    F[6:9, 12:15] = -state.R * dt
    G = np.zeros((N_IMU, 12))
    G[0:3, 0:3] = -np.eye(3) * dt
    G[6:9, 3:6] = -state.R * dt
    G[9:12, 6:9] = np.eye(3) * dt
    G[12:15, 9:12] = np.eye(3) * dt
    Q_c = np.diag([
        GYR_NOISE_STD**2]*3 + [ACC_NOISE_STD**2]*3 +
        [GYR_BIAS_WALK_STD**2]*3 + [ACC_BIAS_WALK_STD**2]*3)
    return new_state, F @ P @ F.T + G @ Q_c @ G.T


def ieskf_update(state_prior, P_prior, points_body, online_map):
    online_map.build_tree()
    if online_map.tree is None or len(online_map) < NN_K:
        return state_prior, P_prior
    map_pts = np.array(online_map.points)

    matches = []
    for pb in points_body:
        pw = state_prior.R @ pb + state_prior.p
        dists, idxs = online_map.tree.query(pw, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue
        plane = fit_plane(map_pts[idxs])
        if plane is None:
            continue
        n_w, d = plane
        if abs(np.dot(n_w, pw) + d) > MAX_PLANE_DIST:
            continue
        matches.append((pb, n_w, d))

    if len(matches) < 5:
        return state_prior, P_prior
    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]

    n_meas = len(matches)
    R_meas = LIDAR_POINT_VAR * np.eye(n_meas)
    state = state_prior.copy()
    H_f = K_f = None

    for it in range(MAX_IEKF_ITERS):
        h = np.zeros(n_meas)
        H = np.zeros((n_meas, N_IMU))
        for i, (pb, nw, dw) in enumerate(matches):
            pw = state.R @ pb + state.p
            h[i] = np.dot(nw, pw) + dw
            H[i, 0:3] = -nw @ state.R @ skew(pb)
            H[i, 3:6] = nw
        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)
        dx = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx)
        state = state_prior.boxplus(dx + delta)
        H_f, K_f = H, K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    IKH = np.eye(N_IMU) - K_f @ H_f
    P_new = IKH @ P_prior @ IKH.T + K_f @ R_meas @ K_f.T
    return state, 0.5 * (P_new + P_new.T)


def run_ieskf(imu_data, lidar_data):
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = State15(R0, p0.copy(), v0.copy(),
                    INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2
    omap = OnlineMap()
    hist = {"t": [], "p_est": [], "p_true": [], "pos_err": []}
    li = 0
    for s in imu_data:
        t, gyro, acc = s["t"], s["gyro"], s["acc"]
        state, P = predict15(state, P, gyro, acc, DT_IMU)
        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                if len(omap) >= NN_K:
                    state, P = ieskf_update(state, P, scan["points_body"], omap)
                wp = (state.R @ scan["points_body"].T).T + state.p
                omap.add_points(wp)
            li += 1
        _, p_true, _, _, _ = trajectory_at(t)
        hist["t"].append(t)
        hist["p_est"].append(state.p.copy())
        hist["p_true"].append(p_true.copy())
        hist["pos_err"].append(np.linalg.norm(state.p - p_true))
    for k in hist:
        hist[k] = np.array(hist[k])
    return hist


# ============================================================
# IMSC-ESKF with windowed map
# ============================================================

class MSCState:
    def __init__(self, R, p, v, bg, ba):
        self.R = R.copy()
        self.p = p.copy()
        self.v = v.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()
        self.window_R = []
        self.window_p = []
        self.window_scans = []

    @property
    def n_window(self):
        return len(self.window_R)

    @property
    def n_dof(self):
        return N_IMU + 6 * self.n_window

    def copy(self):
        s = MSCState(self.R, self.p, self.v, self.bg, self.ba)
        s.window_R = [R.copy() for R in self.window_R]
        s.window_p = [p.copy() for p in self.window_p]
        s.window_scans = list(self.window_scans)
        return s

    def boxplus(self, dx):
        s = MSCState(
            self.R @ rodrigues_exp(dx[0:3]),
            self.p + dx[3:6],
            self.v + dx[6:9],
            self.bg + dx[9:12],
            self.ba + dx[12:15],
        )
        s.window_scans = list(self.window_scans)
        for i in range(self.n_window):
            base = N_IMU + 6 * i
            s.window_R.append(self.window_R[i] @ rodrigues_exp(dx[base:base+3]))
            s.window_p.append(self.window_p[i] + dx[base+3:base+6])
        return s

    def boxminus(self, other):
        dx = np.zeros(self.n_dof)
        dx[0:3] = rodrigues_log(other.R.T @ self.R)
        dx[3:6] = self.p - other.p
        dx[6:9] = self.v - other.v
        dx[9:12] = self.bg - other.bg
        dx[12:15] = self.ba - other.ba
        for i in range(self.n_window):
            base = N_IMU + 6 * i
            dx[base:base+3] = rodrigues_log(other.window_R[i].T @ self.window_R[i])
            dx[base+3:base+6] = self.window_p[i] - other.window_p[i]
        return dx


def predict_msc(state, P, gyro, acc, dt):
    omega = gyro - state.bg
    a_body = acc - state.ba
    a_world = state.R @ a_body + GRAVITY
    new_state = state.copy()
    new_state.R = state.R @ rodrigues_exp(omega * dt)
    new_state.p = state.p + state.v * dt + 0.5 * a_world * dt * dt
    new_state.v = state.v + a_world * dt
    n = state.n_dof
    F = np.eye(n)
    F[0:3, 0:3] = rodrigues_exp(-omega * dt)
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    F[6:9, 0:3] = -state.R @ skew(a_body) * dt
    F[6:9, 12:15] = -state.R * dt
    G = np.zeros((n, 12))
    G[0:3, 0:3] = -np.eye(3) * dt
    G[6:9, 3:6] = -state.R * dt
    G[9:12, 6:9] = np.eye(3) * dt
    G[12:15, 9:12] = np.eye(3) * dt
    Q_c = np.diag([
        GYR_NOISE_STD**2]*3 + [ACC_NOISE_STD**2]*3 +
        [GYR_BIAS_WALK_STD**2]*3 + [ACC_BIAS_WALK_STD**2]*3)
    return new_state, F @ P @ F.T + G @ Q_c @ G.T


def clone_pose(state, P, scan_body):
    new_state = state.copy()
    new_state.window_R.append(state.R.copy())
    new_state.window_p.append(state.p.copy())
    new_state.window_scans.append(scan_body)
    n_old = P.shape[0]
    n_new = n_old + 6
    P_new = np.zeros((n_new, n_new))
    P_new[:n_old, :n_old] = P
    J = np.zeros((6, n_old))
    J[0:3, 0:3] = np.eye(3)
    J[3:6, 3:6] = np.eye(3)
    P_new[n_old:n_new, :n_old] = J @ P
    P_new[:n_old, n_old:n_new] = P_new[n_old:n_new, :n_old].T
    P_new[n_old:n_new, n_old:n_new] = J @ P @ J.T
    return new_state, P_new


def marginalize_oldest_simple(state, P):
    if state.n_window < 2:
        return state, P
    new_state = state.copy()
    new_state.window_R = new_state.window_R[1:]
    new_state.window_p = new_state.window_p[1:]
    new_state.window_scans = new_state.window_scans[1:]
    keep = list(range(0, N_IMU)) + list(range(N_IMU + 6, P.shape[0]))
    return new_state, P[np.ix_(keep, keep)]


def imsc_eskf_update(state_prior, P_prior, curr_pose_idx, wmap):
    if len(wmap) < NN_K:
        return state_prior, P_prior

    world_map, source_indices = wmap.get_world_points_and_indices(
        state_prior.window_R, state_prior.window_p)
    tree = cKDTree(world_map)

    R_k = state_prior.window_R[curr_pose_idx]
    p_k = state_prior.window_p[curr_pose_idx]
    scan = state_prior.window_scans[curr_pose_idx]

    matches = []
    n_check = min(len(scan), MAX_POINTS_PER_SCAN)
    indices = (np.random.choice(len(scan), n_check, replace=False)
               if len(scan) > n_check else range(len(scan)))

    for idx in indices:
        pb_k = scan[idx]
        pw_k = R_k @ pb_k + p_k
        dists, nn_idxs = tree.query(pw_k, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue
        neighbors = world_map[nn_idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        normal, d = plane
        if abs(np.dot(normal, pw_k) + d) > MAX_PLANE_DIST:
            continue

        src_idx = source_indices[nn_idxs[0]]
        if src_idx == curr_pose_idx:
            continue

        src_body_pt = wmap.body_points[nn_idxs[0]]
        matches.append((pb_k, src_body_pt, normal, curr_pose_idx, src_idx))

    if len(matches) < 5:
        return state_prior, P_prior
    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]

    n_meas = len(matches)
    R_meas = LIDAR_POINT_VAR * np.eye(n_meas)
    n_dof = state_prior.n_dof
    state = state_prior.copy()
    H_f = K_f = None

    for it in range(MAX_IEKF_ITERS):
        h = np.zeros(n_meas)
        H = np.zeros((n_meas, n_dof))

        for i, (pb_k, pb_j, nw, kidx, jidx) in enumerate(matches):
            Rk = state.window_R[kidx]
            pk = state.window_p[kidx]
            Rj = state.window_R[jidx]
            pj = state.window_p[jidx]

            pwk = Rk @ pb_k + pk
            pwj = Rj @ pb_j + pj
            h[i] = np.dot(nw, pwk - pwj)

            bk = N_IMU + 6 * kidx
            bj = N_IMU + 6 * jidx
            H[i, bk:bk+3] = -nw @ Rk @ skew(pb_k)
            H[i, bk+3:bk+6] = nw
            H[i, bj:bj+3] = nw @ Rj @ skew(pb_j)
            H[i, bj+3:bj+6] = -nw

        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)
        dx = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx)
        state = state_prior.boxplus(dx + delta)
        H_f, K_f = H, K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    IKH = np.eye(n_dof) - K_f @ H_f
    P_new = IKH @ P_prior @ IKH.T + K_f @ R_meas @ K_f.T
    return state, 0.5 * (P_new + P_new.T)


def run_imsc_eskf(imu_data, lidar_data):
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = MSCState(R0, p0.copy(), v0.copy(),
                     INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2
    wmap = WindowedMap()
    hist = {"t": [], "p_est": [], "p_true": [], "pos_err": [], "n_matches": [], "n_window": []}
    li = 0
    for s in imu_data:
        t, gyro, acc = s["t"], s["gyro"], s["acc"]
        state, P = predict_msc(state, P, gyro, acc, DT_IMU)
        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                state, P = clone_pose(state, P, scan["points_body"])
                k_idx = state.n_window - 1

                n_matches = 0
                if len(wmap) >= NN_K:
                    state, P = imsc_eskf_update(state, P, k_idx, wmap)

                wmap.add_points(scan["points_body"], k_idx,
                                state.window_R[k_idx], state.window_p[k_idx])

                while state.n_window > WINDOW_SIZE:
                    state, P = marginalize_oldest_simple(state, P)
                    wmap.remove_oldest_pose()

                hist["n_matches"].append(n_matches)
                hist["n_window"].append(state.n_window)
            li += 1

        _, p_true, _, _, _ = trajectory_at(t)
        hist["t"].append(t)
        hist["p_est"].append(state.p.copy())
        hist["p_true"].append(p_true.copy())
        hist["pos_err"].append(np.linalg.norm(state.p - p_true))

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist


def plot_comparison(h_ieskf, h_imsc):
    t1, t2 = h_ieskf["t"], h_imsc["t"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(h_ieskf["p_true"][:, 0], h_ieskf["p_true"][:, 1],
            "k-", linewidth=2, label="ground truth")
    ax.plot(h_ieskf["p_est"][:, 0], h_ieskf["p_est"][:, 1],
            "r-", linewidth=1, alpha=0.8, label="IESKF (scan-to-map)")
    ax.plot(h_imsc["p_est"][:, 0], h_imsc["p_est"][:, 1],
            "b-", linewidth=1, alpha=0.8, label=f"IMSC-ESKF (win={WINDOW_SIZE})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory comparison")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t1, h_ieskf["pos_err"], "r-", linewidth=1, label="IESKF")
    ax.plot(t2, h_imsc["pos_err"], "b-", linewidth=1, label="IMSC-ESKF")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title("Position error over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[2]
    labels = ["IESKF", f"IMSC-ESKF\n(win={WINDOW_SIZE})"]
    means = [np.mean(h_ieskf["pos_err"]), np.mean(h_imsc["pos_err"])]
    maxes = [np.max(h_ieskf["pos_err"]), np.max(h_imsc["pos_err"])]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, means, w, label="mean", color=["#cc4444", "#4444cc"])
    ax.bar(x + w/2, maxes, w, label="max", color=["#ff8888", "#8888ff"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("position error (m)")
    ax.set_title("Accuracy comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_imsc_accuracy.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_imsc_accuracy.png")


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")
imu_data = [{"t": row[0], "gyro": row[1:4], "acc": row[4:7]} for row in imu_arr]
lidar_data = list(lidar_arr)

print("\nRunning standard IESKF (scan-to-map)...")
np.random.seed(123)
h_ieskf = run_ieskf(imu_data, lidar_data)
print(f"  Position error: mean={np.mean(h_ieskf['pos_err']):.4f} m, "
      f"max={np.max(h_ieskf['pos_err']):.4f} m, final={h_ieskf['pos_err'][-1]:.4f} m")

results = {}
for ws in [10, 20, 50, 100, 200]:
    WINDOW_SIZE_RUN = ws
    old_ws = WINDOW_SIZE
    import sim_imsc_eskf_accuracy as _self
    _self.WINDOW_SIZE = ws

    print(f"\nRunning IMSC-ESKF (window={ws})...")
    np.random.seed(123)
    h = run_imsc_eskf(imu_data, lidar_data)
    me = np.mean(h['pos_err'])
    mx = np.max(h['pos_err'])
    print(f"  Position error: mean={me:.4f} m, max={mx:.4f} m")
    results[ws] = h

    _self.WINDOW_SIZE = old_ws

print(f"\n{'Window':>8} {'Mean err':>10} {'Max err':>10} {'Ratio':>8}")
print("-" * 40)
ieskf_mean = np.mean(h_ieskf['pos_err'])
print(f"{'IESKF':>8} {ieskf_mean:>10.4f} {np.max(h_ieskf['pos_err']):>10.4f} {'1.00x':>8}")
for ws in sorted(results.keys()):
    me = np.mean(results[ws]['pos_err'])
    mx = np.max(results[ws]['pos_err'])
    print(f"{ws:>8} {me:>10.4f} {mx:>10.4f} {me/ieskf_mean:>7.2f}x")

best_ws = min(results.keys(), key=lambda w: np.mean(results[w]['pos_err']))
h_imsc = results[best_ws]
WINDOW_SIZE = best_ws
plot_comparison(h_ieskf, h_imsc)

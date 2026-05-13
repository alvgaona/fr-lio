"""Iterated Multi-State Constraint Error-State Kalman Filter (IMSC-ESKF)
for LiDAR-inertial odometry.

Maintains a sliding window of cloned poses in the state. Point-to-plane
measurements between scans create multi-state constraints linking pose
pairs. Drift covariance emerges naturally from the filter's own cross-
covariance structure — no external CRLB computation needed.

State layout (15 + 6K DOF):
  [0:3]    dtheta   current rotation
  [3:6]    dp       current position
  [6:9]    dv       current velocity
  [9:12]   dbg      gyro bias
  [12:15]  dba      accel bias
  [15+6i : 15+6i+3] dtheta_i  window pose i rotation
  [15+6i+3 : 15+6i+6] dp_i    window pose i position

Run: pixi run python sim_msc_eskf.py
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
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
N_IMU = 15

DT_IMU_REF = 1.0 / IMU_RATE
GYR_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
ACC_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
GYR_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)
ACC_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)

LIDAR_POINT_VAR = 0.001
MAX_PLANE_DIST = 0.5
MAX_POINTS_PER_UPDATE = 300

WINDOW_POSE_PROCESS_NOISE_POS = 0.0
WINDOW_POSE_PROCESS_NOISE_ROT = 0.0
MAX_IEKF_ITERS = 3
CONVERGENCE_THRESH = 1e-4

NN_K = 5
PLANE_FIT_THRESH = 0.05

WINDOW_SIZE = 250
MATCH_PREV_SCANS = 1

MAP_VOXEL_SIZE = 0.2
MAP_MAX_POINTS_PER_UPDATE = 1000
MAP_MAX_IEKF_ITERS = 3

S2S_MAX_POINTS = 300
S2S_NN_K = 5
S2S_MAX_NN_DIST = 1.0
S2S_MIN_EIGENVALUE = 1e-6
S2S_MIN_VALID_POINTS = 100
S2S_MAX_RESIDUAL = 0.3
S2S_ADAPTIVE_REJECT_RATIO = 10.0
S2S_ADAPTIVE_WINDOW = 20
S2S_MIN_TRANS_M = 0.01
S2S_MIN_ROT_RAD = 0.001
S2S_PERSIST_ALPHA = 1.0
S2S_PERSIST_NORMAL_TAU = 0.95
S2S_PERSIST_DIST_EPS = 0.10
S2S_PERSIST_HISTORY = 5


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


class WindowedMap:
    """Map that stores points in body frame with their source pose index.

    Points are transformed to world frame on-the-fly using the current
    state estimate for their source pose. When a pose is marginalized,
    its points are removed.
    """
    def __init__(self):
        self.body_points = []
        self.pose_indices = []
        self.voxel_set = set()

    def add_points(self, body_points, pose_idx, state):
        R = state.window_R[pose_idx]
        p = state.window_p[pose_idx]
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

    def get_world_points(self, state):
        world = np.zeros((len(self.body_points), 3))
        for i, (pb, pidx) in enumerate(zip(self.body_points, self.pose_indices)):
            R = state.window_R[pidx]
            p = state.window_p[pidx]
            world[i] = R @ pb + p
        return world

    def __len__(self):
        return len(self.body_points)


def find_windowed_map_correspondences(state, points_body_curr, curr_pose_idx, wmap):
    if len(wmap) < NN_K:
        return []

    world_map = wmap.get_world_points(state)
    tree = cKDTree(world_map)

    R_k = state.window_R[curr_pose_idx]
    p_k = state.window_p[curr_pose_idx]

    matches = []
    n_check = min(len(points_body_curr), MAP_MAX_POINTS_PER_UPDATE)
    indices = (np.random.choice(len(points_body_curr), n_check, replace=False)
               if len(points_body_curr) > n_check else range(len(points_body_curr)))

    for idx in indices:
        p_body_k = points_body_curr[idx]
        p_world_k = R_k @ p_body_k + p_k

        dists, nn_idxs = tree.query(p_world_k, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue

        neighbors = world_map[nn_idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        normal, d = plane

        if abs(np.dot(normal, p_world_k) + d) > MAX_PLANE_DIST:
            continue

        closest_idx = nn_idxs[0]
        source_pose_idx = wmap.pose_indices[closest_idx]
        source_body_pt = wmap.body_points[closest_idx]

        if source_pose_idx == curr_pose_idx:
            continue

        matches.append((p_body_k, source_body_pt, normal, curr_pose_idx, source_pose_idx))

    return matches


def windowed_map_update(state_prior, P_prior, all_matches):
    if len(all_matches) < 10:
        return state_prior, P_prior

    if len(all_matches) > MAP_MAX_POINTS_PER_UPDATE:
        idx = np.random.choice(len(all_matches), MAP_MAX_POINTS_PER_UPDATE, replace=False)
        all_matches = [all_matches[i] for i in idx]

    n_meas = len(all_matches)
    n_dof = state_prior.n_dof
    state = state_prior.copy()
    H_final = None
    K_final = None
    R_scalar = LIDAR_POINT_VAR

    for j in range(MAP_MAX_IEKF_ITERS):
        h = np.zeros(n_meas)
        H = np.zeros((n_meas, n_dof))

        for i, (p_body_k, p_body_j, n_w, k_idx, j_idx) in enumerate(all_matches):
            R_k = state.window_R[k_idx]
            p_k = state.window_p[k_idx]
            R_j = state.window_R[j_idx]
            p_j = state.window_p[j_idx]

            pw_k = R_k @ p_body_k + p_k
            pw_j = R_j @ p_body_j + p_j

            h[i] = np.dot(n_w, pw_k - pw_j)

            base_k = N_IMU + 6 * k_idx
            base_j = N_IMU + 6 * j_idx

            H[i, base_k:base_k+3] = -n_w @ R_k @ skew(p_body_k)
            H[i, base_k+3:base_k+6] = n_w
            H[i, base_j:base_j+3] = n_w @ R_j @ skew(p_body_j)
            H[i, base_j+3:base_j+6] = -n_w

        if j == MAP_MAX_IEKF_ITERS - 1 or np.linalg.norm(h) < CONVERGENCE_THRESH:
            residual_sq_sum = np.dot(h, h)
            dof = max(n_meas - 6, 1)
            R_scalar = max(residual_sq_sum / dof, LIDAR_POINT_VAR)

        R_meas = R_scalar * np.eye(n_meas)
        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)

        dx_from_prior = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx_from_prior)
        state = state_prior.boxplus(dx_from_prior + delta)

        H_final = H
        K_final = K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    R_final = R_scalar * np.eye(n_meas)
    IKH = np.eye(n_dof) - K_final @ H_final
    P_new = IKH @ P_prior @ IKH.T + K_final @ R_final @ K_final.T
    P_new = 0.5 * (P_new + P_new.T)
    return state, P_new


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


def predict(state, P, gyro, acc, dt):
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

    Q_continuous = np.diag([
        GYR_NOISE_STD**2, GYR_NOISE_STD**2, GYR_NOISE_STD**2,
        ACC_NOISE_STD**2, ACC_NOISE_STD**2, ACC_NOISE_STD**2,
        GYR_BIAS_WALK_STD**2, GYR_BIAS_WALK_STD**2, GYR_BIAS_WALK_STD**2,
        ACC_BIAS_WALK_STD**2, ACC_BIAS_WALK_STD**2, ACC_BIAS_WALK_STD**2,
    ])
    Q = G @ Q_continuous @ G.T
    new_P = F @ P @ F.T + Q

    for i in range(state.n_window):
        base = N_IMU + 6 * i
        new_P[base, base] += WINDOW_POSE_PROCESS_NOISE_ROT
        new_P[base+1, base+1] += WINDOW_POSE_PROCESS_NOISE_ROT
        new_P[base+2, base+2] += WINDOW_POSE_PROCESS_NOISE_ROT
        new_P[base+3, base+3] += WINDOW_POSE_PROCESS_NOISE_POS
        new_P[base+4, base+4] += WINDOW_POSE_PROCESS_NOISE_POS
        new_P[base+5, base+5] += WINDOW_POSE_PROCESS_NOISE_POS

    return new_state, new_P


def clone_pose(state, P, scan_body):
    new_state = state.copy()
    new_state.window_R.append(state.R.copy())
    new_state.window_p.append(state.p.copy())
    new_state.window_scans.append(scan_body)

    n_old = P.shape[0]
    n_new = n_old + 6
    P_new = np.zeros((n_new, n_new))
    P_new[:n_old, :n_old] = P

    J_clone = np.zeros((6, n_old))
    J_clone[0:3, 0:3] = np.eye(3)
    J_clone[3:6, 3:6] = np.eye(3)

    P_new[n_old:n_new, :n_old] = J_clone @ P
    P_new[:n_old, n_old:n_new] = P_new[n_old:n_new, :n_old].T
    P_new[n_old:n_new, n_old:n_new] = J_clone @ P @ J_clone.T

    return new_state, P_new


def marginalize_oldest(state, P, P_drift_accum):
    """Remove oldest pose. Accumulate the within-window drift before removal.

    Before removing, extract the relative covariance between the oldest
    pose and the next-oldest. Add it to the drift accumulator. Then drop
    the oldest rows/columns (simple removal, not Schur complement — Schur
    makes remaining states more confident, which is wrong for drift).
    """
    if state.n_window < 2:
        return state, P, P_drift_accum

    n = P.shape[0]
    oldest_base = N_IMU
    next_base = N_IMU + 6

    J_rel = np.zeros((6, n))
    J_rel[0:3, next_base:next_base+3] = np.eye(3)
    J_rel[0:3, oldest_base:oldest_base+3] = -np.eye(3)
    J_rel[3:6, next_base+3:next_base+6] = np.eye(3)
    J_rel[3:6, oldest_base+3:oldest_base+6] = -np.eye(3)
    P_rel_step = J_rel @ P @ J_rel.T
    P_rel_step = 0.5 * (P_rel_step + P_rel_step.T)
    P_drift_accum = P_drift_accum + P_rel_step

    new_state = state.copy()
    new_state.window_R = new_state.window_R[1:]
    new_state.window_p = new_state.window_p[1:]
    new_state.window_scans = new_state.window_scans[1:]

    keep = list(range(0, N_IMU)) + list(range(N_IMU + 6, n))
    P_new = P[np.ix_(keep, keep)]

    return new_state, P_new, P_drift_accum


def find_msc_correspondences(state, scan_curr, pose_k_idx, pose_j_idx):
    R_k = state.window_R[pose_k_idx]
    p_k = state.window_p[pose_k_idx]
    R_j = state.window_R[pose_j_idx]
    p_j = state.window_p[pose_j_idx]
    scan_j = state.window_scans[pose_j_idx]

    if len(scan_j) < NN_K or len(scan_curr) < NN_K:
        return []

    world_j = (R_j @ scan_j.T).T + p_j
    tree_j = cKDTree(world_j)

    matches = []
    n_check = min(len(scan_curr), MAX_POINTS_PER_UPDATE)
    indices = np.random.choice(len(scan_curr), n_check, replace=False) if len(scan_curr) > n_check else range(len(scan_curr))

    for idx in indices:
        p_body_k = scan_curr[idx]
        p_world_k = R_k @ p_body_k + p_k

        dists, nn_idxs = tree_j.query(p_world_k, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue

        neighbors = world_j[nn_idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        normal, d = plane

        residual = np.dot(normal, p_world_k) + d
        if abs(residual) > MAX_PLANE_DIST:
            continue

        p_body_j = scan_j[nn_idxs[0]]
        matches.append((p_body_k, p_body_j, normal, pose_k_idx, pose_j_idx))

    return matches


def compute_msc_residuals_and_H(state, matches):
    n_dof = state.n_dof
    n_meas = len(matches)
    h = np.zeros(n_meas)
    H = np.zeros((n_meas, n_dof))

    for i, (p_body_k, p_body_j, n_w, k_idx, j_idx) in enumerate(matches):
        R_k = state.window_R[k_idx]
        p_k = state.window_p[k_idx]
        R_j = state.window_R[j_idx]
        p_j = state.window_p[j_idx]

        pw_k = R_k @ p_body_k + p_k
        pw_j = R_j @ p_body_j + p_j

        h[i] = np.dot(n_w, pw_k - pw_j)

        base_k = N_IMU + 6 * k_idx
        base_j = N_IMU + 6 * j_idx

        H[i, base_k:base_k+3] = -n_w @ R_k @ skew(p_body_k)
        H[i, base_k+3:base_k+6] = n_w
        H[i, base_j:base_j+3] = n_w @ R_j @ skew(p_body_j)
        H[i, base_j+3:base_j+6] = -n_w

    return h, H


def msc_update(state_prior, P_prior, all_matches):
    if len(all_matches) < 10:
        return state_prior, P_prior

    if len(all_matches) > MAX_POINTS_PER_UPDATE:
        idx = np.random.choice(len(all_matches), MAX_POINTS_PER_UPDATE, replace=False)
        all_matches = [all_matches[i] for i in idx]

    n_meas = len(all_matches)
    state = state_prior.copy()
    H_final = None
    K_final = None
    R_scalar = LIDAR_POINT_VAR

    for j in range(MAX_IEKF_ITERS):
        h, H = compute_msc_residuals_and_H(state, all_matches)

        if j == MAX_IEKF_ITERS - 1 or np.linalg.norm(h) < CONVERGENCE_THRESH:
            residual_sq_sum = np.dot(h, h)
            dof = max(n_meas - 6, 1)
            R_scalar = max(residual_sq_sum / dof, LIDAR_POINT_VAR)

        R_meas = R_scalar * np.eye(n_meas)
        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)

        dx_from_prior = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx_from_prior)
        state = state_prior.boxplus(dx_from_prior + delta)

        H_final = H
        K_final = K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    R_final = R_scalar * np.eye(n_meas)
    IKH = np.eye(state.n_dof) - K_final @ H_final
    P_new = IKH @ P_prior @ IKH.T + K_final @ R_final @ K_final.T
    P_new = 0.5 * (P_new + P_new.T)
    return state, P_new


def extract_window_drift(state, P):
    """Extract drift as the relative covariance between the newest
    and oldest poses currently in the window."""
    if state.n_window < 2:
        return np.zeros((6, 6))

    oldest_base = N_IMU
    newest_base = N_IMU + 6 * (state.n_window - 1)
    n = P.shape[0]

    J = np.zeros((6, n))
    J[0:3, newest_base:newest_base+3] = np.eye(3)
    J[0:3, oldest_base:oldest_base+3] = -np.eye(3)
    J[3:6, newest_base+3:newest_base+6] = np.eye(3)
    J[3:6, oldest_base+3:oldest_base+6] = -np.eye(3)
    return J @ P @ J.T


def extract_oldest_marginal(state, P):
    """Extract the marginal covariance of the oldest pose in the window."""
    if state.n_window < 1:
        return np.zeros((6, 6))
    oldest_base = N_IMU
    return P[oldest_base:oldest_base+6, oldest_base:oldest_base+6].copy()


def compute_scan_to_scan_covariance(points_body_curr, R_curr, t_curr,
                                    prev_scan_points, prev_tree, R_prev, t_prev):
    R_rel = R_prev.T @ R_curr
    t_rel = R_prev.T @ (t_curr - t_prev)
    max_points = min(len(points_body_curr), S2S_MAX_POINTS)
    J = np.zeros((max_points, 6))
    residual_sum_sq = 0.0
    valid_count = 0
    planes_world = []

    for i in range(max_points):
        p_cur = points_body_curr[i]
        p_in_prev = R_rel @ p_cur + t_rel
        dists, idxs = prev_tree.query(p_in_prev, k=S2S_NN_K)
        if dists[-1] > S2S_MAX_NN_DIST:
            continue
        neighbors = prev_scan_points[idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        normal, d = plane
        residual = np.dot(normal, p_in_prev) + d
        if abs(residual) > S2S_MAX_RESIDUAL:
            continue
        residual_sum_sq += residual ** 2
        n_w = R_prev @ normal
        d_w = d - float(n_w @ t_prev)
        planes_world.append(np.array([n_w[0], n_w[1], n_w[2], d_w]))
        J[valid_count, 0:3] = normal
        # Option B (left-perturbation on R_rel, delta_theta in body_{k-1}
        # tangent). Skew is of the rotated-only point q = R_rel @ p_cur,
        # NOT of p_in_prev = R_rel @ p_cur + t_rel. See chapter5.tex,
        # Remark [Rotated-only vs. fully transformed point].
        J[valid_count, 3:6] = -normal @ skew(R_rel @ p_cur)
        valid_count += 1

    if valid_count < S2S_MIN_VALID_POINTS:
        return np.zeros((6, 6)), {"valid_count": valid_count, "planes_world": []}

    R_s2s = residual_sum_sq / (valid_count - 6)
    J_valid = J[:valid_count]
    FIM = J_valid.T @ J_valid
    eigvals, eigvecs = np.linalg.eigh(FIM)
    inv_eigvals = np.where(eigvals > S2S_MIN_EIGENVALUE, 1.0 / eigvals, 0.0)
    P_rel = R_s2s * (eigvecs @ np.diag(inv_eigvals) @ eigvecs.T)
    return P_rel, {"valid_count": valid_count, "R_s2s": R_s2s, "planes_world": planes_world}


def run_msc_eskf(imu_data, lidar_data):
    R0, p0, v0, _, _ = trajectory_at(0.0)
    state = MSCState(R0, p0.copy(), v0.copy(),
                     INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P = np.diag([
        0.02, 0.02, 0.02,
        0.05, 0.05, 0.05,
        0.05, 0.05, 0.05,
        0.001, 0.001, 0.001,
        0.005, 0.005, 0.005,
    ]) ** 2

    P_drift_accum = np.zeros((6, 6))
    wmap = WindowedMap()

    prev_scan_points_crlb = None
    prev_tree_crlb = None
    R_prev_crlb = None
    t_prev_crlb = None
    P_drift_crlb = np.zeros((6, 6))
    r_s2s_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    p_rel_trace_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    plane_history = deque(maxlen=S2S_PERSIST_HISTORY)

    history = {
        "t": [], "p_est": [], "p_true": [],
        "msc_drift_pos_trace": [], "msc_drift_rot_trace": [],
        "crlb_drift_pos_trace": [], "crlb_drift_rot_trace": [],
        "filter_pos_var": [], "filter_rot_var": [],
        "pos_err": [], "rot_err": [],
        "nees_pos_msc": [], "nees_rot_msc": [],
        "nees_pos_crlb": [], "nees_rot_crlb": [],
        "n_window": [], "n_matches": [],
    }

    lidar_idx = 0
    scan_count = 0
    for sample in imu_data:
        t = sample["t"]
        gyro = sample["gyro"]
        acc = sample["acc"]

        state, P = predict(state, P, gyro, acc, DT_IMU)

        while lidar_idx < len(lidar_data) and lidar_data[lidar_idx]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[lidar_idx]
            if len(scan["points_body"]) > 0:
                state, P = clone_pose(state, P, scan["points_body"])
                scan_count += 1
                k_idx = state.n_window - 1

                n_matches_total = 0
                if state.n_window >= 2:
                    all_matches = []
                    for offset in range(1, min(MATCH_PREV_SCANS + 1, state.n_window)):
                        j_idx = k_idx - offset
                        matches = find_msc_correspondences(
                            state, scan["points_body"], k_idx, j_idx)
                        all_matches.extend(matches)
                    if all_matches:
                        state, P = msc_update(state, P, all_matches)
                        n_matches_total = len(all_matches)

                while state.n_window > WINDOW_SIZE:
                    state, P, P_drift_accum = marginalize_oldest(state, P, P_drift_accum)

                if prev_tree_crlb is not None:
                    t_rel_gate = np.linalg.norm(state.p - t_prev_crlb)
                    rot_rel_gate = np.linalg.norm(rodrigues_log(R_prev_crlb.T @ state.R))
                    if t_rel_gate > S2S_MIN_TRANS_M or rot_rel_gate > S2S_MIN_ROT_RAD:
                        P_rel, s2s_debug = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan_points_crlb, prev_tree_crlb, R_prev_crlb, t_prev_crlb)
                        r_s2s_val = s2s_debug.get("R_s2s", 0.0)
                        p_rel_trace = np.trace(P_rel)

                        reject = False
                        if len(r_s2s_window) >= 5:
                            r_med = float(np.median(r_s2s_window))
                            if r_s2s_val > S2S_ADAPTIVE_REJECT_RATIO * max(r_med, 1e-12):
                                reject = True
                        if not reject and len(p_rel_trace_window) >= 5:
                            p_med = float(np.median(p_rel_trace_window))
                            if p_rel_trace > S2S_ADAPTIVE_REJECT_RATIO * max(p_med, 1e-12):
                                reject = True

                        curr_planes = s2s_debug.get("planes_world", [])
                        persist_frac = 0.0
                        if plane_history and curr_planes:
                            hits = 0
                            for pc in curr_planes:
                                nc = pc[0:3]
                                dc = pc[3]
                                matched = False
                                for old_set in plane_history:
                                    for pp in old_set:
                                        dot = float(nc @ pp[0:3])
                                        if abs(dot) < S2S_PERSIST_NORMAL_TAU:
                                            continue
                                        dp_signed = pp[3] if dot >= 0 else -pp[3]
                                        if abs(dc - dp_signed) < S2S_PERSIST_DIST_EPS:
                                            matched = True
                                            break
                                    if matched:
                                        break
                                if matched:
                                    hits += 1
                            persist_frac = hits / len(curr_planes)
                        persist_scale = max(0.0, 1.0 - S2S_PERSIST_ALPHA * persist_frac)

                        if not reject:
                            Adj_prev = np.zeros((6, 6))
                            Adj_prev[0:3, 0:3] = R_prev_crlb
                            Adj_prev[3:6, 3:6] = R_prev_crlb
                            P_rel_w = Adj_prev @ P_rel @ Adj_prev.T
                            P_drift_crlb += persist_scale * P_rel_w
                            if r_s2s_val > 0:
                                r_s2s_window.append(r_s2s_val)
                            if p_rel_trace > 0:
                                p_rel_trace_window.append(p_rel_trace)
                            if curr_planes:
                                plane_history.append(curr_planes)

                prev_scan_points_crlb = scan["points_body"].copy()
                prev_tree_crlb = cKDTree(prev_scan_points_crlb)
                R_prev_crlb = state.R.copy()
                t_prev_crlb = state.p.copy()

                history["n_matches"].append(n_matches_total)

            lidar_idx += 1

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        pos_err = state.p - p_true

        P_window_rel = extract_window_drift(state, P)
        P_drift_msc_total = P_drift_accum + P_window_rel

        P_msc_pos = P_drift_msc_total[3:6, 3:6]
        P_msc_rot = P_drift_msc_total[0:3, 0:3]
        try:
            nees_p_msc = float(pos_err @ np.linalg.solve(P_msc_pos, pos_err))
        except np.linalg.LinAlgError:
            nees_p_msc = np.nan

        try:
            nees_r_msc = float(rot_err @ np.linalg.solve(P_msc_rot, rot_err))
        except np.linalg.LinAlgError:
            nees_r_msc = np.nan

        Adj_curT = np.zeros((6, 6))
        Adj_curT[0:3, 0:3] = state.R.T
        Adj_curT[3:6, 3:6] = state.R.T
        P_drift_crlb_body = Adj_curT @ P_drift_crlb @ Adj_curT.T

        imu_pos_var = np.trace(P[3:6, 3:6])
        imu_rot_var = np.trace(P[0:3, 0:3])

        P_pub_crlb_pos = P[3:6, 3:6] + P_drift_crlb_body[0:3, 0:3]
        P_pub_crlb_rot = P[0:3, 0:3] + P_drift_crlb_body[3:6, 3:6]
        try:
            nees_p_crlb = float(pos_err @ np.linalg.solve(P_pub_crlb_pos, pos_err))
        except np.linalg.LinAlgError:
            nees_p_crlb = np.nan
        try:
            nees_r_crlb = float(rot_err @ np.linalg.solve(P_pub_crlb_rot, rot_err))
        except np.linalg.LinAlgError:
            nees_r_crlb = np.nan

        history["t"].append(t)
        history["p_est"].append(state.p.copy())
        history["p_true"].append(p_true.copy())
        history["pos_err"].append(pos_err.copy())
        history["rot_err"].append(rot_err.copy())
        history["filter_pos_var"].append(imu_pos_var)
        history["filter_rot_var"].append(imu_rot_var)
        history["msc_drift_pos_trace"].append(np.trace(P_drift_msc_total[3:6, 3:6]))
        history["msc_drift_rot_trace"].append(np.trace(P_drift_msc_total[0:3, 0:3]))
        history["crlb_drift_pos_trace"].append(np.trace(P_drift_crlb[0:3, 0:3]))
        history["crlb_drift_rot_trace"].append(np.trace(P_drift_crlb[3:6, 3:6]))
        history["nees_pos_msc"].append(nees_p_msc)
        history["nees_rot_msc"].append(nees_r_msc)
        history["nees_pos_crlb"].append(nees_p_crlb)
        history["nees_rot_crlb"].append(nees_r_crlb)
        history["n_window"].append(state.n_window)

    for key in history:
        history[key] = np.array(history[key])
    return history


def plot_results(hist):
    t = hist["t"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(hist["p_true"][:, 0], hist["p_true"][:, 1], "k-", linewidth=2.0, label="ground truth")
    ax.plot(hist["p_est"][:, 0], hist["p_est"][:, 1], "r-", linewidth=1.0, alpha=0.7, label="MSC-ESKF")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, hist["msc_drift_pos_trace"], "b-", linewidth=1.5, label="MSC-ESKF drift")
    ax.plot(t, hist["crlb_drift_pos_trace"], "r--", linewidth=1.5, label="CRLB drift")
    ax.plot(t, hist["filter_pos_var"], "k:", linewidth=1.0, label="filter P (IMU block)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trace (m²)")
    ax.set_title("Position drift covariance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, hist["msc_drift_rot_trace"], "b-", linewidth=1.5, label="MSC-ESKF drift")
    ax.plot(t, hist["crlb_drift_rot_trace"], "r--", linewidth=1.5, label="CRLB drift")
    ax.plot(t, hist["filter_rot_var"], "k:", linewidth=1.0, label="filter P (IMU block)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trace (rad²)")
    ax.set_title("Rotation drift covariance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    ax.plot(t, err_norm, "k-", linewidth=1.0, label="actual error")
    ax.plot(t, np.sqrt(hist["filter_pos_var"]), "k:", linewidth=1.0, label="filter std")
    ax.plot(t, np.sqrt(np.maximum(hist["msc_drift_pos_trace"], 0)), "b-", linewidth=1.0, label="MSC drift std")
    P_crlb_total = hist["filter_pos_var"] + hist["crlb_drift_pos_trace"]
    ax.plot(t, np.sqrt(P_crlb_total), "r--", linewidth=1.0, label="CRLB total std")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Position error vs uncertainty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(t, hist["nees_pos_msc"], "b-", linewidth=0.8, alpha=0.7, label="MSC-ESKF")
    ax.plot(t, hist["nees_pos_crlb"], "r-", linewidth=0.8, alpha=0.7, label="CRLB")
    ax.axhline(3.0, color="k", linestyle="-", linewidth=1.0, label="expected (3)")
    ax.axhline(9.348, color="gray", linestyle="--", linewidth=0.8, label="95% upper")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("NEES (3-DOF)")
    ax.set_title("Position NEES")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    n_matches = hist["n_matches"]
    scan_times = np.linspace(t[0], t[-1], len(n_matches)) if len(n_matches) > 0 else []
    if len(n_matches) > 0:
        ax.plot(scan_times, n_matches, "g-", linewidth=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("correspondences")
    ax.set_title("Multi-state constraint matches per scan")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_msc_eskf.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_msc_eskf.png")


def print_summary(hist):
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    print(f"\n[IMSC-ESKF Results]")
    print(f"  Position error: mean={np.mean(err_norm):.4f} m, max={np.max(err_norm):.4f} m, "
          f"final={err_norm[-1]:.4f} m")
    print(f"  Final filter pos var (trace): {hist['filter_pos_var'][-1]:.6f} m²")
    print(f"  Final MSC drift pos (trace): {hist['msc_drift_pos_trace'][-1]:.6f} m²")
    print(f"  Final CRLB drift pos (trace): {hist['crlb_drift_pos_trace'][-1]:.6f} m²")
    print(f"  Final MSC drift rot (trace): {hist['msc_drift_rot_trace'][-1]:.6f} rad²")
    print(f"  Final CRLB drift rot (trace): {hist['crlb_drift_rot_trace'][-1]:.6f} rad²")

    n_matches = hist["n_matches"]
    if len(n_matches) > 0:
        print(f"  Matches per scan: mean={np.mean(n_matches):.0f}, "
              f"min={np.min(n_matches):.0f}, max={np.max(n_matches):.0f}")

    for label, key in [("pos (msc)", "nees_pos_msc"), ("rot (msc)", "nees_rot_msc"),
                       ("pos (crlb)", "nees_pos_crlb"), ("rot (crlb)", "nees_rot_crlb")]:
        vals = hist[key]
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        print(f"  NEES {label}: mean={finite.mean():.3f}  median={np.median(finite):.3f}")


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

imu_data = [{"t": row[0], "gyro": row[1:4], "acc": row[4:7]} for row in imu_arr]
lidar_data = list(lidar_arr)

print(f"Running IMSC-ESKF (window={WINDOW_SIZE}, match_prev={MATCH_PREV_SCANS})...")

all_nees_msc = []
all_nees_crlb = []
for seed in [123, 456, 789, 42, 99]:
    np.random.seed(seed)
    hist = run_msc_eskf(imu_data, lidar_data)
    finite_msc = hist["nees_pos_msc"][np.isfinite(hist["nees_pos_msc"])]
    finite_crlb = hist["nees_pos_crlb"][np.isfinite(hist["nees_pos_crlb"])]
    mid = len(finite_msc) // 4
    all_nees_msc.append(np.median(finite_msc[mid:]))
    all_nees_crlb.append(np.median(finite_crlb[mid:]))
    err = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    print(f"  seed={seed}: err_mean={np.mean(err):.3f}m, "
          f"MSC_nees_med={np.median(finite_msc[mid:]):.1f}, "
          f"CRLB_nees_med={np.median(finite_crlb[mid:]):.1f}, "
          f"MSC_drift={hist['msc_drift_pos_trace'][-1]:.6f}, "
          f"CRLB_drift={hist['crlb_drift_pos_trace'][-1]:.6f}")

print(f"\nAcross seeds:")
print(f"  MSC NEES median of medians: {np.median(all_nees_msc):.1f}")
print(f"  CRLB NEES median of medians: {np.median(all_nees_crlb):.1f}")

np.random.seed(123)
hist = run_msc_eskf(imu_data, lidar_data)
print_summary(hist)
plot_results(hist)

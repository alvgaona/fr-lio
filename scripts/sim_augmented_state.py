"""Origin-augmented IESKF: drift covariance from the filter itself.

Augments the standard 15-DOF state (R, p, v, bg, ba) with a 6-DOF
origin anchor (R0, p0) that is never observed and never moves. The
cross-covariance between the current pose and the origin grows over
time, and the covariance of the relative pose (current w.r.t. origin)
is the filter's own estimate of accumulated drift.

This is compared against the scan-to-scan CRLB approach from
sim_iekf_3d.py to validate that the augmented state produces
consistent (and potentially richer) drift estimates.

State layout (21 DOF tangent):
  [0:3]   dtheta   - current rotation (SO(3) right perturbation)
  [3:6]   dp       - current position
  [6:9]   dv       - current velocity
  [9:12]  dbg      - gyro bias
  [12:15] dba      - accel bias
  [15:18] dtheta0  - origin rotation
  [18:21] dp0      - origin position

Run: python sim_augmented_state.py
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
np.random.seed(123)

DT_IMU = 1.0 / IMU_RATE
N_CURRENT = 15
N_ORIGIN = 6
N_AUG = N_CURRENT + N_ORIGIN  # 21

DT_IMU_REF = 1.0 / IMU_RATE
GYR_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
ACC_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
GYR_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)
ACC_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)

LIDAR_POINT_VAR = 0.001
MAX_PLANE_DIST = 0.5
MAX_POINTS_PER_SCAN = 1000
MAX_IEKF_ITERS = 3
CONVERGENCE_THRESH = 1e-4

NN_K = 5
PLANE_FIT_THRESH = 0.05
MAP_VOXEL_SIZE = 0.2

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


class AugState:
    def __init__(self, R, p, v, bg, ba, R0, p0):
        self.R = R.copy()
        self.p = p.copy()
        self.v = v.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()
        self.R0 = R0.copy()
        self.p0 = p0.copy()

    def copy(self):
        return AugState(self.R, self.p, self.v, self.bg, self.ba, self.R0, self.p0)

    def boxplus(self, dx):
        R_new = self.R @ rodrigues_exp(dx[0:3])
        p_new = self.p + dx[3:6]
        v_new = self.v + dx[6:9]
        bg_new = self.bg + dx[9:12]
        ba_new = self.ba + dx[12:15]
        R0_new = self.R0 @ rodrigues_exp(dx[15:18])
        p0_new = self.p0 + dx[18:21]
        return AugState(R_new, p_new, v_new, bg_new, ba_new, R0_new, p0_new)

    def boxminus(self, other):
        dtheta = rodrigues_log(other.R.T @ self.R)
        dp = self.p - other.p
        dv = self.v - other.v
        dbg = self.bg - other.bg
        dba = self.ba - other.ba
        dtheta0 = rodrigues_log(other.R0.T @ self.R0)
        dp0 = self.p0 - other.p0
        return np.concatenate([dtheta, dp, dv, dbg, dba, dtheta0, dp0])


def predict(state, P, gyro, acc, dt):
    omega = gyro - state.bg
    a_body = acc - state.ba
    a_world = state.R @ a_body + GRAVITY

    new_R = state.R @ rodrigues_exp(omega * dt)
    new_p = state.p + state.v * dt + 0.5 * a_world * dt * dt
    new_v = state.v + a_world * dt
    new_state = AugState(new_R, new_p, new_v, state.bg, state.ba, state.R0, state.p0)

    F = np.eye(N_AUG)
    F[0:3, 0:3] = rodrigues_exp(-omega * dt)
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    F[6:9, 0:3] = -state.R @ skew(a_body) * dt
    F[6:9, 12:15] = -state.R * dt
    # Origin block [15:21, 15:21] stays identity (no dynamics)

    G = np.zeros((N_AUG, 12))
    G[0:3, 0:3] = -np.eye(3) * dt
    G[6:9, 3:6] = -state.R * dt
    G[9:12, 6:9] = np.eye(3) * dt
    G[12:15, 9:12] = np.eye(3) * dt
    # Origin rows of G are zero (no noise drives the origin)

    Q_continuous = np.diag([
        GYR_NOISE_STD ** 2, GYR_NOISE_STD ** 2, GYR_NOISE_STD ** 2,
        ACC_NOISE_STD ** 2, ACC_NOISE_STD ** 2, ACC_NOISE_STD ** 2,
        GYR_BIAS_WALK_STD ** 2, GYR_BIAS_WALK_STD ** 2, GYR_BIAS_WALK_STD ** 2,
        ACC_BIAS_WALK_STD ** 2, ACC_BIAS_WALK_STD ** 2, ACC_BIAS_WALK_STD ** 2,
    ])
    Q = G @ Q_continuous @ G.T

    new_P = F @ P @ F.T + Q
    return new_state, new_P


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


def find_correspondences(state, points_body, online_map):
    online_map.build_tree()
    if online_map.tree is None or len(online_map) < NN_K:
        return []
    matches = []
    map_points = np.array(online_map.points)
    for p_body in points_body:
        p_world = state.R @ p_body + state.p
        dists, idxs = online_map.tree.query(p_world, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue
        neighbors = map_points[idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        normal, d = plane
        if abs(np.dot(normal, p_world) + d) > MAX_PLANE_DIST:
            continue
        matches.append((p_body, normal, d))
    return matches


def compute_residuals_and_H(state, matches):
    n_meas = len(matches)
    h = np.zeros(n_meas)
    H = np.zeros((n_meas, N_AUG))
    for i, (p_body, n_w, d_w) in enumerate(matches):
        p_world = state.R @ p_body + state.p
        h[i] = np.dot(n_w, p_world) + d_w
        H[i, 0:3] = -n_w @ state.R @ skew(p_body)
        H[i, 3:6] = n_w
        # H[i, 15:21] = 0  (origin is not observed — already zero)
    return h, H


def iekf_update(state_prior, P_prior, points_body, online_map):
    matches = find_correspondences(state_prior, points_body, online_map)
    if len(matches) < 5:
        return state_prior, P_prior

    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]

    R_meas = LIDAR_POINT_VAR * np.eye(len(matches))
    state = state_prior.copy()

    H_final = None
    K_final = None
    for j in range(MAX_IEKF_ITERS):
        h, H = compute_residuals_and_H(state, matches)
        S = H @ P_prior @ H.T + R_meas
        K = P_prior @ H.T @ np.linalg.inv(S)

        dx_from_prior = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx_from_prior)
        state = state_prior.boxplus(dx_from_prior + delta)

        H_final = H
        K_final = K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    P_new = (np.eye(N_AUG) - K_final @ H_final) @ P_prior
    return state, P_new


def extract_relative_cov(P):
    """Extract the covariance of the relative pose (current w.r.t. origin).

    Relative position: dp_rel = p - p0
    Relative rotation: dtheta_rel = Log(R0^T R)

    The Jacobian of the relative pose w.r.t. the augmented state is:
      J_rel = [I_3  0  0  0  0  -I_3  0 ]   (rotation)
              [0    I  0  0  0   0   -I ]   (position)

    P_rel = J_rel @ P @ J_rel^T
    """
    J = np.zeros((6, N_AUG))
    # Relative rotation: dtheta_rel = dtheta - dtheta0
    J[0:3, 0:3] = np.eye(3)
    J[0:3, 15:18] = -np.eye(3)
    # Relative position: dp_rel = dp - dp0
    J[3:6, 3:6] = np.eye(3)
    J[3:6, 18:21] = -np.eye(3)
    return J @ P @ J.T


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


def run_filter(imu_data, lidar_data):
    from collections import deque

    R0_true, p0_true, v0, _, _ = trajectory_at(0.0)
    state = AugState(
        R=R0_true, p=p0_true.copy(), v=v0.copy(),
        bg=INITIAL_GYR_BIAS.copy(), ba=INITIAL_ACC_BIAS.copy(),
        R0=R0_true.copy(), p0=p0_true.copy(),
    )

    P = np.zeros((N_AUG, N_AUG))
    P[0:15, 0:15] = np.diag([
        0.02, 0.02, 0.02,
        0.05, 0.05, 0.05,
        0.05, 0.05, 0.05,
        0.001, 0.001, 0.001,
        0.005, 0.005, 0.005,
    ]) ** 2
    P[15:18, 15:18] = np.diag([0.02, 0.02, 0.02]) ** 2
    P[18:21, 18:21] = np.diag([0.05, 0.05, 0.05]) ** 2
    # Initial cross-covariance: origin and current start identical
    P[0:3, 15:18] = P[0:3, 0:3].copy()
    P[15:18, 0:3] = P[0:3, 0:3].copy()
    P[3:6, 18:21] = P[3:6, 3:6].copy()
    P[18:21, 3:6] = P[3:6, 3:6].copy()

    online_map = OnlineMap()

    prev_scan_points = None
    prev_tree = None
    R_prev_s2s = None
    t_prev_s2s = None
    P_drift_crlb = np.zeros((6, 6))

    r_s2s_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    p_rel_trace_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    plane_history = deque(maxlen=S2S_PERSIST_HISTORY)

    history = {
        "t": [], "p_est": [], "p_true": [],
        "aug_drift_pos_trace": [], "aug_drift_rot_trace": [],
        "crlb_drift_pos_trace": [], "crlb_drift_rot_trace": [],
        "filter_pos_var": [], "filter_rot_var": [],
        "pos_err": [], "rot_err": [],
        "nees_pos_aug": [], "nees_rot_aug": [],
        "nees_pos_crlb": [], "nees_rot_crlb": [],
        "origin_pos_var": [], "origin_rot_var": [],
        "cross_cov_pos_trace": [], "cross_cov_rot_trace": [],
    }

    lidar_idx = 0
    for sample in imu_data:
        t = sample["t"]
        gyro = sample["gyro"]
        acc = sample["acc"]

        state, P = predict(state, P, gyro, acc, DT_IMU)

        while lidar_idx < len(lidar_data) and lidar_data[lidar_idx]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[lidar_idx]
            if len(scan["points_body"]) > 0:
                if len(online_map) >= NN_K:
                    state, P = iekf_update(state, P, scan["points_body"], online_map)

                if prev_tree is not None:
                    t_rel_gate = np.linalg.norm(state.p - t_prev_s2s)
                    rot_rel_gate = np.linalg.norm(rodrigues_log(R_prev_s2s.T @ state.R))
                    if t_rel_gate > S2S_MIN_TRANS_M or rot_rel_gate > S2S_MIN_ROT_RAD:
                        P_rel, s2s_debug = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan_points, prev_tree, R_prev_s2s, t_prev_s2s,
                        )
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
                            Adj_prev[0:3, 0:3] = R_prev_s2s
                            Adj_prev[3:6, 3:6] = R_prev_s2s
                            P_rel_w = Adj_prev @ P_rel @ Adj_prev.T
                            P_drift_crlb += persist_scale * P_rel_w
                            if r_s2s_val > 0:
                                r_s2s_window.append(r_s2s_val)
                            if p_rel_trace > 0:
                                p_rel_trace_window.append(p_rel_trace)
                            if curr_planes:
                                plane_history.append(curr_planes)

                prev_scan_points = scan["points_body"].copy()
                prev_tree = cKDTree(prev_scan_points)
                R_prev_s2s = state.R.copy()
                t_prev_s2s = state.p.copy()

                world_points = (state.R @ scan["points_body"].T).T + state.p
                online_map.add_points(world_points)
            lidar_idx += 1

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        pos_err = state.p - p_true

        P_rel_aug = extract_relative_cov(P)

        # NEES for augmented approach: use relative covariance
        # The "drift error" is the error in (current pose relative to origin)
        # vs (true current pose relative to true origin).
        # Since origin is initialized at truth, drift error = pose error.
        P_pos_aug = P_rel_aug[3:6, 3:6]
        P_rot_aug = P_rel_aug[0:3, 0:3]
        try:
            nees_p_aug = float(pos_err @ np.linalg.solve(P_pos_aug, pos_err))
        except np.linalg.LinAlgError:
            nees_p_aug = np.nan
        try:
            nees_r_aug = float(rot_err @ np.linalg.solve(P_rot_aug, rot_err))
        except np.linalg.LinAlgError:
            nees_r_aug = np.nan

        # NEES for CRLB approach: P_filter + P_drift
        Adj_curT = np.zeros((6, 6))
        Adj_curT[0:3, 0:3] = state.R.T
        Adj_curT[3:6, 3:6] = state.R.T
        P_drift_body = Adj_curT @ P_drift_crlb @ Adj_curT.T

        P_pub_crlb = np.zeros((6, 6))
        P_pub_crlb[0:3, 0:3] = P[0:3, 0:3] + P_drift_body[3:6, 3:6]
        P_pub_crlb[3:6, 3:6] = P[3:6, 3:6] + P_drift_body[0:3, 0:3]
        try:
            nees_p_crlb = float(pos_err @ np.linalg.solve(P_pub_crlb[3:6, 3:6], pos_err))
        except np.linalg.LinAlgError:
            nees_p_crlb = np.nan
        try:
            nees_r_crlb = float(rot_err @ np.linalg.solve(P_pub_crlb[0:3, 0:3], rot_err))
        except np.linalg.LinAlgError:
            nees_r_crlb = np.nan

        history["t"].append(t)
        history["p_est"].append(state.p.copy())
        history["p_true"].append(p_true.copy())
        history["pos_err"].append(pos_err.copy())
        history["rot_err"].append(rot_err.copy())
        history["filter_pos_var"].append(np.trace(P[3:6, 3:6]))
        history["filter_rot_var"].append(np.trace(P[0:3, 0:3]))
        history["aug_drift_pos_trace"].append(np.trace(P_rel_aug[3:6, 3:6]))
        history["aug_drift_rot_trace"].append(np.trace(P_rel_aug[0:3, 0:3]))
        history["crlb_drift_pos_trace"].append(np.trace(P_drift_crlb[0:3, 0:3]))
        history["crlb_drift_rot_trace"].append(np.trace(P_drift_crlb[3:6, 3:6]))
        history["origin_pos_var"].append(np.trace(P[18:21, 18:21]))
        history["origin_rot_var"].append(np.trace(P[15:18, 15:18]))
        history["cross_cov_pos_trace"].append(np.trace(P[3:6, 18:21]))
        history["cross_cov_rot_trace"].append(np.trace(P[0:3, 15:18]))
        history["nees_pos_aug"].append(nees_p_aug)
        history["nees_rot_aug"].append(nees_r_aug)
        history["nees_pos_crlb"].append(nees_p_crlb)
        history["nees_rot_crlb"].append(nees_r_crlb)

    for key in history:
        if key not in ("p_est", "p_true"):
            history[key] = np.array(history[key])
        else:
            history[key] = np.array(history[key])
    return history


def plot_results(hist):
    t = hist["t"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(hist["p_true"][:, 0], hist["p_true"][:, 1], "k-", linewidth=2.0, label="ground truth")
    ax.plot(hist["p_est"][:, 0], hist["p_est"][:, 1], "r-", linewidth=1.0, alpha=0.7, label="estimate")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Drift covariance comparison: position
    ax = axes[0, 1]
    ax.plot(t, hist["aug_drift_pos_trace"], "b-", linewidth=1.5, label="augmented state")
    ax.plot(t, hist["crlb_drift_pos_trace"], "r--", linewidth=1.5, label="scan-to-scan CRLB")
    ax.plot(t, hist["filter_pos_var"], "k:", linewidth=1.0, label="filter P (local)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trace (m²)")
    ax.set_title("Position drift covariance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Drift covariance comparison: rotation
    ax = axes[0, 2]
    ax.plot(t, hist["aug_drift_rot_trace"], "b-", linewidth=1.5, label="augmented state")
    ax.plot(t, hist["crlb_drift_rot_trace"], "r--", linewidth=1.5, label="scan-to-scan CRLB")
    ax.plot(t, hist["filter_rot_var"], "k:", linewidth=1.0, label="filter P (local)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trace (rad²)")
    ax.set_title("Rotation drift covariance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Position error vs uncertainty envelopes
    ax = axes[1, 0]
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    ax.plot(t, err_norm, "k-", linewidth=1.0, label="actual error")
    ax.plot(t, np.sqrt(hist["filter_pos_var"]), "k:", linewidth=1.0, label="filter std")
    ax.plot(t, np.sqrt(hist["aug_drift_pos_trace"]), "b-", linewidth=1.0, label="augmented std")
    P_crlb_total = hist["filter_pos_var"] + hist["crlb_drift_pos_trace"]
    ax.plot(t, np.sqrt(P_crlb_total), "r--", linewidth=1.0, label="CRLB total std")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Position error vs uncertainty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # 5. Position NEES comparison
    ax = axes[1, 1]
    ax.plot(t, hist["nees_pos_aug"], "b-", linewidth=0.8, alpha=0.7, label="augmented")
    ax.plot(t, hist["nees_pos_crlb"], "r-", linewidth=0.8, alpha=0.7, label="CRLB")
    ax.axhline(3.0, color="k", linestyle="-", linewidth=1.0, label="expected (3)")
    ax.axhline(9.348, color="gray", linestyle="--", linewidth=0.8, label="95% upper")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("NEES (3-DOF)")
    ax.set_title("Position NEES")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6. Covariance decomposition: origin, current, cross
    ax = axes[1, 2]
    ax.plot(t, hist["origin_pos_var"], "g-", linewidth=1.2, label="origin P (pos)")
    ax.plot(t, hist["filter_pos_var"], "r-", linewidth=1.2, label="current P (pos)")
    ax.plot(t, hist["cross_cov_pos_trace"], "m-", linewidth=1.2, label="cross-cov (pos)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("trace")
    ax.set_title("Covariance decomposition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_augmented_state.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_augmented_state.png")


def print_summary(hist):
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    print(f"\n[Augmented State IESKF]")
    print(f"  Position error: mean={np.mean(err_norm):.4f} m, max={np.max(err_norm):.4f} m, "
          f"final={err_norm[-1]:.4f} m")
    print(f"  Final filter pos var (trace): {hist['filter_pos_var'][-1]:.6f} m²")
    print(f"  Final augmented drift pos (trace): {hist['aug_drift_pos_trace'][-1]:.6f} m²")
    print(f"  Final CRLB drift pos (trace): {hist['crlb_drift_pos_trace'][-1]:.6f} m²")
    print(f"  Final augmented drift rot (trace): {hist['aug_drift_rot_trace'][-1]:.6f} rad²")
    print(f"  Final CRLB drift rot (trace): {hist['crlb_drift_rot_trace'][-1]:.6f} rad²")
    print(f"  Final origin pos var (trace): {hist['origin_pos_var'][-1]:.6f} m²")
    print(f"  Final cross-cov pos (trace): {hist['cross_cov_pos_trace'][-1]:.6f}")

    for label, key in [("pos (aug)", "nees_pos_aug"), ("rot (aug)", "nees_rot_aug"),
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

print("Running augmented-state IESKF...")
np.random.seed(123)
hist = run_filter(imu_data, lidar_data)

print_summary(hist)
plot_results(hist)

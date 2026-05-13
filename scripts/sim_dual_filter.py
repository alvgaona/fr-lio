"""Dual-filter architecture: IESKF + 2-pose MSC-ESKF.

Filter 1 (IESKF): scan-to-map with persistent online map for accurate
state estimation. Standard 15-DOF error-state.

Filter 2 (2-pose MSC-ESKF): current + previous pose (21 DOF). Scan-to-scan
multi-state constraint gives per-step relative covariance. Accumulated into
P_drift. Only used for covariance — state estimate comes from filter 1.

Published: IESKF state, covariance = P_filter + P_drift_msc.
Compared against CRLB accumulation on the same data.

Run: pixi run python sim_dual_filter.py
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
N_2POSE = 21

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
S2S_MIN_VALID_POINTS = 100
S2S_MAX_RESIDUAL = 0.3
S2S_MIN_EIGENVALUE = 1e-6
S2S_MIN_TRANS_M = 0.01
S2S_MIN_ROT_RAD = 0.001
S2S_ADAPTIVE_REJECT_RATIO = 10.0
S2S_ADAPTIVE_WINDOW_SIZE = 20
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
        R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
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
    if np.max(np.abs(neighbors @ normal + d)) > PLANE_FIT_THRESH:
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


# ============================================================
# Filter 1: Standard IESKF (scan-to-map)
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
            self.R @ rodrigues_exp(dx[0:3]), self.p + dx[3:6],
            self.v + dx[6:9], self.bg + dx[9:12], self.ba + dx[12:15])

    def boxminus(self, other):
        return np.concatenate([
            rodrigues_log(other.R.T @ self.R),
            self.p - other.p, self.v - other.v,
            self.bg - other.bg, self.ba - other.ba])


def make_F_G(state, gyro, acc, dt, n_dof):
    omega = gyro - state.bg
    a_body = acc - state.ba
    F = np.eye(n_dof)
    F[0:3, 0:3] = rodrigues_exp(-omega * dt)
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    F[6:9, 0:3] = -state.R @ skew(a_body) * dt
    F[6:9, 12:15] = -state.R * dt
    G = np.zeros((n_dof, 12))
    G[0:3, 0:3] = -np.eye(3) * dt
    G[6:9, 3:6] = -state.R * dt
    G[9:12, 6:9] = np.eye(3) * dt
    G[12:15, 9:12] = np.eye(3) * dt
    return F, G


Q_CONT = np.diag(
    [GYR_NOISE_STD**2]*3 + [ACC_NOISE_STD**2]*3 +
    [GYR_BIAS_WALK_STD**2]*3 + [ACC_BIAS_WALK_STD**2]*3)


def imu_propagate(state, gyro, acc, dt):
    omega = gyro - state.bg
    a_body = acc - state.ba
    a_world = state.R @ a_body + GRAVITY
    return State15(
        state.R @ rodrigues_exp(omega * dt),
        state.p + state.v * dt + 0.5 * a_world * dt * dt,
        state.v + a_world * dt, state.bg, state.ba)


def predict15(state, P, gyro, acc, dt):
    new_state = imu_propagate(state, gyro, acc, dt)
    F, G = make_F_G(state, gyro, acc, dt, N_IMU)
    return new_state, F @ P @ F.T + G @ Q_CONT @ G.T


def ieskf_update(state_prior, P_prior, points_body, omap):
    omap.build_tree()
    if omap.tree is None or len(omap) < NN_K:
        return state_prior, P_prior
    map_pts = np.array(omap.points)
    matches = []
    for pb in points_body:
        pw = state_prior.R @ pb + state_prior.p
        dists, idxs = omap.tree.query(pw, k=NN_K)
        if dists[-1] > MAX_PLANE_DIST:
            continue
        plane = fit_plane(map_pts[idxs])
        if plane is None:
            continue
        nw, d = plane
        if abs(np.dot(nw, pw) + d) > MAX_PLANE_DIST:
            continue
        matches.append((pb, nw, d))
    if len(matches) < 5:
        return state_prior, P_prior
    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]
    n = len(matches)
    R_m = LIDAR_POINT_VAR * np.eye(n)
    state = state_prior.copy()
    Hf = Kf = None
    for _ in range(MAX_IEKF_ITERS):
        h = np.zeros(n)
        H = np.zeros((n, N_IMU))
        for i, (pb, nw, dw) in enumerate(matches):
            pw = state.R @ pb + state.p
            h[i] = np.dot(nw, pw) + dw
            H[i, 0:3] = -nw @ state.R @ skew(pb)
            H[i, 3:6] = nw
        K = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R_m)
        dx = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx)
        state = state_prior.boxplus(dx + delta)
        Hf, Kf = H, K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break
    IKH = np.eye(N_IMU) - Kf @ Hf
    P_new = IKH @ P_prior @ IKH.T + Kf @ R_m @ Kf.T
    return state, 0.5 * (P_new + P_new.T)


# ============================================================
# Filter 2: 2-pose MSC-ESKF (scan-to-scan, 21 DOF)
# ============================================================

class State2Pose:
    def __init__(self, R, p, v, bg, ba, R_prev, p_prev):
        self.R = R.copy()
        self.p = p.copy()
        self.v = v.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()
        self.R_prev = R_prev.copy()
        self.p_prev = p_prev.copy()

    def copy(self):
        return State2Pose(self.R, self.p, self.v, self.bg, self.ba,
                          self.R_prev, self.p_prev)

    def boxplus(self, dx):
        return State2Pose(
            self.R @ rodrigues_exp(dx[0:3]), self.p + dx[3:6],
            self.v + dx[6:9], self.bg + dx[9:12], self.ba + dx[12:15],
            self.R_prev @ rodrigues_exp(dx[15:18]), self.p_prev + dx[18:21])

    def boxminus(self, other):
        return np.concatenate([
            rodrigues_log(other.R.T @ self.R),
            self.p - other.p, self.v - other.v,
            self.bg - other.bg, self.ba - other.ba,
            rodrigues_log(other.R_prev.T @ self.R_prev),
            self.p_prev - other.p_prev])


def predict2pose(state, P, gyro, acc, dt):
    new_state = State2Pose(
        state.R @ rodrigues_exp((gyro - state.bg) * dt),
        state.p + state.v * dt + 0.5 * (state.R @ (acc - state.ba) + GRAVITY) * dt * dt,
        state.v + (state.R @ (acc - state.ba) + GRAVITY) * dt,
        state.bg, state.ba, state.R_prev, state.p_prev)
    F, G = make_F_G(state, gyro, acc, dt, N_2POSE)
    return new_state, F @ P @ F.T + G @ Q_CONT @ G.T


def clone_and_replace_prev(state, P, scan_prev):
    """Replace prev pose with current pose. Return new state, P, old scan."""
    new_state = State2Pose(state.R, state.p, state.v, state.bg, state.ba,
                           state.R.copy(), state.p.copy())
    J = np.zeros((N_2POSE, N_2POSE))
    J[:N_IMU, :N_IMU] = np.eye(N_IMU)
    J[15:18, 0:3] = np.eye(3)
    J[18:21, 3:6] = np.eye(3)
    P_new = J @ P @ J.T
    return new_state, P_new


def msc_s2s_update(state_prior, P_prior, scan_curr, scan_prev):
    """Multi-state constraint between current pose (indices 0:6) and
    previous pose (indices 15:21) using scan-to-scan correspondences."""
    if scan_prev is None or len(scan_prev) < NN_K or len(scan_curr) < NN_K:
        return state_prior, P_prior

    R_cur = state_prior.R
    p_cur = state_prior.p
    R_prv = state_prior.R_prev
    p_prv = state_prior.p_prev

    world_prev = (R_prv @ scan_prev.T).T + p_prv
    tree_prev = cKDTree(world_prev)

    n_check = min(len(scan_curr), S2S_MAX_POINTS)
    indices = (np.random.choice(len(scan_curr), n_check, replace=False)
               if len(scan_curr) > n_check else range(len(scan_curr)))

    matches = []
    for idx in indices:
        pb_k = scan_curr[idx]
        pw_k = R_cur @ pb_k + p_cur
        dists, nn_idxs = tree_prev.query(pw_k, k=S2S_NN_K)
        if dists[-1] > S2S_MAX_NN_DIST:
            continue
        neighbors = world_prev[nn_idxs]
        plane = fit_plane(neighbors)
        if plane is None:
            continue
        nw, d = plane
        residual = np.dot(nw, pw_k) + d
        if abs(residual) > S2S_MAX_RESIDUAL:
            continue
        pb_j = scan_prev[nn_idxs[0]]
        matches.append((pb_k, pb_j, nw))

    if len(matches) < 10:
        return state_prior, P_prior

    n_meas = len(matches)
    state = state_prior.copy()
    Hf = Kf = None
    R_scalar = LIDAR_POINT_VAR

    for it in range(MAX_IEKF_ITERS):
        h = np.zeros(n_meas)
        H = np.zeros((n_meas, N_2POSE))
        for i, (pb_k, pb_j, nw) in enumerate(matches):
            Rk = state.R
            pk = state.p
            Rj = state.R_prev
            pj = state.p_prev
            pwk = Rk @ pb_k + pk
            pwj = Rj @ pb_j + pj
            h[i] = np.dot(nw, pwk - pwj)
            H[i, 0:3] = -nw @ Rk @ skew(pb_k)
            H[i, 3:6] = nw
            H[i, 15:18] = nw @ Rj @ skew(pb_j)
            H[i, 18:21] = -nw

        residual_sq = np.dot(h, h)
        dof = max(n_meas - 6, 1)
        R_scalar = max(residual_sq / dof, LIDAR_POINT_VAR)

        R_m = R_scalar * np.eye(n_meas)
        K = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R_m)
        dx = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx)
        state = state_prior.boxplus(dx + delta)
        Hf, Kf = H, K
        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    R_f = R_scalar * np.eye(n_meas)
    IKH = np.eye(N_2POSE) - Kf @ Hf
    P_new = IKH @ P_prior @ IKH.T + Kf @ R_f @ Kf.T
    return state, 0.5 * (P_new + P_new.T)


def extract_rel_cov(P):
    """Cov(current - prev) from the 21x21 P."""
    J = np.zeros((6, N_2POSE))
    J[0:3, 0:3] = np.eye(3)
    J[0:3, 15:18] = -np.eye(3)
    J[3:6, 3:6] = np.eye(3)
    J[3:6, 18:21] = -np.eye(3)
    return J @ P @ J.T


# ============================================================
# CRLB (from sim_iekf_3d.py, for comparison)
# ============================================================

def compute_crlb(pts_curr, R_cur, p_cur, pts_prev, tree_prev, R_prv, p_prv):
    R_rel = R_prv.T @ R_cur
    t_rel = R_prv.T @ (p_cur - p_prv)
    n_pts = min(len(pts_curr), S2S_MAX_POINTS)
    J = np.zeros((n_pts, 6))
    r_sq = 0.0
    valid = 0
    planes_w = []
    for i in range(n_pts):
        pc = pts_curr[i]
        p_in_prev = R_rel @ pc + t_rel
        dists, idxs = tree_prev.query(p_in_prev, k=S2S_NN_K)
        if dists[-1] > S2S_MAX_NN_DIST:
            continue
        plane = fit_plane(pts_prev[idxs])
        if plane is None:
            continue
        n, d = plane
        r = np.dot(n, p_in_prev) + d
        if abs(r) > S2S_MAX_RESIDUAL:
            continue
        r_sq += r * r
        nw = R_prv @ n
        dw = d - float(nw @ p_prv)
        planes_w.append(np.array([nw[0], nw[1], nw[2], dw]))
        J[valid, 0:3] = n
        # Option B (left-perturbation on R_rel, delta_theta in body_{k-1}
        # tangent). Skew is of the rotated-only point q = R_rel @ pc, NOT
        # of p_in_prev = R_rel @ pc + t_rel. See chapter5.tex, Remark
        # [Rotated-only vs. fully transformed point].
        J[valid, 3:6] = -n @ skew(R_rel @ pc)
        valid += 1
    if valid < S2S_MIN_VALID_POINTS:
        return np.zeros((6, 6)), {"planes_world": []}
    R_s2s = r_sq / (valid - 6)
    Jv = J[:valid]
    FIM = Jv.T @ Jv
    ev, evec = np.linalg.eigh(FIM)
    inv_ev = np.where(ev > S2S_MIN_EIGENVALUE, 1.0 / ev, 0.0)
    P_rel = R_s2s * (evec @ np.diag(inv_ev) @ evec.T)
    return P_rel, {"R_s2s": R_s2s, "planes_world": planes_w}


# ============================================================
# Main loop
# ============================================================

def run_dual(imu_data, lidar_data):
    R0, p0, v0, _, _ = trajectory_at(0.0)
    P0 = np.diag([0.02]*3 + [0.05]*3 + [0.05]*3 + [0.001]*3 + [0.005]*3) ** 2

    st1 = State15(R0, p0.copy(), v0.copy(),
                  INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy())
    P1 = P0.copy()
    omap = OnlineMap()

    st2 = State2Pose(R0, p0.copy(), v0.copy(),
                     INITIAL_GYR_BIAS.copy(), INITIAL_ACC_BIAS.copy(),
                     R0.copy(), p0.copy())
    P2 = np.zeros((N_2POSE, N_2POSE))
    P2[:N_IMU, :N_IMU] = P0.copy()
    P2[15:18, 15:18] = P0[0:3, 0:3]
    P2[18:21, 18:21] = P0[3:6, 3:6]
    P2[0:3, 15:18] = P0[0:3, 0:3]
    P2[15:18, 0:3] = P0[0:3, 0:3]
    P2[3:6, 18:21] = P0[3:6, 3:6]
    P2[18:21, 3:6] = P0[3:6, 3:6]

    P_drift_msc = np.zeros((6, 6))
    P_drift_crlb = np.zeros((6, 6))

    prev_scan = None
    prev_scan_crlb = None
    prev_tree_crlb = None
    R_prev_crlb = None
    p_prev_crlb = None
    scan_count = 0
    has_prev_msc = False

    r_s2s_window = deque(maxlen=S2S_ADAPTIVE_WINDOW_SIZE)
    p_rel_trace_window = deque(maxlen=S2S_ADAPTIVE_WINDOW_SIZE)
    plane_history = deque(maxlen=S2S_PERSIST_HISTORY)

    hist = {
        "t": [], "p_est": [], "p_true": [], "pos_err": [],
        "msc_drift_pos": [], "msc_drift_rot": [],
        "crlb_drift_pos": [], "crlb_drift_rot": [],
        "filter_pos_var": [],
        "nees_pos_msc": [], "nees_rot_msc": [],
        "nees_pos_crlb": [], "nees_rot_crlb": [],
    }

    li = 0
    for sample in imu_data:
        t = sample["t"]
        gyro, acc = sample["gyro"], sample["acc"]

        st1, P1 = predict15(st1, P1, gyro, acc, DT_IMU)
        st2, P2 = predict2pose(st2, P2, gyro, acc, DT_IMU)

        while li < len(lidar_data) and lidar_data[li]["t"] <= t + DT_IMU / 2:
            scan = lidar_data[li]
            if len(scan["points_body"]) > 0:
                scan_count += 1

                if len(omap) >= NN_K:
                    st1, P1 = ieskf_update(st1, P1, scan["points_body"], omap)
                wp = (st1.R @ scan["points_body"].T).T + st1.p
                omap.add_points(wp)

                if has_prev_msc:
                    st2, P2 = msc_s2s_update(st2, P2, scan["points_body"], prev_scan)

                    P_rel_msc = extract_rel_cov(P2)
                    R_oldest = st2.R_prev
                    Adj = np.zeros((6, 6))
                    Adj[0:3, 0:3] = R_oldest
                    Adj[3:6, 3:6] = R_oldest
                    P_rel_w = Adj @ P_rel_msc @ Adj.T
                    P_drift_msc += P_rel_w

                st2, P2 = clone_and_replace_prev(st2, P2, scan["points_body"])
                st2.R = st1.R.copy()
                st2.p = st1.p.copy()
                st2.v = st1.v.copy()
                st2.bg = st1.bg.copy()
                st2.ba = st1.ba.copy()

                prev_scan = scan["points_body"].copy()
                has_prev_msc = True

                if prev_tree_crlb is not None:
                    tg = np.linalg.norm(st1.p - p_prev_crlb)
                    rg = np.linalg.norm(rodrigues_log(R_prev_crlb.T @ st1.R))
                    if tg > S2S_MIN_TRANS_M or rg > S2S_MIN_ROT_RAD:
                        P_rel_c, dbg = compute_crlb(
                            scan["points_body"], st1.R, st1.p,
                            prev_scan_crlb, prev_tree_crlb, R_prev_crlb, p_prev_crlb)
                        r_val = dbg.get("R_s2s", 0.0)
                        ptr = np.trace(P_rel_c)

                        reject = False
                        if len(r_s2s_window) >= 5:
                            rm = float(np.median(r_s2s_window))
                            if r_val > S2S_ADAPTIVE_REJECT_RATIO * max(rm, 1e-12):
                                reject = True
                        if not reject and len(p_rel_trace_window) >= 5:
                            pm = float(np.median(p_rel_trace_window))
                            if ptr > S2S_ADAPTIVE_REJECT_RATIO * max(pm, 1e-12):
                                reject = True

                        curr_planes = dbg.get("planes_world", [])
                        pf = 0.0
                        if plane_history and curr_planes:
                            hits = 0
                            for pc in curr_planes:
                                nc = pc[0:3]; dc = pc[3]
                                matched = False
                                for old_set in plane_history:
                                    for pp in old_set:
                                        dot = float(nc @ pp[0:3])
                                        if abs(dot) < S2S_PERSIST_NORMAL_TAU:
                                            continue
                                        dp_s = pp[3] if dot >= 0 else -pp[3]
                                        if abs(dc - dp_s) < S2S_PERSIST_DIST_EPS:
                                            matched = True; break
                                    if matched: break
                                if matched: hits += 1
                            pf = hits / len(curr_planes)
                        ps = max(0.0, 1.0 - S2S_PERSIST_ALPHA * pf)

                        if not reject:
                            Adj_p = np.zeros((6, 6))
                            Adj_p[0:3, 0:3] = R_prev_crlb
                            Adj_p[3:6, 3:6] = R_prev_crlb
                            P_drift_crlb += ps * (Adj_p @ P_rel_c @ Adj_p.T)
                            if r_val > 0: r_s2s_window.append(r_val)
                            if ptr > 0: p_rel_trace_window.append(ptr)
                            if curr_planes: plane_history.append(curr_planes)

                prev_scan_crlb = scan["points_body"].copy()
                prev_tree_crlb = cKDTree(prev_scan_crlb)
                R_prev_crlb = st1.R.copy()
                p_prev_crlb = st1.p.copy()

            li += 1

        _, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(trajectory_at(t)[0].T @ st1.R)
        pos_err = st1.p - p_true

        Adj_T = np.zeros((6, 6))
        Adj_T[0:3, 0:3] = st1.R.T
        Adj_T[3:6, 3:6] = st1.R.T

        P_msc_body = Adj_T @ P_drift_msc @ Adj_T.T
        P_pub_msc_pos = P1[3:6, 3:6] + P_msc_body[0:3, 0:3]
        P_pub_msc_rot = P1[0:3, 0:3] + P_msc_body[3:6, 3:6]

        P_crlb_body = Adj_T @ P_drift_crlb @ Adj_T.T
        P_pub_crlb_pos = P1[3:6, 3:6] + P_crlb_body[0:3, 0:3]
        P_pub_crlb_rot = P1[0:3, 0:3] + P_crlb_body[3:6, 3:6]

        def safe_nees(err, Pcov):
            try:
                return float(err @ np.linalg.solve(Pcov, err))
            except np.linalg.LinAlgError:
                return np.nan

        hist["t"].append(t)
        hist["p_est"].append(st1.p.copy())
        hist["p_true"].append(p_true.copy())
        hist["pos_err"].append(np.linalg.norm(pos_err))
        hist["filter_pos_var"].append(np.trace(P1[3:6, 3:6]))
        hist["msc_drift_pos"].append(np.trace(P_drift_msc[0:3, 0:3]))
        hist["msc_drift_rot"].append(np.trace(P_drift_msc[3:6, 3:6]))
        hist["crlb_drift_pos"].append(np.trace(P_drift_crlb[0:3, 0:3]))
        hist["crlb_drift_rot"].append(np.trace(P_drift_crlb[3:6, 3:6]))
        hist["nees_pos_msc"].append(safe_nees(pos_err, P_pub_msc_pos))
        hist["nees_rot_msc"].append(safe_nees(rot_err, P_pub_msc_rot))
        hist["nees_pos_crlb"].append(safe_nees(pos_err, P_pub_crlb_pos))
        hist["nees_rot_crlb"].append(safe_nees(rot_err, P_pub_crlb_rot))

    for k in hist:
        hist[k] = np.array(hist[k])
    return hist


def plot_results(hist):
    t = hist["t"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(hist["p_true"][:, 0], hist["p_true"][:, 1], "k-", lw=2, label="truth")
    ax.plot(hist["p_est"][:, 0], hist["p_est"][:, 1], "r-", lw=1, alpha=0.8, label="IESKF")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title("Trajectory (IESKF state)")

    ax = axes[0, 1]
    ax.plot(t, hist["msc_drift_pos"], "b-", lw=1.5, label="2-pose MSC")
    ax.plot(t, hist["crlb_drift_pos"], "r--", lw=1.5, label="CRLB")
    ax.plot(t, hist["filter_pos_var"], "k:", lw=1, label="filter P")
    ax.set_xlabel("t (s)"); ax.set_ylabel("trace (m²)")
    ax.set_title("Position drift covariance"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, hist["msc_drift_rot"], "b-", lw=1.5, label="2-pose MSC")
    ax.plot(t, hist["crlb_drift_rot"], "r--", lw=1.5, label="CRLB")
    ax.set_xlabel("t (s)"); ax.set_ylabel("trace (rad²)")
    ax.set_title("Rotation drift covariance"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    err = hist["pos_err"]
    ax.plot(t, err, "k-", lw=1, label="actual error")
    msc_std = np.sqrt(hist["filter_pos_var"] + hist["msc_drift_pos"])
    crlb_std = np.sqrt(hist["filter_pos_var"] + hist["crlb_drift_pos"])
    ax.plot(t, msc_std, "b-", lw=1, label="MSC total std")
    ax.plot(t, crlb_std, "r--", lw=1, label="CRLB total std")
    ax.set_yscale("log"); ax.set_xlabel("t (s)"); ax.set_ylabel("m")
    ax.set_title("Error vs uncertainty"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(t, hist["nees_pos_msc"], "b-", lw=0.8, alpha=0.7, label="MSC")
    ax.plot(t, hist["nees_pos_crlb"], "r-", lw=0.8, alpha=0.7, label="CRLB")
    ax.axhline(3, color="k", ls="-", lw=1, label="expected (3)")
    ax.axhline(9.348, color="gray", ls="--", lw=0.8, label="95% upper")
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("t (s)"); ax.set_ylabel("NEES")
    ax.set_title("Position NEES"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t, hist["nees_rot_msc"], "b-", lw=0.8, alpha=0.7, label="MSC")
    ax.plot(t, hist["nees_rot_crlb"], "r-", lw=0.8, alpha=0.7, label="CRLB")
    ax.axhline(3, color="k", ls="-", lw=1, label="expected (3)")
    ax.axhline(9.348, color="gray", ls="--", lw=0.8)
    ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("t (s)"); ax.set_ylabel("NEES")
    ax.set_title("Rotation NEES"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_dual_filter.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_dual_filter.png")


print("Loading simulated data...")
imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
imu_data = [{"t": r[0], "gyro": r[1:4], "acc": r[4:7]} for r in imu_arr]
lidar_data = list(lidar_arr)
print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

all_nees = {"msc_pos": [], "msc_rot": [], "crlb_pos": [], "crlb_rot": []}
for seed in [123, 456, 789, 42, 99]:
    np.random.seed(seed)
    h = run_dual(imu_data, lidar_data)
    mid = len(h["t"]) // 4
    for k, nk in [("msc_pos", "nees_pos_msc"), ("msc_rot", "nees_rot_msc"),
                  ("crlb_pos", "nees_pos_crlb"), ("crlb_rot", "nees_rot_crlb")]:
        v = h[nk][mid:]
        all_nees[k].append(np.median(v[np.isfinite(v)]))
    err = h["pos_err"]
    print(f"  seed={seed}: err={np.mean(err):.4f}m  "
          f"MSC_nees={np.median(h['nees_pos_msc'][mid:]):.2f}  "
          f"CRLB_nees={np.median(h['nees_pos_crlb'][mid:]):.2f}  "
          f"MSC_drift={h['msc_drift_pos'][-1]:.6f}  "
          f"CRLB_drift={h['crlb_drift_pos'][-1]:.6f}")

print(f"\nAcross 5 seeds:")
print(f"  MSC  pos NEES median: {np.median(all_nees['msc_pos']):.2f}")
print(f"  CRLB pos NEES median: {np.median(all_nees['crlb_pos']):.2f}")
print(f"  MSC  rot NEES median: {np.median(all_nees['msc_rot']):.2f}")
print(f"  CRLB rot NEES median: {np.median(all_nees['crlb_rot']):.2f}")

np.random.seed(123)
hist = run_dual(imu_data, lidar_data)
plot_results(hist)

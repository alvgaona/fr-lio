"""Stage 3: 3D IEKF with point-to-plane LiDAR updates and scan-to-scan CRLB
drift covariance for honest published uncertainty.

Mirrors the FAST-LIO implementation: the IEKF runs on an online voxel map,
while a parallel scan-to-scan registration computes a per-step Cramer-Rao
Lower Bound (CRLB) on the relative pose uncertainty between consecutive
scans. The per-step CRLBs are accumulated into a P_drift matrix that is
added (post-hoc) to the published covariance. The filter state itself is
untouched.

State: x = (R, p, v, bg, ba) -- 15 DOF tangent space
Gravity is treated as a known constant (not estimated).

Run: python sim_iekf_3d.py
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
N_STATE_DIM = 15

DT_IMU_REF = 1.0 / IMU_RATE
GYR_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
ACC_NOISE_STD = np.sqrt(0.1 * DT_IMU_REF)
GYR_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)
ACC_BIAS_WALK_STD = np.sqrt(0.0001 * DT_IMU_REF)

LIDAR_POINT_VAR = 0.001
SIGMA_RANGE_PERPOINT = 0.02  # Mid-360 datasheet range noise (m), used by per-point cov
MAX_PLANE_DIST = 0.5
# After voxel downsampling, FAST-LIO typically uses a few hundred effective
# points per IEKF update. 1000 is in line with that.
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
# Require a minimum number of valid correspondences before trusting the
# scan-to-scan CRLB. If registration quality collapses (e.g., a broad
# failure of nearest-neighbor matching due to a filter state jump), skip
# the scan rather than accumulating a spurious spike.
S2S_MIN_VALID_POINTS = 100
# Reject correspondences whose point-to-plane residual exceeds this value.
# Prevents outliers (e.g., wrong-wall matches in parallel-wall geometries)
# from dominating the empirical R_s2s estimate.
S2S_MAX_RESIDUAL = 0.3
# Adaptive rejection: skip scans where the empirical R_s2s or the
# per-scan P_rel trace are more than this multiple above the running
# median of recent scans. This is scene-independent: whatever the
# "normal" noise level is in the current environment becomes the
# baseline automatically.
S2S_ADAPTIVE_REJECT_RATIO = 10.0
S2S_ADAPTIVE_WINDOW = 20
# Motion gating: skip P_drift accumulation when the relative pose between
# consecutive scans is below these thresholds. Prevents spurious drift
# growth during stationary periods (hover) where the per-step CRLB is
# dominated by measurement noise rather than real motion information.
S2S_MIN_TRANS_M = 0.01
S2S_MIN_ROT_RAD = 0.001
# Persistence-based discount: fraction of current planes matching a plane used
# in any of the last N s2s calls -> discount accumulation by (1 - alpha * f).
S2S_PERSIST_ALPHA = 1.0
S2S_PERSIST_NORMAL_TAU = 0.95
S2S_PERSIST_DIST_EPS = 0.10
S2S_PERSIST_HISTORY = 5

DEGEN_THRESHOLD = 100.0


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def rodrigues_exp(omega):
    """Exponential map from R^3 to SO(3) using Rodrigues' formula."""
    angle = np.linalg.norm(omega)
    if angle < 1e-10:
        return np.eye(3) + skew(omega)
    axis = omega / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rodrigues_log(R):
    """Logarithm map from SO(3) to R^3."""
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


class State:
    def __init__(self, R, p, v, bg, ba):
        self.R = R.copy()
        self.p = p.copy()
        self.v = v.copy()
        self.bg = bg.copy()
        self.ba = ba.copy()

    def copy(self):
        return State(self.R, self.p, self.v, self.bg, self.ba)

    def boxplus(self, dx):
        R_new = self.R @ rodrigues_exp(dx[0:3])
        p_new = self.p + dx[3:6]
        v_new = self.v + dx[6:9]
        bg_new = self.bg + dx[9:12]
        ba_new = self.ba + dx[12:15]
        return State(R_new, p_new, v_new, bg_new, ba_new)

    def boxminus(self, other):
        dtheta = rodrigues_log(other.R.T @ self.R)
        dp = self.p - other.p
        dv = self.v - other.v
        dbg = self.bg - other.bg
        dba = self.ba - other.ba
        return np.concatenate([dtheta, dp, dv, dbg, dba])


def predict(state, P, gyro, acc, dt):
    """IMU forward propagation. Returns (new_state, new_P)."""
    omega = gyro - state.bg
    a_body = acc - state.ba
    a_world = state.R @ a_body + GRAVITY

    new_R = state.R @ rodrigues_exp(omega * dt)
    new_p = state.p + state.v * dt + 0.5 * a_world * dt * dt
    new_v = state.v + a_world * dt
    new_state = State(new_R, new_p, new_v, state.bg, state.ba)

    F = np.eye(N_STATE_DIM)
    F[0:3, 0:3] = rodrigues_exp(-omega * dt)
    F[0:3, 9:12] = -np.eye(3) * dt
    F[3:6, 6:9] = np.eye(3) * dt
    F[6:9, 0:3] = -state.R @ skew(a_body) * dt
    F[6:9, 12:15] = -state.R * dt

    G = np.zeros((N_STATE_DIM, 12))
    G[0:3, 0:3] = -np.eye(3) * dt
    G[6:9, 3:6] = -state.R * dt
    G[9:12, 6:9] = np.eye(3) * dt
    G[12:15, 9:12] = np.eye(3) * dt

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
    """Incremental map of world-frame points with k-NN plane fitting.

    Built from previous scans transformed using the EKF state estimate. The
    map drifts with the state estimate, just like FAST-LIO2's ikd-Tree.

    Optional source-pose tagging (source_idx) is the foundation for post-LC
    map correction: each map point remembers which keyframe inserted it, so
    after pose graph optimization corrects keyframe k by Δ_k, all points with
    source_idx == k can be transformed by Δ_k to bring the map into the
    corrected frame.
    """
    def __init__(self):
        self.points = []
        self.voxel_set = set()
        self.source_idx = []  # per-point keyframe index of insertion
        self.tree = None
        self._dirty = False

    def add_points(self, world_points, source_idx=-1):
        for p in world_points:
            key = tuple(np.round(p / MAP_VOXEL_SIZE).astype(int))
            if key not in self.voxel_set:
                self.voxel_set.add(key)
                self.points.append(p.copy())
                self.source_idx.append(source_idx)
        self._dirty = True

    def build_tree(self):
        if self._dirty and len(self.points) > 0:
            self.tree = cKDTree(np.array(self.points))
            self._dirty = False

    def correct_map(self, original_poses, corrected_poses):
        """Transform each map point by Δ_k of its source keyframe k.

        For point p inserted at original pose (R_o, t_o), corrected to (R_c, t_c):
            p_body = R_o^T (p_world - t_o)
            p_world_new = R_c p_body + t_c
        Equivalently:
            p_world_new = R_c R_o^T p_world + (t_c - R_c R_o^T t_o)

        Skips points whose source_idx < 0 (untagged) or out of pose range.
        Rebuilds voxel_set + cKDTree from the corrected points.
        """
        if not self.points or not original_poses or not corrected_poses:
            return 0
        n_poses = len(original_poses)
        n_corrected = 0
        new_points = []
        new_source = []
        for p, idx in zip(self.points, self.source_idx):
            if idx < 0 or idx >= n_poses:
                new_points.append(p.copy())
                new_source.append(idx)
                continue
            R_o, t_o = original_poses[idx]
            R_c, t_c = corrected_poses[idx]
            p_body = R_o.T @ (p - t_o)
            p_new = R_c @ p_body + t_c
            new_points.append(p_new)
            new_source.append(idx)
            n_corrected += 1
        # Rebuild voxel_set; later insertions for the same voxel will be
        # deduplicated against the corrected positions.
        self.points = new_points
        self.source_idx = new_source
        self.voxel_set = set(
            tuple(np.round(p / MAP_VOXEL_SIZE).astype(int)) for p in self.points
        )
        self._dirty = True
        return n_corrected

    def __len__(self):
        return len(self.points)


class GlobalMap:
    """Shadow / global map that absorbs points evicted from the working map.

    Mirrors FAST-LIO's missing piece: the working ikd-Tree is sliding-cube
    pruned for bounded memory; this structure persists those evicted points so
    the full traversed environment is preserved and can be LC-corrected.

    Voxel-deduplicated by the same MAP_VOXEL_SIZE so memory grows only with
    spatial coverage, not with revisitation.

    Production C++ analog: a separate ikd-Tree (or similar) that receives
    points box-deleted from the working tree, never queried by the filter,
    corrected post-LC via per-source-pose Δ.
    """
    def __init__(self):
        self.points = []
        self.source_idx = []
        self.voxel_set = set()

    def absorb(self, evicted_points, evicted_source_idx):
        for p, s in zip(evicted_points, evicted_source_idx):
            arr = np.asarray(p)
            key = tuple(np.round(arr / MAP_VOXEL_SIZE).astype(int))
            if key not in self.voxel_set:
                self.voxel_set.add(key)
                self.points.append(arr.copy())
                self.source_idx.append(s)

    def correct_map(self, original_poses, corrected_poses):
        """Same per-source-pose Δ transform as OnlineMap.correct_map.
        Rebuilds voxel_set after correction so dedup remains consistent.
        """
        if not self.points or not original_poses or not corrected_poses:
            return 0
        n_poses = len(original_poses)
        n_corrected = 0
        new_points = []
        new_source = []
        for p, idx in zip(self.points, self.source_idx):
            if idx < 0 or idx >= n_poses:
                new_points.append(p.copy())
                new_source.append(idx)
                continue
            R_o, t_o = original_poses[idx]
            R_c, t_c = corrected_poses[idx]
            p_body = R_o.T @ (p - t_o)
            p_new = R_c @ p_body + t_c
            new_points.append(p_new)
            new_source.append(idx)
            n_corrected += 1
        self.points = new_points
        self.source_idx = new_source
        self.voxel_set = set(
            tuple(np.round(p / MAP_VOXEL_SIZE).astype(int)) for p in self.points
        )
        return n_corrected

    def __len__(self):
        return len(self.points)


def fit_plane(neighbors):
    """Fit a plane to a set of 3D points.

    Returns (normal, offset, residual) where the plane equation is
    n^T x + d = 0. Returns None if the fit is poor.
    """
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
    """For each body-frame point, find a local plane in the online map.

    Returns list of (p_body, normal, d, sigma_p) tuples, where sigma_p is the
    3x3 spatial covariance of the k-NN neighbors used for plane fitting.
    sigma_p has a small eigenvalue along the plane normal (good plane => tight
    weighting along the residual direction) and larger eigenvalues along the
    surface — used by GICP-style per-point covariance weighting.
    """
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
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        denom = max(len(neighbors) - 1, 1)
        sigma_p = (centered.T @ centered) / denom
        matches.append((p_body, normal, d, sigma_p))
    return matches


def compute_residuals_and_H(state, matches):
    """Compute residual vector h and Jacobian H for the matches.

    Matches are (p_body, n_w, d_w, sigma_p) — sigma_p is unused here (used by
    iekf_update for per-point R weighting).
    """
    n_meas = len(matches)
    h = np.zeros(n_meas)
    H = np.zeros((n_meas, N_STATE_DIM))
    for i, m in enumerate(matches):
        p_body, n_w, d_w = m[0], m[1], m[2]
        p_world = state.R @ p_body + state.p
        h[i] = np.dot(n_w, p_world) + d_w
        H[i, 0:3] = -n_w @ state.R @ skew(p_body)
        H[i, 3:6] = n_w
    return h, H


def compute_scan_to_scan_covariance(points_body_curr, R_curr, t_curr,
                                    prev_scan_points, prev_tree, R_prev, t_prev,
                                    use_perpoint_cov=False):
    """Compute the Cramer-Rao Lower Bound on the relative pose covariance
    between the current scan and the previous one.

    Mirrors the FAST-LIO implementation in laserMapping.cpp. Returns a 6x6
    covariance matrix [position; rotation]. If there are not enough valid
    correspondences, returns a zero matrix.

    Arguments:
        points_body_curr: (N, 3) array of current scan points in body frame
        R_curr, t_curr:    current state estimate (rotation, position)
        prev_scan_points:  (M, 3) array of previous scan points in body frame
        prev_tree:         cKDTree built on prev_scan_points
        R_prev, t_prev:    state estimate when the previous scan was captured
        use_perpoint_cov:  if True, weight each point's FIM contribution by
                           1 / (sigma_range^2 + n^T Σ_p n) where Σ_p is the
                           local k-NN spatial covariance. Replaces the scalar
                           empirical R_s2s with a per-point variance — gives
                           a fully data-driven CRLB with no scalar fudge.
    """
    R_rel = R_prev.T @ R_curr
    t_rel = R_prev.T @ (t_curr - t_prev)

    max_points = min(len(points_body_curr), S2S_MAX_POINTS)
    J = np.zeros((max_points, 6))
    per_point_var = np.zeros(max_points)
    residual_sum_sq = 0.0
    valid_count = 0
    planes_world = []
    sr2 = SIGMA_RANGE_PERPOINT ** 2

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
        J[valid_count, 3:6] = -normal @ skew(R_rel @ p_cur)

        if use_perpoint_cov:
            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            sigma_p = (centered.T @ centered) / max(len(neighbors) - 1, 1)
            per_point_var[valid_count] = sr2 + float(normal @ sigma_p @ normal)
        valid_count += 1

    if valid_count < S2S_MIN_VALID_POINTS:
        return np.zeros((6, 6)), {
            "valid_count": valid_count, "R_s2s": 0.0,
            "min_eigval": 0.0, "max_eigval": 0.0,
            "n_clipped": 0,
            "planes_world": [],
        }

    J_valid = J[:valid_count]

    if use_perpoint_cov:
        # FIM = sum_i (1 / r_i^2) J_i J_i^T = J^T diag(1/r^2) J.
        # No scalar inflation: per-point variance is the noise model.
        inv_var = 1.0 / per_point_var[:valid_count]
        FIM = (J_valid.T * inv_var) @ J_valid
        # For diagnostic compatibility, report effective R_s2s as median
        # per-point variance (representative scale).
        R_s2s = float(np.median(per_point_var[:valid_count]))
    else:
        FIM = J_valid.T @ J_valid
        R_s2s = residual_sum_sq / (valid_count - 6)

    eigvals, eigvecs = np.linalg.eigh(FIM)
    n_clipped = int(np.sum(eigvals <= S2S_MIN_EIGENVALUE))
    inv_eigvals = np.where(eigvals > S2S_MIN_EIGENVALUE, 1.0 / eigvals, 0.0)
    inv_FIM = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T

    if use_perpoint_cov:
        P_rel = inv_FIM
    else:
        P_rel = R_s2s * inv_FIM

    debug = {
        "valid_count": valid_count,
        "R_s2s": R_s2s,
        "eigvals": eigvals.tolist(),
        "n_clipped": n_clipped,
        "planes_world": planes_world,
    }
    return P_rel, debug


def information_weighted_projection(H, P_prior, meas_var=LIDAR_POINT_VAR):
    """Adaptive, threshold-free projection of the pose update onto the
    subspace where the measurement adds meaningful information relative
    to the prior.

    For each eigendirection of the measurement information matrix
    M = H^T R^-1 H, compute the ratio w_i = lambda_i / (lambda_i + v_i^T P^-1 v_i),
    which is the fraction of the posterior information coming from the
    measurement in that direction.

    - w_i -> 1 when the measurement dominates (strong new info)
    - w_i -> 0 when the prior dominates (measurement adds nothing)
    - Smooth transition in between

    The projection P_w = sum_i w_i v_i v_i^T is applied to the pose block
    of the state update. In well-behaved conditions (measurement info >>
    prior info), this reduces to the identity and has no effect on the
    standard Kalman update. In degenerate cases, it smoothly suppresses
    directions where the measurement is uninformative.

    Returns the 6x6 projection matrix and the weights per direction.
    """
    H_pose = H[:, 0:6]
    M = (H_pose.T @ H_pose) / meas_var

    P_pose = P_prior[0:6, 0:6]
    try:
        P_pose_inv = np.linalg.inv(P_pose)
    except np.linalg.LinAlgError:
        return np.eye(6), np.ones(6)

    eigvals, eigvecs = np.linalg.eigh(M)
    prior_info = np.einsum("ij,jk,ik->i", eigvecs.T, P_pose_inv, eigvecs.T)
    weights = eigvals / np.maximum(eigvals + prior_info, 1e-12)

    proj = eigvecs @ np.diag(weights) @ eigvecs.T
    return proj, weights


def iekf_update(state_prior, P_prior, points_body, online_map, use_fej,
                use_degen_suppression=False, use_perpoint_cov=False):
    """IEKF update with point-to-plane measurements.

    If use_fej is True, freezes the measurement Jacobian at the propagated
    prior for the entire update cycle. Otherwise, re-linearizes at every
    iteration (standard IEKF).

    If use_perpoint_cov is True, replaces the scalar LIDAR_POINT_VAR with a
    per-match variance derived from the local k-NN spatial covariance
    (GICP-style). Each match's R = sigma_range^2 + n^T Σ_p n.
    """
    matches = find_correspondences(state_prior, points_body, online_map)
    if len(matches) < 5:
        return state_prior, P_prior

    if len(matches) > MAX_POINTS_PER_SCAN:
        idx = np.random.choice(len(matches), MAX_POINTS_PER_SCAN, replace=False)
        matches = [matches[i] for i in idx]

    if use_perpoint_cov:
        sr2 = SIGMA_RANGE_PERPOINT ** 2
        diag = np.empty(len(matches))
        for i, m in enumerate(matches):
            n_w, sigma_p = m[1], m[3]
            diag[i] = sr2 + float(n_w @ sigma_p @ n_w)
        R_meas = np.diag(diag)
    else:
        R_meas = LIDAR_POINT_VAR * np.eye(len(matches))

    H_fej = None
    K_fej = None
    if use_fej:
        _, H_fej = compute_residuals_and_H(state_prior, matches)
        S = H_fej @ P_prior @ H_fej.T + R_meas
        K_fej = P_prior @ H_fej.T @ np.linalg.inv(S)

    state = state_prior.copy()
    H_final = None
    K_final = None

    for j in range(MAX_IEKF_ITERS):
        h, H_iter = compute_residuals_and_H(state, matches)
        if use_fej:
            H = H_fej
            K = K_fej
        else:
            H = H_iter
            S = H @ P_prior @ H.T + R_meas
            K = P_prior @ H.T @ np.linalg.inv(S)

        dx_from_prior = state.boxminus(state_prior)
        delta = K @ (-h - H @ dx_from_prior)

        if use_degen_suppression:
            # Adaptive, threshold-free projection of the pose update onto
            # the subspace where the measurement contributes meaningfully
            # relative to the prior. See information_weighted_projection.
            proj6, _ = information_weighted_projection(H_iter, P_prior)
            delta[0:6] = proj6 @ delta[0:6]

        state = state_prior.boxplus(dx_from_prior + delta)

        H_final = H
        K_final = K

        if np.linalg.norm(delta) < CONVERGENCE_THRESH:
            break

    if use_fej:
        H_final = H_fej
        K_final = K_fej

    P_new = (np.eye(N_STATE_DIM) - K_final @ H_final) @ P_prior
    return state, P_new


def initial_state_and_cov():
    R0, p0, v0, _, _ = trajectory_at(0.0)
    # Initial perturbation suppressed for NEES consistency tests; only the
    # random walk from IMU/lidar noise should drive the published-cov check.
    state = State(
        R=R0,
        p=p0.copy(),
        v=v0.copy(),
        bg=INITIAL_GYR_BIAS.copy(),
        ba=INITIAL_ACC_BIAS.copy(),
    )
    P = np.diag([
        0.02, 0.02, 0.02,
        0.05, 0.05, 0.05,
        0.05, 0.05, 0.05,
        0.001, 0.001, 0.001,
        0.005, 0.005, 0.005,
    ]) ** 2
    return state, P


def run_filter(imu_data, lidar_data, use_fej, use_s2s=False,
               use_degen_suppression=False):
    from collections import deque

    state, P = initial_state_and_cov()
    online_map = OnlineMap()

    prev_scan_points = None
    prev_tree = None
    R_prev_s2s = None
    t_prev_s2s = None
    P_drift = np.zeros((6, 6))  # world-frame accumulation

    # Running windows for adaptive rejection of anomalous scans
    r_s2s_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    p_rel_trace_window = deque(maxlen=S2S_ADAPTIVE_WINDOW)
    plane_history = deque(maxlen=S2S_PERSIST_HISTORY)

    history = {
        "t": [], "p_est": [], "p_true": [],
        "yaw_var": [], "pos_var": [], "map_size": [],
        "rot_err": [], "err_body": [],
        "pos_var_inflated": [], "yaw_var_inflated": [],
        "P_drift_pos_trace": [], "P_drift_rot_trace": [],
        "nees_pos": [], "nees_rot": [], "persist_frac": [],
        "final_map": None,
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
                    state, P = iekf_update(state, P, scan["points_body"], online_map,
                                           use_fej, use_degen_suppression)

                if use_s2s and prev_tree is not None:
                    t_rel_gate = np.linalg.norm(state.p - t_prev_s2s)
                    rot_rel_gate = np.linalg.norm(rodrigues_log(R_prev_s2s.T @ state.R))
                    if t_rel_gate > S2S_MIN_TRANS_M or rot_rel_gate > S2S_MIN_ROT_RAD:
                        P_rel, s2s_debug = compute_scan_to_scan_covariance(
                            scan["points_body"], state.R, state.p,
                            prev_scan_points, prev_tree, R_prev_s2s, t_prev_s2s,
                        )
                        r_s2s_val = s2s_debug["R_s2s"]
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

                        # Persistence fraction against last N scan plane sets
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
                            # Frame fix: rotate P_rel into world tangent before sum.
                            Adj_prev = np.zeros((6, 6))
                            Adj_prev[0:3, 0:3] = R_prev_s2s
                            Adj_prev[3:6, 3:6] = R_prev_s2s
                            P_rel_w = Adj_prev @ P_rel @ Adj_prev.T
                            P_drift += persist_scale * P_rel_w
                            if r_s2s_val > 0:
                                r_s2s_window.append(r_s2s_val)
                            if p_rel_trace > 0:
                                p_rel_trace_window.append(p_rel_trace)
                            if curr_planes:
                                plane_history.append(curr_planes)
                        history["persist_frac"].append(persist_frac)

                if use_s2s:
                    prev_scan_points = scan["points_body"].copy()
                    prev_tree = cKDTree(prev_scan_points)
                    R_prev_s2s = state.R.copy()
                    t_prev_s2s = state.p.copy()

                world_points = (state.R @ scan["points_body"].T).T + state.p
                online_map.add_points(world_points)
            lidar_idx += 1

        R_true, p_true, _, _, _ = trajectory_at(t)
        rot_err = rodrigues_log(R_true.T @ state.R)
        err_body = state.R.T @ (state.p - p_true)

        # Frame fix: rotate world-frame P_drift back into current body tangent
        # before inflating the published covariance.
        Adj_curT = np.zeros((6, 6))
        Adj_curT[0:3, 0:3] = state.R.T
        Adj_curT[3:6, 3:6] = state.R.T
        P_drift_body = Adj_curT @ P_drift @ Adj_curT.T

        P_pub = P.copy()
        P_pub[3:6, 3:6] += P_drift_body[0:3, 0:3]
        P_pub[0:3, 0:3] += P_drift_body[3:6, 3:6]

        # NEES (position and rotation) using ground truth and published cov
        Ppos = P_pub[3:6, 3:6]
        eps_p = state.p - p_true
        try:
            nees_p = float(eps_p @ np.linalg.solve(Ppos, eps_p))
        except np.linalg.LinAlgError:
            nees_p = np.nan
        Prot = P_pub[0:3, 0:3]
        try:
            nees_r = float(rot_err @ np.linalg.solve(Prot, rot_err))
        except np.linalg.LinAlgError:
            nees_r = np.nan
        history["nees_pos"].append(nees_p)
        history["nees_rot"].append(nees_r)

        history["t"].append(t)
        history["p_est"].append(state.p.copy())
        history["p_true"].append(p_true.copy())
        history["yaw_var"].append(P[2, 2])
        history["pos_var"].append(np.trace(P[3:6, 3:6]))
        history["yaw_var_inflated"].append(P_pub[2, 2])
        history["pos_var_inflated"].append(np.trace(P_pub[3:6, 3:6]))
        history["P_drift_pos_trace"].append(np.trace(P_drift[0:3, 0:3]))
        history["P_drift_rot_trace"].append(np.trace(P_drift[3:6, 3:6]))
        history["map_size"].append(len(online_map))
        history["rot_err"].append(rot_err)
        history["err_body"].append(err_body)

    final_map = np.array(online_map.points) if len(online_map) > 0 else np.zeros((0, 3))
    array_keys = [
        "t", "p_est", "p_true", "yaw_var", "pos_var", "map_size",
        "rot_err", "err_body",
        "yaw_var_inflated", "pos_var_inflated",
        "P_drift_pos_trace", "P_drift_rot_trace",
        "nees_pos", "nees_rot",
    ]
    for key in array_keys:
        history[key] = np.array(history[key])
    history["final_map"] = final_map
    return history


def plot_comparison(hist_std, hist_s2s, hist_full=None):
    """Compare standard IEKF against IEKF with scan-to-scan CRLB inflation
    and (optionally) with degeneracy suppression.

    The standard and s2s filters have identical state estimates (s2s only
    affects the published covariance), so this plot focuses on covariance
    behavior rather than trajectory differences. The hist_full variant (with
    degeneracy suppression) has its own state trajectory.
    """
    t = hist_std["t"]
    err_norm = np.linalg.norm(hist_std["p_est"] - hist_std["p_true"], axis=1)

    fig = plt.figure(figsize=(16, 11))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(hist_std["p_true"][:, 0], hist_std["p_true"][:, 1], "k-", linewidth=2.0, label="ground truth")
    ax.plot(hist_std["p_est"][:, 0], hist_std["p_est"][:, 1], "r-", linewidth=1.0, alpha=0.7, label="IEKF (std / s2s)")
    if hist_full is not None:
        ax.plot(hist_full["p_est"][:, 0], hist_full["p_est"][:, 1], "g-", linewidth=1.0, alpha=0.8, label="IEKF + degen")
    gt_x = hist_std["p_true"][:, 0]
    gt_y = hist_std["p_true"][:, 1]
    x_range = max(gt_x.max() - gt_x.min(), 1.0)
    y_range = max(gt_y.max() - gt_y.min(), 1.0)
    half_extent = max(x_range, y_range) * 2.0
    cx = (gt_x.max() + gt_x.min()) / 2.0
    cy = (gt_y.max() + gt_y.min()) / 2.0
    ax.set_xlim(cx - half_extent, cx + half_extent)
    ax.set_ylim(cy - half_extent, cy + half_extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory (zoom around ground truth)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, err_norm, "k-", linewidth=1.2, label="actual err (std)")
    ax.plot(t, np.sqrt(hist_std["pos_var"]), "r--", linewidth=1.0, label="filter std")
    ax.plot(t, np.sqrt(hist_s2s["pos_var_inflated"]), "b-", linewidth=1.0, label="published (s2s)")
    if hist_full is not None:
        err_full = np.linalg.norm(hist_full["p_est"] - hist_full["p_true"], axis=1)
        ax.plot(t, err_full, color="#006600", linewidth=1.2, alpha=0.8, label="actual err (degen)")
        ax.plot(t, np.sqrt(hist_full["pos_var_inflated"]), color="#00aa00", linewidth=1.0, linestyle=":", label="published (s2s+degen)")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Position error and uncertainty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, np.abs(hist_std["rot_err"][:, 2]), "k-", linewidth=1.2, label="|yaw err| (std)")
    ax.plot(t, np.sqrt(hist_std["yaw_var"]), "r--", linewidth=1.0, label="filter std")
    ax.plot(t, np.sqrt(hist_s2s["yaw_var_inflated"]), "b-", linewidth=1.0, label="published (s2s)")
    if hist_full is not None:
        ax.plot(t, np.abs(hist_full["rot_err"][:, 2]), color="#006600", linewidth=1.2, alpha=0.8, label="|yaw err| (degen)")
        ax.plot(t, np.sqrt(hist_full["yaw_var_inflated"]), color="#00aa00", linewidth=1.0, linestyle=":", label="published (s2s+degen)")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rad")
    ax.set_title("Yaw error and uncertainty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, hist_s2s["P_drift_pos_trace"], "b-", linewidth=1.2, label="s2s")
    if hist_full is not None:
        ax.plot(t, hist_full["P_drift_pos_trace"], color="#00aa00", linewidth=1.2, linestyle=":", label="s2s+degen")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P_drift position trace (m^2)")
    ax.set_title("P_drift position growth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, hist_s2s["P_drift_rot_trace"], "b-", linewidth=1.2, label="s2s")
    if hist_full is not None:
        ax.plot(t, hist_full["P_drift_rot_trace"], color="#00aa00", linewidth=1.2, linestyle=":", label="s2s+degen")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P_drift rotation trace (rad^2)")
    ax.set_title("P_drift rotation growth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig.add_subplot(2, 3, 6)
    nees_std = (hist_std["rot_err"][:, 2] ** 2) / np.maximum(hist_std["yaw_var"], 1e-12)
    nees_s2s = (hist_s2s["rot_err"][:, 2] ** 2) / np.maximum(hist_s2s["yaw_var_inflated"], 1e-12)
    ax.plot(t, nees_std, "r-", linewidth=1.0, alpha=0.7, label="filter only")
    ax.plot(t, nees_s2s, "b-", linewidth=1.0, alpha=0.7, label="s2s")
    if hist_full is not None:
        nees_full = (hist_full["rot_err"][:, 2] ** 2) / np.maximum(hist_full["yaw_var_inflated"], 1e-12)
        ax.plot(t, nees_full, color="#00aa00", linewidth=1.0, alpha=0.8, linestyle=":", label="s2s+degen")
    ax.axhline(3.84, color="black", linestyle="--", linewidth=0.8, label="chi^2 95%")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("yaw NEES")
    ax.set_title("Yaw NEES")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_iekf_3d_results.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_iekf_3d_results.png")


def plot_scan_to_scan(hist):
    """Plot showing the scan-to-scan CRLB drift covariance accumulation.

    Shows P_drift position and rotation on log scale to reveal the linear
    growth rate, plus actual position error for comparison.
    """
    t = hist["t"]
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(t, hist["P_drift_pos_trace"], "b-", linewidth=1.2, label="P_drift trace (pos)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P_drift position trace (m^2)")
    ax.set_title("P_drift position accumulation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(t, hist["P_drift_rot_trace"], "b-", linewidth=1.2, label="P_drift trace (rot)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("P_drift rotation trace (rad^2)")
    ax.set_title("P_drift rotation accumulation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(t, err_norm, "k-", linewidth=1.0, label="actual error")
    ax.plot(t, np.sqrt(hist["pos_var"]), "r--", linewidth=1.0, label="filter pos std")
    ax.plot(t, np.sqrt(hist["pos_var_inflated"]), "b-", linewidth=1.0, label="published pos std")
    ax.set_yscale("log")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position (m)")
    ax.set_title("Published (s2s) vs filter pos uncertainty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_scan_to_scan.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_scan_to_scan.png")


def plot_nees(hist):
    """NEES consistency check using ground-truth error vs published covariance.

    A consistent 3-DOF NEES has expected value 3 and a 95% single-sample
    interval of [0.216, 9.348] (chi^2_3). Values below indicate a conservative
    filter; values above indicate overconfidence.
    """
    from scipy.stats import chi2
    t = hist["t"]
    nees_p = np.array(hist["nees_pos"])
    nees_r = np.array(hist["nees_rot"])
    lo, hi = chi2.ppf([0.025, 0.975], df=3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, vals, title in [
        (axes[0], nees_p, "Position NEES (3-DOF)"),
        (axes[1], nees_r, "Rotation NEES (3-DOF)"),
    ]:
        ax.plot(t, vals, "b-", linewidth=0.8)
        ax.axhline(3.0, color="k", linestyle="-", linewidth=1.0, label="expected (3)")
        ax.axhline(lo, color="g", linestyle="--", linewidth=1.0, label=f"95% lower ({lo:.2f})")
        ax.axhline(hi, color="r", linestyle="--", linewidth=1.0, label=f"95% upper ({hi:.2f})")
        finite = vals[np.isfinite(vals)]
        if len(finite):
            mean_v = finite.mean()
            ax.text(0.02, 0.95, f"mean={mean_v:.2f}", transform=ax.transAxes,
                    va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("NEES")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sim_nees.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/sim_nees.png")

    # Print summary
    for label, vals in [("pos", nees_p), ("rot", nees_r)]:
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        inside = np.mean((finite >= lo) & (finite <= hi))
        below = np.mean(finite < lo)
        above = np.mean(finite > hi)
        print(f"  NEES {label}: mean={finite.mean():.3f}  median={np.median(finite):.3f}  "
              f"in-95={inside:.2%}  below={below:.2%}  above={above:.2%}")


def summarize(name, hist):
    err_norm = np.linalg.norm(hist["p_est"] - hist["p_true"], axis=1)
    mid = len(hist["t"]) // 2
    rot_err_late = hist["rot_err"][mid:]
    yaw_var_late = hist["yaw_var"][mid:]
    yaw_var_inf_late = hist["yaw_var_inflated"][mid:]
    yaw_err_late = rot_err_late[:, 2]
    print(f"\n[{name}]")
    print(f"  Position error: mean={np.mean(err_norm):.4f} m, max={np.max(err_norm):.4f} m, "
          f"final={err_norm[-1]:.4f} m")
    print(f"  Yaw error mean={np.mean(yaw_err_late):+.5f} std={np.std(yaw_err_late):.5f} rad")
    print(f"  Filter yaw std (mean over 2nd half)={np.mean(np.sqrt(yaw_var_late)):.5f} rad")
    print(f"  Published yaw std (inflated, mean 2nd half)={np.mean(np.sqrt(yaw_var_inf_late)):.5f} rad")
    print(f"  Final P_drift pos trace: {hist['P_drift_pos_trace'][-1]:.6f} m^2")
    print(f"  Final P_drift rot trace: {hist['P_drift_rot_trace'][-1]:.6f} rad^2")
    yaw_overconfidence = abs(np.mean(yaw_err_late)) / max(np.mean(np.sqrt(yaw_var_late)), 1e-9)
    yaw_overconfidence_inf = abs(np.mean(yaw_err_late)) / max(np.mean(np.sqrt(yaw_var_inf_late)), 1e-9)
    print(f"  Yaw overconfidence ratio (filter) = {yaw_overconfidence:.2f}x")
    print(f"  Yaw overconfidence ratio (published) = {yaw_overconfidence_inf:.2f}x")


if __name__ == "__main__":
    print("Loading simulated data...")
    imu_arr = np.load(f"{OUT_DIR}/sim_imu.npy")
    lidar_arr = np.load(f"{OUT_DIR}/sim_lidar.npy", allow_pickle=True)
    print(f"  {len(imu_arr)} IMU samples, {len(lidar_arr)} LiDAR scans")

    imu_data = [{"t": row[0], "gyro": row[1:4], "acc": row[4:7]} for row in imu_arr]
    lidar_data = list(lidar_arr)

    print("Running standard IEKF...")
    np.random.seed(123)
    hist_std = run_filter(imu_data, lidar_data, use_fej=False, use_s2s=False)

    print("Running standard IEKF + scan-to-scan CRLB drift...")
    np.random.seed(123)
    hist_s2s = run_filter(imu_data, lidar_data, use_fej=False, use_s2s=True)

    print("Running standard IEKF + scan-to-scan + degeneracy suppression...")
    np.random.seed(123)
    hist_full = run_filter(imu_data, lidar_data, use_fej=False, use_s2s=True,
                           use_degen_suppression=True)

    summarize("Standard IEKF", hist_std)
    summarize("Standard IEKF + scan-to-scan", hist_s2s)
    summarize("Standard IEKF + scan-to-scan + degen", hist_full)

    plot_comparison(hist_std, hist_s2s, hist_full)
    plot_scan_to_scan(hist_s2s)
    print("\nNEES (Standard IEKF + scan-to-scan):")
    plot_nees(hist_s2s)

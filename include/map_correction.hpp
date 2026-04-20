#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "common_lib.h"
#include "ikd-Tree/ikd_Tree.h"

/*
 * Shadow-map voxel dedup key: exact 3-int32 coordinate tuple, cheap hash.
 * No false collisions, unlike XOR-of-primes on int64.
 */
struct VoxelKey {
    int32_t x, y, z;
    bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const noexcept {
        size_t h = static_cast<uint32_t>(k.x) * 73856093u;
        h ^= static_cast<uint32_t>(k.y) * 19349663u + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= static_cast<uint32_t>(k.z) * 83492791u + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

inline VoxelKey shadow_voxel_key(float x, float y, float z, double voxel_size)
{
    return VoxelKey{
        static_cast<int32_t>(std::floor(x / voxel_size)),
        static_cast<int32_t>(std::floor(y / voxel_size)),
        static_cast<int32_t>(std::floor(z / voxel_size))
    };
}

/*
 * transform_points_by_source_delta — apply per-source-keyframe Δ to points.
 *
 * For each point p with valid `normal_x` source_idx k, transforms:
 *   p_new = deltas[k] · p
 * Points with out-of-range source_idx are left unchanged.
 *
 * Returns the number of points actually transformed. Pure function — no
 * ikd-Tree, no concurrency, fully unit-testable.
 */
inline int transform_points_by_source_delta(
    PointVector& points,
    const std::vector<Eigen::Isometry3d>& deltas)
{
    int n_transformed = 0;
    for (auto& p : points) {
        int k = static_cast<int>(p.normal_x);
        if (k < 0 || k >= static_cast<int>(deltas.size())) continue;
        Eigen::Vector3d pw(p.x, p.y, p.z);
        Eigen::Vector3d pw_new = deltas[k] * pw;
        p.x = static_cast<float>(pw_new.x());
        p.y = static_cast<float>(pw_new.y());
        p.z = static_cast<float>(pw_new.z());
        ++n_transformed;
    }
    return n_transformed;
}

/*
 * compute_keyframe_deltas — Δ_k = corrected[k] · orig[k]^{-1}.
 * Pure function.
 */
inline std::vector<Eigen::Isometry3d> compute_keyframe_deltas(
    const std::vector<Eigen::Isometry3d>& corrected,
    const std::vector<Eigen::Isometry3d>& orig)
{
    std::vector<Eigen::Isometry3d> deltas(corrected.size());
    for (size_t k = 0; k < corrected.size(); ++k) {
        deltas[k] = corrected[k] * orig[k].inverse();
    }
    return deltas;
}

/*
 * correct_map_core — pure function, unit-testable.
 *
 * Applies a loop-closure correction: for each keyframe k, computes
 *   Δ_k = corrected_poses[k] · keyframe_poses_orig[k]^{-1}
 * and transforms every point in the shadow ikd-Tree by Δ of its source
 * keyframe (encoded in the point's `normal_x` field).
 *
 * On success:
 *   - Rebuilds `tree` and `voxel_set` with transformed points
 *   - Writes the new T_map_odom = Δ of the last keyframe to `T_map_odom_out`
 *   - Replaces `keyframe_poses_orig_inout` with `corrected_poses` (in-place)
 *
 * Returns the number of points in the rebuilt tree, or negative on error
 * (size mismatch between corrected_poses and keyframe_poses_orig).
 *
 * This function owns NO concurrency — caller is responsible for locking
 * the tree mutex if the tree is shared with other threads.
 */
inline int correct_map_core(
    const std::vector<Eigen::Isometry3d>& corrected_poses,
    std::vector<Eigen::Isometry3d>& keyframe_poses_orig_inout,
    KD_TREE<PointType>& tree,
    std::unordered_set<VoxelKey, VoxelKeyHash>& voxel_set,
    double voxel_size,
    Eigen::Isometry3d& T_map_odom_out)
{
    if (corrected_poses.size() != keyframe_poses_orig_inout.size()) return -2;
    if (corrected_poses.empty()) return 0;

    auto deltas = compute_keyframe_deltas(corrected_poses, keyframe_poses_orig_inout);

    PointVector all_points;
    tree.flatten(tree.Root_Node, all_points, NOT_RECORD);
    transform_points_by_source_delta(all_points, deltas);

    voxel_set.clear();
    for (const auto& p : all_points) {
        voxel_set.insert(shadow_voxel_key(p.x, p.y, p.z, voxel_size));
    }

    if (!all_points.empty()) {
        tree.Build(all_points);
    }

    // Across multiple LC events, T_map_odom must accumulate. Each call to
    // correct_map_core applies an INCREMENTAL delta to the shadow tree (its
    // points were already in the previous map frame). The cumulative
    // transform from live odom to the current map frame is the product of
    // all incremental deltas, not the latest delta alone.
    Eigen::Isometry3d incremental_delta =
        corrected_poses.back() * keyframe_poses_orig_inout.back().inverse();
    T_map_odom_out = T_map_odom_out * incremental_delta;
    keyframe_poses_orig_inout = corrected_poses;

    return static_cast<int>(all_points.size());
}

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>
#include "common_lib.h"

/*
 * LCKeyframe — a pose + downsampled scan snapshot captured at specific
 * moments during live odometry. Keyframes are the granularity at which
 * loop closure is detected and pose-graph corrections are applied. Each
 * keyframe's `source_idx` matches the source_idx stamped onto shadow-map
 * points inserted during the keyframe's corresponding scan, so the map
 * correction plan's Δ-transform uses the same indexing.
 */
struct LCKeyframe {
    int source_idx;                          // matches map_correction tagging
    double timestamp;                        // lidar_end_time of the scan
    Eigen::Isometry3d pose_odom;             // pose in odom frame at insertion
    PointCloudXYZI::Ptr scan_downsampled;    // downsampled body-frame scan
    // Snapshot of accumulated scan-to-scan CRLB drift covariance at
    // keyframe-creation time. When lc.use_crlb_edges is enabled, the LC
    // edge noise between kf_i and kf_j is derived from the cumulative
    // P_drift (kf_j) - P_drift (kf_i) — i.e. the drift that is expected
    // to have accumulated between the two visits. Units: 6x6 in
    // (position, rotation) block order (matching laserMapping.cpp).
    Eigen::Matrix<double, 6, 6> p_drift_snapshot =
        Eigen::Matrix<double, 6, 6>::Zero();
};

/*
 * KeyframeDB — append-only store of LCKeyframes with a spatial index on
 * keyframe positions for radius-based LC candidate queries.
 *
 * Thread-safe: internal mutex. Main thread calls add(); LC worker thread
 * calls radius_search/get/size.
 *
 * Spatial index strategy: rebuild the kd-tree every `rebuild_every` adds.
 * For typical keyframe cadences (every few scans), this amortizes to
 * O(log n) per add with no latency spike on the main thread's call path.
 */
class KeyframeDB {
public:
    explicit KeyframeDB(int rebuild_every = 10) : rebuild_every_(rebuild_every) {}

    void add(LCKeyframe kf)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        kfs_.push_back(std::move(kf));
        if (static_cast<int>(kfs_.size()) - last_rebuild_size_ >= rebuild_every_) {
            rebuild_tree_locked();
        }
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return kfs_.size();
    }

    /*
     * Radius + time-gap search: return indices of keyframes whose position
     * is within `radius` of `query_pos` AND whose timestamp is at least
     * `min_time_gap` seconds older than `query_time`. Results sorted by
     * increasing distance.
     */
    std::vector<int> radius_search(
        const Eigen::Vector3d& query_pos,
        double radius,
        double query_time,
        double min_time_gap) const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<int> result;
        if (kfs_.empty() || !pos_tree_ || positions_->empty()) return result;

        pcl::PointXYZ q;
        q.x = static_cast<float>(query_pos.x());
        q.y = static_cast<float>(query_pos.y());
        q.z = static_cast<float>(query_pos.z());

        std::vector<int> idxs;
        std::vector<float> sqd;
        pos_tree_->radiusSearch(q, static_cast<float>(radius), idxs, sqd);

        for (int i : idxs) {
            if (i < 0 || i >= static_cast<int>(kfs_.size())) continue;
            if (query_time - kfs_[i].timestamp < min_time_gap) continue;
            result.push_back(i);
        }
        return result;
    }

    LCKeyframe get(int idx) const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return kfs_.at(idx);
    }

    /*
     * Snapshot of all keyframe poses in the order they were added. Useful
     * for passing to correct_map(corrected_poses).
     */
    std::vector<Eigen::Isometry3d> poses_snapshot() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<Eigen::Isometry3d> out;
        out.reserve(kfs_.size());
        for (const auto& kf : kfs_) out.push_back(kf.pose_odom);
        return out;
    }

private:
    // Must be called with mtx_ held.
    void rebuild_tree_locked()
    {
        positions_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        positions_->points.reserve(kfs_.size());
        for (const auto& kf : kfs_) {
            pcl::PointXYZ p;
            p.x = static_cast<float>(kf.pose_odom.translation().x());
            p.y = static_cast<float>(kf.pose_odom.translation().y());
            p.z = static_cast<float>(kf.pose_odom.translation().z());
            positions_->points.push_back(p);
        }
        positions_->width = positions_->points.size();
        positions_->height = 1;
        positions_->is_dense = false;
        pos_tree_ = std::make_shared<pcl::KdTreeFLANN<pcl::PointXYZ>>();
        pos_tree_->setInputCloud(positions_);
        last_rebuild_size_ = static_cast<int>(kfs_.size());
    }

    mutable std::mutex mtx_;
    std::vector<LCKeyframe> kfs_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr positions_;
    std::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> pos_tree_;
    int rebuild_every_ = 10;
    int last_rebuild_size_ = 0;
};

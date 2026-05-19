// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2025-2026 Alvaro J. Gaona <alvgaona@gmail.com>

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <fr_lio/map_correction.hpp>

namespace {

PointType make_tagged_point(float x, float y, float z, int source_idx)
{
    PointType p;
    p.x = x;
    p.y = y;
    p.z = z;
    p.normal_x = static_cast<float>(source_idx);
    return p;
}

Eigen::Isometry3d iso_translation(double tx, double ty, double tz)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() = Eigen::Vector3d(tx, ty, tz);
    return T;
}

Eigen::Isometry3d iso_rotz_translation(double yaw_rad, double tx, double ty, double tz)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(tx, ty, tz);
    return T;
}

}  // namespace

TEST(VoxelKey, ExactNoFalseCollision)
{
    VoxelKey a{1, 2, 3};
    VoxelKey b{2, 1, 3};
    VoxelKey c{3, 2, 1};
    EXPECT_FALSE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(b == c);
    std::unordered_set<VoxelKey, VoxelKeyHash> s;
    s.insert(a); s.insert(b); s.insert(c);
    EXPECT_EQ(s.size(), 3u);
}

TEST(VoxelKey, ShadowVoxelKeyFloorsCorrectly)
{
    VoxelKey k = shadow_voxel_key(0.05f, -0.15f, 0.35f, 0.2);
    EXPECT_EQ(k.x, 0);
    EXPECT_EQ(k.y, -1);  // floor(-0.75) = -1
    EXPECT_EQ(k.z, 1);
}

TEST(ComputeKeyframeDeltas, IdentityCase)
{
    std::vector<Eigen::Isometry3d> orig = {
        iso_translation(0, 0, 0),
        iso_translation(1, 0, 0),
        iso_translation(2, 0, 0),
    };
    auto deltas = compute_keyframe_deltas(orig, orig);
    ASSERT_EQ(deltas.size(), 3u);
    for (const auto& d : deltas) {
        EXPECT_TRUE(d.isApprox(Eigen::Isometry3d::Identity(), 1e-9));
    }
}

TEST(ComputeKeyframeDeltas, PureTranslationCorrection)
{
    std::vector<Eigen::Isometry3d> orig = {
        iso_translation(0, 0, 0),
        iso_translation(1, 0, 0),
    };
    std::vector<Eigen::Isometry3d> corrected = {
        iso_translation(0, 1, 0),
        iso_translation(1, 2, 0),
    };
    auto deltas = compute_keyframe_deltas(corrected, orig);
    ASSERT_EQ(deltas.size(), 2u);
    EXPECT_NEAR(deltas[0].translation().y(), 1.0, 1e-9);
    EXPECT_NEAR(deltas[1].translation().y(), 2.0, 1e-9);
}

TEST(ComputeKeyframeDeltas, RotationCorrection)
{
    std::vector<Eigen::Isometry3d> orig = {
        iso_rotz_translation(0.0, 0, 0, 0),
    };
    std::vector<Eigen::Isometry3d> corrected = {
        iso_rotz_translation(M_PI / 2.0, 0, 0, 0),  // 90° yaw correction
    };
    auto deltas = compute_keyframe_deltas(corrected, orig);
    ASSERT_EQ(deltas.size(), 1u);
    // Applying Δ to a point at (1,0,0) should rotate it to ~(0,1,0).
    Eigen::Vector3d p(1, 0, 0);
    Eigen::Vector3d p_new = deltas[0] * p;
    EXPECT_NEAR(p_new.x(), 0.0, 1e-9);
    EXPECT_NEAR(p_new.y(), 1.0, 1e-9);
}

TEST(TransformPointsBySourceDelta, PerSourceTranslation)
{
    PointVector pts = {
        make_tagged_point(1.0f, 0.0f, 0.0f, 0),
        make_tagged_point(2.0f, 0.0f, 0.0f, 1),
        make_tagged_point(3.0f, 0.0f, 0.0f, 2),
    };
    std::vector<Eigen::Isometry3d> deltas = {
        iso_translation(0, 1, 0),
        iso_translation(0, 2, 0),
        iso_translation(0, 3, 0),
    };
    int n = transform_points_by_source_delta(pts, deltas);
    EXPECT_EQ(n, 3);
    EXPECT_NEAR(pts[0].y, 1.0f, 1e-5);
    EXPECT_NEAR(pts[1].y, 2.0f, 1e-5);
    EXPECT_NEAR(pts[2].y, 3.0f, 1e-5);
    EXPECT_NEAR(pts[0].x, 1.0f, 1e-5);
    EXPECT_NEAR(pts[1].x, 2.0f, 1e-5);
    EXPECT_NEAR(pts[2].x, 3.0f, 1e-5);
}

TEST(TransformPointsBySourceDelta, IdentityDeltaPreservesPoints)
{
    PointVector pts = {
        make_tagged_point(1.0f, 2.0f, 3.0f, 0),
        make_tagged_point(4.0f, 5.0f, 6.0f, 1),
    };
    std::vector<Eigen::Isometry3d> deltas = {
        Eigen::Isometry3d::Identity(),
        Eigen::Isometry3d::Identity(),
    };
    PointVector before = pts;
    int n = transform_points_by_source_delta(pts, deltas);
    EXPECT_EQ(n, 2);
    for (size_t i = 0; i < pts.size(); ++i) {
        EXPECT_NEAR(pts[i].x, before[i].x, 1e-5);
        EXPECT_NEAR(pts[i].y, before[i].y, 1e-5);
        EXPECT_NEAR(pts[i].z, before[i].z, 1e-5);
    }
}

TEST(TransformPointsBySourceDelta, OutOfRangeSourceSkipped)
{
    PointVector pts = {
        make_tagged_point(1.0f, 0.0f, 0.0f, 0),
        make_tagged_point(2.0f, 0.0f, 0.0f, 999),   // out of range
        make_tagged_point(3.0f, 0.0f, 0.0f, -1),    // negative
    };
    std::vector<Eigen::Isometry3d> deltas = {
        iso_translation(10, 0, 0),
    };
    int n = transform_points_by_source_delta(pts, deltas);
    EXPECT_EQ(n, 1);
    EXPECT_NEAR(pts[0].x, 11.0f, 1e-5);  // shifted
    EXPECT_NEAR(pts[1].x, 2.0f, 1e-5);   // unchanged
    EXPECT_NEAR(pts[2].x, 3.0f, 1e-5);   // unchanged
}

TEST(TransformPointsBySourceDelta, RotationPlusTranslation)
{
    // Point at (1,0,0) tagged with source 0.
    // Δ = 90° yaw + 1m +x translation.
    // Expected: (1,0,0) rotated 90° around z = (0,1,0), then + (1,0,0) = (1,1,0).
    PointVector pts = { make_tagged_point(1.0f, 0.0f, 0.0f, 0) };
    std::vector<Eigen::Isometry3d> deltas = {
        iso_rotz_translation(M_PI / 2.0, 1, 0, 0),
    };
    transform_points_by_source_delta(pts, deltas);
    EXPECT_NEAR(pts[0].x, 1.0f, 1e-5);
    EXPECT_NEAR(pts[0].y, 1.0f, 1e-5);
    EXPECT_NEAR(pts[0].z, 0.0f, 1e-5);
}

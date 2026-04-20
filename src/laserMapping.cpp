// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <unordered_set>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "map_correction.hpp"
#include "lc_keyframe_db.hpp"
#include "lc_worker.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool use_scan_to_scan_cov = false;
PointCloudXYZI::Ptr feats_down_body_prev(new PointCloudXYZI());
pcl::KdTreeFLANN<PointType> kdtree_prev_scan;
bool prev_scan_valid = false;
M3D R_prev_s2s = M3D::Identity();
V3D t_prev_s2s = V3D::Zero();
Eigen::Matrix<double, 6, 6> P_drift = Eigen::Matrix<double, 6, 6>::Zero();

// Scan-to-scan robustness constants (ported from sim_iekf_3d.py)
static constexpr int    S2S_MIN_VALID_POINTS     = 100;
static constexpr double S2S_MIN_TRANS_M          = 0.01;
static constexpr double S2S_MIN_ROT_RAD          = 0.001;
static constexpr size_t S2S_ADAPTIVE_WINDOW      = 20;
static constexpr double S2S_ADAPTIVE_REJECT_RATIO = 10.0;
// Persistence-based discount: fraction of current planes matching a plane used
// in any of the last N s2s calls -> discount accumulation by (1 - alpha * f).
static constexpr double S2S_PERSIST_ALPHA        = 1.0;
static constexpr double S2S_PERSIST_NORMAL_TAU   = 0.95;  // cos(~18 deg)
static constexpr double S2S_PERSIST_DIST_EPS     = 0.10;  // meters
static constexpr size_t S2S_PERSIST_HISTORY      = 5;
std::deque<double> s2s_trace_window;
std::deque<std::vector<Eigen::Vector4d>> s2s_plane_history;
/**************************/

float res_last[100000] = {0.0};
double per_point_var[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;
bool use_perpoint_cov = false;
double huber_k = 3.0;  // Huber cap on standardized residual |r|/√R_i; inflates R_i above this
double point_range_noise_std = 0.02;
double point_range_noise_var = 0.0004;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf;
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

/* Shadow global map — preserves points evicted from the working tree via
   voxel-deduplicated absorption. Gated by mapping.enable_shadow_map.
   Voxel key and shadow_voxel_key live in map_correction.hpp. */
KD_TREE<PointType> global_ikdtree;
std::unordered_set<VoxelKey, VoxelKeyHash> global_voxel_set;
std::mutex mtx_global_map;
bool use_shadow_map = false;
double shadow_voxel_size = 0.2;

/* Source-pose-tagged map correction — every point inserted into the working
   ikd-Tree (and therefore inherited by the shadow tree on eviction) carries
   the index of its source keyframe in `normal_x`. On loop closure, the
   shadow tree is rebuilt by transforming each point by the Δ of its source
   keyframe. Gated by mapping.enable_map_correction. */
bool use_map_correction = false;
int cur_source_idx = 0;
std::vector<Eigen::Isometry3d> keyframe_poses_orig;
Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();

/* Loop closure — keyframe DB populated from map_incremental when
   mapping.enable_lc is true. LC worker thread (future) consumes this
   DB to detect revisits. Gated by lc_enable. */
bool lc_enable = false;
int lc_keyframe_every_scans = 5;
double lc_keyframe_min_dist = 0.5;  // m
KeyframeDB lc_keyframe_db;
int lc_last_kf_scan = 0;
V3D lc_last_kf_pos = V3D::Zero();
int lc_scan_counter = 0;
LCWorker lc_worker;
int lc_max_queue_size = 32;
double lc_radius = 8.0;          // m — candidate search radius
double lc_min_time_gap = 30.0;   // s — minimum time between keyframes for LC candidacy
double lc_min_spacing = 5.0;     // s — minimum time between accepted LC events
double lc_last_accepted_time = -1e9;
double lc_icp_max_dist = 1.5;    // m — max correspondence distance for GICP
double lc_icp_fitness_thresh = 0.3;  // reject GICP result if mean residual exceeds this
int    lc_icp_max_iter = 30;
double lc_max_rel_t_m = 2.0;     // m — reject LC if GICP's final |t_rel| exceeds this
// iSAM2 graph state — owned by the LC worker thread. Populated incrementally
// as keyframes arrive and LC edges are accepted.
std::unique_ptr<gtsam::ISAM2> lc_isam;
std::mutex mtx_isam;  // guards lc_isam access from worker + any external query
double lc_odom_pos_sigma = 0.05;  // m — fixed isotropic odom-edge trans noise
double lc_odom_rot_sigma = 0.01;  // rad — fixed isotropic odom-edge rot noise
double lc_lc_pos_sigma   = 0.15;  // m — fixed LC-edge trans noise (wider than odom)
double lc_lc_rot_sigma   = 0.05;  // rad — fixed LC-edge rot noise
double lc_trigger_pos_m   = 0.10;  // m — fire trigger_correction if any keyframe moves ≥ this
double lc_trigger_rot_rad = 0.02;  // rad — fire trigger_correction if any keyframe rotates ≥ this
std::atomic<bool> lc_correction_in_flight{false};

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

struct ImuPropState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    V3D pos = V3D::Zero();
    V3D vel = V3D::Zero();
    M3D rot = M3D::Identity();
    V3D bg = V3D::Zero();
    V3D ba = V3D::Zero();
    V3D grav = V3D::Zero();
    Eigen::Matrix<double, 23, 23> P = Eigen::Matrix<double, 23, 23>::Zero();
    double timestamp = 0.0;
    bool valid = false;
};

ImuPropState fwd_prop_anchor;
std::mutex mtx_fwd_prop;

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void publish_path_corrected_(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub,
                             const std::string& frame_id,
                             const rclcpp::Time& stamp,
                             const std::vector<Eigen::Isometry3d>& poses)
{
    nav_msgs::msg::Path msg;
    msg.header.frame_id = frame_id;
    msg.header.stamp = stamp;
    msg.poses.reserve(poses.size());
    for (const auto& T : poses) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg.header;
        ps.pose.position.x = T.translation().x();
        ps.pose.position.y = T.translation().y();
        ps.pose.position.z = T.translation().z();
        Eigen::Quaterniond q(T.linear());
        q.normalize();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        ps.pose.orientation.w = q.w();
        msg.poses.push_back(ps);
    }
    pub->publish(msg);
}

void publish_cloud_corrected_(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
                              const std::string& frame_id,
                              const rclcpp::Time& stamp,
                              const PointVector& points)
{
    if (points.empty()) return;
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    cloud->points.assign(points.begin(), points.end());
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.frame_id = frame_id;
    msg.header.stamp = stamp;
    pub->publish(msg);
}

/*
 * correct_map(corrected_poses) — thin wrapper forwarding globals to the
 * pure `correct_map_core` in map_correction.hpp. Acquires the shadow-tree
 * mutex. Only meaningful when `use_map_correction` is true.
 */
int correct_map(const std::vector<Eigen::Isometry3d>& corrected_poses)
{
    if (!use_map_correction) return -1;
    std::lock_guard<std::mutex> lock(mtx_global_map);
    return correct_map_core(
        corrected_poses,
        keyframe_poses_orig,
        global_ikdtree,
        global_voxel_set,
        shadow_voxel_size,
        T_map_odom);
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    if (!use_shadow_map || points_history.empty()) return;

    PointVector to_insert;
    to_insert.reserve(points_history.size());
    size_t evicted = points_history.size();
    {
        std::lock_guard<std::mutex> lock(mtx_global_map);
        for (const auto& p : points_history) {
            VoxelKey k = shadow_voxel_key(p.x, p.y, p.z, shadow_voxel_size);
            if (global_voxel_set.insert(k).second) {
                to_insert.push_back(p);
            }
        }
        if (!to_insert.empty()) {
            if (global_ikdtree.Root_Node == nullptr) {
                global_ikdtree.Build(to_insert);
            } else {
                global_ikdtree.Add_Points(to_insert, false);
            }
        }
    }
    static rclcpp::Clock shadow_log_clock(RCL_ROS_TIME);
    RCLCPP_INFO_THROTTLE(rclcpp::get_logger("laser_mapping"), shadow_log_clock, 2000,
        "shadow_map: evicted=%zu inserted=%zu dedup=%zu working=%d shadow=%d",
        evicted, to_insert.size(), evicted - to_insert.size(),
        ikdtree.validnum(), global_ikdtree.validnum());
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    {
        static rclcpp::Clock fov_log_clock(RCL_ROS_TIME);
        RCLCPP_INFO_THROTTLE(rclcpp::get_logger("laser_mapping"), fov_log_clock, 2000,
            "fov_segment: pos_LiD=[%.2f,%.2f,%.2f] cube=[%.1f,%.1f,%.1f]..[%.1f,%.1f,%.1f] need_move=%d",
            pos_LiD(0), pos_LiD(1), pos_LiD(2),
            LocalMap_Points.vertex_min[0], LocalMap_Points.vertex_min[1], LocalMap_Points.vertex_min[2],
            LocalMap_Points.vertex_max[0], LocalMap_Points.vertex_max[1], LocalMap_Points.vertex_max[2],
            need_move ? 1 : 0);
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    double preprocess_elapsed = omp_get_wtime() - preprocess_start_time;

    mtx_buffer.lock();
    scan_count++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = preprocess_elapsed;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    double preprocess_elapsed = omp_get_wtime() - preprocess_start_time;

    mtx_buffer.lock();
    scan_count++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    s_plot11[scan_count] = preprocess_elapsed;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lock(mtx_buffer);
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** drop stale scans — keep only the latest to stay near real-time ***/
    if (lidar_buffer.size() > 1)
    {
        lidar_pushed = false;
        while (lidar_buffer.size() > 1)
        {
            lidar_buffer.pop_front();
            time_buffer.pop_front();
        }
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void compute_scan_to_scan_covariance(
    const PointCloudXYZI::Ptr &curr_scan,
    const M3D &R_curr, const V3D &t_curr,
    Eigen::Matrix<double, 6, 6> &P_rel_out,
    std::vector<Eigen::Vector4d> &planes_world_out)
{
    P_rel_out.setZero();
    planes_world_out.clear();
    if (!prev_scan_valid || curr_scan->empty() || feats_down_body_prev->empty())
        return;

    M3D R_rel = R_prev_s2s.transpose() * R_curr;
    V3D t_rel = R_prev_s2s.transpose() * (t_curr - t_prev_s2s);

    int max_points = std::min((int)curr_scan->size(), 300);
    int valid_count = 0;
    double residual_sum_sq = 0.0;

    Eigen::MatrixXd J(max_points, 6);
    Eigen::VectorXd per_point_var_s2s(max_points);
    J.setZero();
    per_point_var_s2s.setZero();

    for (int i = 0; i < max_points; i++) {
        V3D p_cur(curr_scan->points[i].x,
                  curr_scan->points[i].y,
                  curr_scan->points[i].z);
        V3D p_in_prev = R_rel * p_cur + t_rel;

        PointType query;
        query.x = p_in_prev(0);
        query.y = p_in_prev(1);
        query.z = p_in_prev(2);

        std::vector<int> nn_idx(NUM_MATCH_POINTS);
        std::vector<float> nn_dist(NUM_MATCH_POINTS);
        if (kdtree_prev_scan.nearestKSearch(query, NUM_MATCH_POINTS, nn_idx, nn_dist) < NUM_MATCH_POINTS)
            continue;
        if (nn_dist[NUM_MATCH_POINTS - 1] > 5.0)
            continue;

        PointVector neighbors(NUM_MATCH_POINTS);
        for (int k = 0; k < NUM_MATCH_POINTS; k++)
            neighbors[k] = feats_down_body_prev->points[nn_idx[k]];

        VF(4) pabcd;
        if (!esti_plane(pabcd, neighbors, 0.1f))
            continue;

        V3D n(pabcd(0), pabcd(1), pabcd(2));
        double d = pabcd(3);
        double residual = pabcd(0) * p_in_prev(0) + pabcd(1) * p_in_prev(1) + pabcd(2) * p_in_prev(2) + d;
        residual_sum_sq += residual * residual;

        // Plane is expressed in the previous-scan body frame. Transform to world
        // frame using R_prev_s2s, t_prev_s2s for cross-scan persistence matching.
        V3D n_w = R_prev_s2s * n;
        double d_w = d - n_w.dot(t_prev_s2s);
        planes_world_out.emplace_back(n_w(0), n_w(1), n_w(2), d_w);

        // Option B (left-perturbation on R_rel): delta_theta lives in the
        // body_{k-1} tangent frame, matching what the block-diagonal adjoint
        // Adj_prev = diag(R_prev_s2s, R_prev_s2s) (built around lines
        // 1214-1216 for accumulation, 1258-1260 for publication-time
        // transport) expects. The rotation Jacobian is therefore
        // -n^T * [R_rel * p_cur]_x, with the skew outermost.
        //
        // Important: the skew is of the ROTATED-only point q = R_rel * p_cur,
        // NOT of p_in_prev = R_rel * p_cur + t_rel. Using p_in_prev would
        // inject a spurious point-independent row contribution
        // -n^T * [t_rel]_x on every valid correspondence.
        //
        // See chapter5.tex §"Rotation Jacobian" Remark [Rotated-only vs.
        // fully transformed point] for the derivation.
        V3D q = R_rel * p_cur;
        M3D skew_q;
        skew_q << SKEW_SYM_MATRX(q);

        J.block<1,3>(valid_count, 0) = n.transpose();
        J.block<1,3>(valid_count, 3) = -n.transpose() * skew_q;

        if (use_perpoint_cov) {
            // Per-point variance from k-NN spatial cov of the previous-scan
            // neighbors. Replaces the lumped scalar R_s2s below.
            Eigen::Matrix3d sigma_p = compute_neighborhood_cov(neighbors);
            per_point_var_s2s(valid_count) = point_range_noise_var
                                             + (n.transpose() * sigma_p * n)(0);
        }

        valid_count++;
    }

    if (valid_count < S2S_MIN_VALID_POINTS) {
        planes_world_out.clear();
        return;
    }

    Eigen::MatrixXd J_valid = J.topRows(valid_count);
    Eigen::Matrix<double, 6, 6> FIM;
    double R_s2s = 0.0;

    if (use_perpoint_cov) {
        // FIM = sum_i (1/R_i) J_i^T J_i = J^T diag(1/R_i) J. The per-point variance
        // captures both sensor range noise and the local plane-fit anisotropy, so no
        // scalar inflation is needed.
        Eigen::VectorXd inv_var = per_point_var_s2s.head(valid_count).array().inverse();
        FIM = J_valid.transpose() * inv_var.asDiagonal() * J_valid;
    } else {
        FIM = J_valid.transpose() * J_valid;
        R_s2s = residual_sum_sq / (valid_count - 6);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(FIM);
    Eigen::Matrix<double, 6, 1> eigenvalues = solver.eigenvalues();
    Eigen::Matrix<double, 6, 6> eigenvectors = solver.eigenvectors();

    double min_eigenvalue = 1e-6;
    Eigen::Matrix<double, 6, 1> inv_eigenvalues;
    for (int i = 0; i < 6; i++) {
        if (eigenvalues(i) > min_eigenvalue)
            inv_eigenvalues(i) = 1.0 / eigenvalues(i);
        else
            inv_eigenvalues(i) = 0.0;
    }

    Eigen::Matrix<double, 6, 6> inv_FIM =
        eigenvectors * inv_eigenvalues.asDiagonal() * eigenvectors.transpose();
    P_rel_out = use_perpoint_cov ? inv_FIM : (R_s2s * inv_FIM);
}

int process_increments = 0;
void map_incremental()
{
    int this_source_idx = -1;
    if (use_map_correction) {
        // Keyframe-grained source_idx: increment only when a new LC keyframe
        // is formed (or at the very first scan). All map points inserted
        // between consecutive keyframes carry the SAME source_idx → they all
        // receive the same Δ during correction, driven by the keyframe's
        // pose correction in iSAM2. This matches LIO-SAM's design and is
        // required so that iSAM2 (which only has nodes for LC keyframes)
        // produces corrections indexed by the same source_idx stamped on
        // shadow-map points.
        Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
        T_odom_base.linear() = state_point.rot.toRotationMatrix();
        T_odom_base.translation() = state_point.pos;

        bool form_lc_keyframe = false;
        if (lc_enable) {
            ++lc_scan_counter;
            V3D delta_pos = state_point.pos - lc_last_kf_pos;
            bool motion_trigger = delta_pos.norm() > lc_keyframe_min_dist;
            bool scan_trigger = (lc_scan_counter - lc_last_kf_scan) >= lc_keyframe_every_scans;
            bool first_kf = (lc_keyframe_db.size() == 0);
            form_lc_keyframe = first_kf || motion_trigger || scan_trigger;
        } else {
            // LC disabled: treat every scan as its own "keyframe" so per-scan
            // source_idx tagging still works for synthetic correction paths
            // (e.g. unit tests or manual PGO output).
            form_lc_keyframe = true;
        }

        if (form_lc_keyframe) {
            keyframe_poses_orig.push_back(T_odom_base);
            this_source_idx = cur_source_idx++;

            if (lc_enable) {
                LCKeyframe kf;
                kf.source_idx = this_source_idx;
                kf.timestamp = lidar_end_time;
                kf.pose_odom = T_odom_base;
                kf.scan_downsampled = PointCloudXYZI::Ptr(new PointCloudXYZI(*feats_down_body));
                LCKeyframe kf_for_worker = kf;
                lc_keyframe_db.add(std::move(kf));
                lc_worker.enqueue(std::move(kf_for_worker));
                lc_last_kf_pos = state_point.pos;
                lc_last_kf_scan = lc_scan_counter;
            }
        } else {
            // Not a keyframe: inherit the source_idx of the most recent
            // keyframe so points we insert this scan share its Δ.
            this_source_idx = cur_source_idx - 1;
        }
    }

    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        if (use_map_correction) {
            feats_down_world->points[i].normal_x = static_cast<float>(this_source_idx);
        }
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}


void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
                // Per-point measurement variance (GICP-style): sigma_range^2 + n^T Σ_p n.
                // Σ_p is the 3x3 spatial cov of the same kd-tree neighbors used for plane
                // fitting. Stash on normvec.curvature; carried through the compaction copy
                // into corr_normvect, then consumed in the H_x scaling loop below.
                if (use_perpoint_cov) {
                    Eigen::Matrix3d sigma_p = compute_neighborhood_cov(points_near);
                    Eigen::Vector3d n_w(pabcd(0), pabcd(1), pabcd(2));
                    double R_i = point_range_noise_var
                                 + (n_w.transpose() * sigma_p * n_w)(0);

                    // Huber-style soft gating: instead of rejecting outliers (which
                    // cascades into divergence when many correspondences drift
                    // together), inflate R_i so the effective std-residual is capped
                    // at `huber_k`. Correspondences with |r|/√R_i ≤ k pass unchanged;
                    // beyond that, R_i = r²/k² so r/√R_i = k. This is the Kalman
                    // analogue of the Huber M-estimator — principled, smooth, robust.
                    const double r_abs = std::abs(static_cast<double>(pd2));
                    const double r_std = r_abs / std::sqrt(R_i);
                    if (r_std > huber_k) {
                        R_i = (r_abs * r_abs) / (huber_k * huber_k);
                    }
                    normvec->points[i].curvature = static_cast<float>(R_i);
                } else {
                    normvec->points[i].curvature = static_cast<float>(LASER_POINT_COV);
                }
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;

        // Per-point R weighting via row scaling: the IKFoM toolkit takes a scalar R,
        // so we pre-scale h_x and h by 1/sqrt(R_i). With R passed as 1.0 to the update,
        // the resulting Kalman gain is mathematically equivalent to using diag(R_i) as
        // the measurement noise covariance.
        if (use_perpoint_cov) {
            const double R_i = static_cast<double>(norm_p.curvature);
            const double inv_sqrt_r = 1.0 / std::sqrt(R_i);
            ekfom_data.h_x.row(i) *= inv_sqrt_r;
            ekfom_data.h(i) *= inv_sqrt_r;
        }
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<string>("frame_prefix", "");
        this->declare_parameter<string>("odom_frame", "odom");
        this->declare_parameter<string>("body_frame", "imu_link");
        this->declare_parameter<bool>("publish.tf_en", true);
        this->declare_parameter<double>("publish.vel_filter_alpha", 1.0);
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("mapping.use_scan_to_scan_cov", false);
        this->declare_parameter<bool>("mapping.use_perpoint_cov", false);
        this->declare_parameter<double>("mapping.point_range_noise_std", 0.02);
        this->declare_parameter<double>("mapping.huber_k", 3.0);
        this->declare_parameter<bool>("mapping.enable_shadow_map", false);
        this->declare_parameter<double>("mapping.shadow_voxel_size", 0.2);
        this->declare_parameter<bool>("mapping.enable_map_correction", false);
        this->declare_parameter<string>("map_frame", "map");
        this->declare_parameter<bool>("lc.enable", false);
        this->declare_parameter<int>("lc.keyframe_every_scans", 5);
        this->declare_parameter<double>("lc.keyframe_min_dist", 0.5);
        this->declare_parameter<int>("lc.max_queue_size", 32);
        this->declare_parameter<double>("lc.radius", 8.0);
        this->declare_parameter<double>("lc.min_time_gap", 30.0);
        this->declare_parameter<double>("lc.min_spacing", 5.0);
        this->declare_parameter<double>("lc.icp_max_dist", 1.5);
        this->declare_parameter<double>("lc.icp_fitness_thresh", 0.3);
        this->declare_parameter<int>("lc.icp_max_iter", 30);
        this->declare_parameter<double>("lc.max_rel_t_m", 2.0);
        this->declare_parameter<double>("lc.odom_pos_sigma", 0.05);
        this->declare_parameter<double>("lc.odom_rot_sigma", 0.01);
        this->declare_parameter<double>("lc.lc_pos_sigma", 0.15);
        this->declare_parameter<double>("lc.lc_rot_sigma", 0.05);
        this->declare_parameter<double>("lc.trigger_pos_m", 0.10);
        this->declare_parameter<double>("lc.trigger_rot_rad", 0.02);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<string>("frame_prefix", frame_prefix_, "");
        this->get_parameter_or<string>("odom_frame", odom_frame_, "odom");
        this->get_parameter_or<string>("body_frame", body_frame_, "imu_link");
        this->get_parameter_or<bool>("publish.tf_en", tf_pub_en_, true);
        this->get_parameter_or<double>("publish.vel_filter_alpha", vel_filter_alpha_, 1.0);
        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("mapping.use_scan_to_scan_cov", use_scan_to_scan_cov, false);
        this->get_parameter_or<bool>("mapping.use_perpoint_cov", use_perpoint_cov, false);
        this->get_parameter_or<double>("mapping.point_range_noise_std", point_range_noise_std, 0.02);
        point_range_noise_var = point_range_noise_std * point_range_noise_std;
        this->get_parameter_or<double>("mapping.huber_k", huber_k, 3.0);
        this->get_parameter_or<bool>("mapping.enable_shadow_map", use_shadow_map, false);
        this->get_parameter_or<double>("mapping.shadow_voxel_size", shadow_voxel_size, 0.2);
        this->get_parameter_or<bool>("mapping.enable_map_correction", use_map_correction, false);
        this->get_parameter_or<string>("map_frame", map_frame_, "map");
        this->get_parameter_or<bool>("lc.enable", lc_enable, false);
        this->get_parameter_or<int>("lc.keyframe_every_scans", lc_keyframe_every_scans, 5);
        this->get_parameter_or<double>("lc.keyframe_min_dist", lc_keyframe_min_dist, 0.5);
        this->get_parameter_or<int>("lc.max_queue_size", lc_max_queue_size, 32);
        this->get_parameter_or<double>("lc.radius", lc_radius, 8.0);
        this->get_parameter_or<double>("lc.min_time_gap", lc_min_time_gap, 30.0);
        this->get_parameter_or<double>("lc.min_spacing", lc_min_spacing, 5.0);
        this->get_parameter_or<double>("lc.icp_max_dist", lc_icp_max_dist, 1.5);
        this->get_parameter_or<double>("lc.icp_fitness_thresh", lc_icp_fitness_thresh, 0.3);
        this->get_parameter_or<int>("lc.icp_max_iter", lc_icp_max_iter, 30);
        this->get_parameter_or<double>("lc.max_rel_t_m", lc_max_rel_t_m, 2.0);
        this->get_parameter_or<double>("lc.odom_pos_sigma", lc_odom_pos_sigma, 0.05);
        this->get_parameter_or<double>("lc.odom_rot_sigma", lc_odom_rot_sigma, 0.01);
        this->get_parameter_or<double>("lc.lc_pos_sigma", lc_lc_pos_sigma, 0.15);
        this->get_parameter_or<double>("lc.lc_rot_sigma", lc_lc_rot_sigma, 0.05);
        this->get_parameter_or<double>("lc.trigger_pos_m", lc_trigger_pos_m, 0.10);
        this->get_parameter_or<double>("lc.trigger_rot_rad", lc_trigger_rot_rad, 0.02);
        if (lc_enable && !use_map_correction) {
            RCLCPP_WARN(this->get_logger(),
                "lc.enable=true requires mapping.enable_map_correction=true; forcing it on");
            use_map_correction = true;
            use_shadow_map = true;  // transitive
        }
        if (use_map_correction && !use_shadow_map) {
            RCLCPP_WARN(this->get_logger(),
                "mapping.enable_map_correction=true requires mapping.enable_shadow_map=true; forcing enable_shadow_map=true");
            use_shadow_map = true;
        }
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        RCLCPP_INFO(this->get_logger(), "=== FAST-LIO Parameters ===");
        RCLCPP_INFO(this->get_logger(), "common.lid_topic: %s", lid_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "common.imu_topic: %s", imu_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "common.time_sync_en: %d", time_sync_en);
        RCLCPP_INFO(this->get_logger(), "common.time_offset_lidar_to_imu: %f", time_diff_lidar_to_imu);
        RCLCPP_INFO(this->get_logger(), "preprocess.lidar_type: %d", p_pre->lidar_type);
        RCLCPP_INFO(this->get_logger(), "preprocess.scan_line: %d", p_pre->N_SCANS);
        RCLCPP_INFO(this->get_logger(), "preprocess.blind: %f", p_pre->blind);
        RCLCPP_INFO(this->get_logger(), "preprocess.timestamp_unit: %d", p_pre->time_unit);
        RCLCPP_INFO(this->get_logger(), "preprocess.scan_rate: %d", p_pre->SCAN_RATE);
        RCLCPP_INFO(this->get_logger(), "point_filter_num: %d", p_pre->point_filter_num);
        RCLCPP_INFO(this->get_logger(), "feature_extract_enable: %d", p_pre->feature_enabled);
        RCLCPP_INFO(this->get_logger(), "max_iteration: %d", NUM_MAX_ITERATIONS);
        RCLCPP_INFO(this->get_logger(), "filter_size_surf: %f", filter_size_surf_min);
        RCLCPP_INFO(this->get_logger(), "filter_size_map: %f", filter_size_map_min);
        RCLCPP_INFO(this->get_logger(), "cube_side_length: %f", cube_len);
        RCLCPP_INFO(this->get_logger(), "mapping.det_range: %f", DET_RANGE);
        RCLCPP_INFO(this->get_logger(), "mapping.fov_degree: %f", fov_deg);
        RCLCPP_INFO(this->get_logger(), "mapping.gyr_cov: %f", gyr_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.acc_cov: %f", acc_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.b_gyr_cov: %f", b_gyr_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.b_acc_cov: %f", b_acc_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.extrinsic_est_en: %d", extrinsic_est_en);
        RCLCPP_INFO(this->get_logger(), "mapping.use_scan_to_scan_cov: %d", use_scan_to_scan_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.use_perpoint_cov: %d", use_perpoint_cov);
        RCLCPP_INFO(this->get_logger(), "mapping.point_range_noise_std: %f", point_range_noise_std);
        RCLCPP_INFO(this->get_logger(), "mapping.huber_k: %f", huber_k);
        RCLCPP_INFO(this->get_logger(), "mapping.enable_shadow_map: %d", use_shadow_map);
        RCLCPP_INFO(this->get_logger(), "mapping.shadow_voxel_size: %f", shadow_voxel_size);
        RCLCPP_INFO(this->get_logger(), "mapping.enable_map_correction: %d", use_map_correction);
        RCLCPP_INFO(this->get_logger(), "map_frame: %s", map_frame_.c_str());
        RCLCPP_INFO(this->get_logger(), "lc.enable: %d", lc_enable);
        RCLCPP_INFO(this->get_logger(), "lc.keyframe_every_scans: %d", lc_keyframe_every_scans);
        RCLCPP_INFO(this->get_logger(), "lc.keyframe_min_dist: %f", lc_keyframe_min_dist);
        RCLCPP_INFO(this->get_logger(), "lc.max_queue_size: %d", lc_max_queue_size);
        RCLCPP_INFO(this->get_logger(), "lc.radius: %f", lc_radius);
        RCLCPP_INFO(this->get_logger(), "lc.min_time_gap: %f", lc_min_time_gap);
        RCLCPP_INFO(this->get_logger(), "lc.min_spacing: %f", lc_min_spacing);
        RCLCPP_INFO(this->get_logger(), "lc.icp_max_dist: %f", lc_icp_max_dist);
        RCLCPP_INFO(this->get_logger(), "lc.icp_fitness_thresh: %f", lc_icp_fitness_thresh);
        RCLCPP_INFO(this->get_logger(), "lc.icp_max_iter: %d", lc_icp_max_iter);
        RCLCPP_INFO(this->get_logger(), "lc.max_rel_t_m: %f", lc_max_rel_t_m);
        RCLCPP_INFO(this->get_logger(), "mapping.extrinsic_T: [%f, %f, %f]", extrinT[0], extrinT[1], extrinT[2]);
        RCLCPP_INFO(this->get_logger(), "publish.path_en: %d", path_en);
        RCLCPP_INFO(this->get_logger(), "publish.scan_publish_en: %d", scan_pub_en);
        RCLCPP_INFO(this->get_logger(), "publish.dense_publish_en: %d", dense_pub_en);
        RCLCPP_INFO(this->get_logger(), "publish.scan_bodyframe_pub_en: %d", scan_body_pub_en);
        RCLCPP_INFO(this->get_logger(), "runtime_pos_log_enable: %d", runtime_pos_log);
        RCLCPP_INFO(this->get_logger(), "===========================");

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = frame_prefix_ + odom_frame_;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        imu_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions imu_opts;
        imu_opts.callback_group = imu_cb_group_;
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, 10,
            std::bind(&LaserMappingNode::imu_cbk, this, std::placeholders::_1),
            imu_opts);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("Laser_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", rclcpp::SensorDataQoS());
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        pubOdomMap_ = this->create_publisher<nav_msgs::msg::Odometry>("odom_map", rclcpp::SensorDataQoS());
        // Correction outputs fire only on successful LC + map rebuild. Use
        // transient-local (latched) QoS so rviz / late subscribers always see
        // the most recent corrected trajectory / cloud, not just what was
        // published during their subscription window.
        pubPathCorrected_ = this->create_publisher<nav_msgs::msg::Path>(
            "path_corrected", rclcpp::QoS(1).transient_local());
        pubCloudMapCorrected_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "cloud_map_corrected", rclcpp::QoS(1).transient_local());

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

        pubShadowMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cloud_shadow_map", 1);

        if (lc_enable) {
            lc_worker.set_max_queue_size(static_cast<size_t>(lc_max_queue_size));
            lc_worker.set_callback([this](const LCKeyframe& kf) {
                this->lc_on_keyframe(kf);
            });
            lc_worker.start();
            RCLCPP_INFO(this->get_logger(), "LC worker thread started");
        }

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        lc_worker.stop();
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

    rclcpp::CallbackGroup::SharedPtr get_imu_callback_group() { return imu_cb_group_; }

private:
    void publish_frame_world()
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = frame_prefix_ + odom_frame_;
        pubLaserCloudFull_->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    void publish_frame_body()
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                                &laserCloudIMUBody->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = frame_prefix_ + body_frame_;
        pubLaserCloudFull_body_->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    void publish_effect_world()
    {
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
        for (int i = 0; i < effct_feat_num; i++)
        {
            RGBpointBodyToWorld(&laserCloudOri->points[i],
                                &laserCloudWorld->points[i]);
        }
        sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
        laserCloudFullRes3.header.frame_id = frame_prefix_ + odom_frame_;
        pubLaserCloudEffect_->publish(laserCloudFullRes3);
    }

    void publish_map()
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub += *laserCloudWorld;

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = frame_prefix_ + odom_frame_;
        pubLaserCloudMap_->publish(laserCloudmsg);
    }

    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                    if (use_shadow_map) {
                        global_ikdtree.set_downsample_param(filter_size_map_min);
                    }
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            // When use_perpoint_cov is on, h_share_model has already pre-scaled h_x and h
            // by 1/sqrt(R_i), so the effective measurement noise covariance the filter
            // should use is the identity. Otherwise keep the historical scalar R.
            const double R_for_filter = use_perpoint_cov ? 1.0 : LASER_POINT_COV;
            kf.update_iterated_dyn_share_modified(R_for_filter, solve_H_time);

            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            if (use_scan_to_scan_cov && flg_EKF_inited) {
                M3D R_curr = state_point.rot.toRotationMatrix();
                V3D t_curr = state_point.pos;

                Eigen::Matrix<double, 6, 6> P_rel;
                std::vector<Eigen::Vector4d> curr_planes_world;
                compute_scan_to_scan_covariance(feats_down_body, R_curr, t_curr, P_rel, curr_planes_world);

                // Persistence discount: count current planes that also appeared
                // in any of the last N s2s plane sets (same physical landmarks).
                double persist_frac = 0.0;
                if (!s2s_plane_history.empty() && !curr_planes_world.empty()) {
                    int hits = 0;
                    for (const auto &pc : curr_planes_world) {
                        Eigen::Vector3d nc(pc(0), pc(1), pc(2));
                        double dc = pc(3);
                        bool matched = false;
                        for (const auto &old_planes : s2s_plane_history) {
                            for (const auto &pp : old_planes) {
                                Eigen::Vector3d np(pp(0), pp(1), pp(2));
                                double dp = pp(3);
                                double dot = nc.dot(np);
                                if (std::abs(dot) < S2S_PERSIST_NORMAL_TAU) continue;
                                double dp_signed = (dot >= 0.0) ? dp : -dp;
                                if (std::abs(dc - dp_signed) < S2S_PERSIST_DIST_EPS) {
                                    matched = true;
                                    break;
                                }
                            }
                            if (matched) break;
                        }
                        if (matched) hits++;
                    }
                    persist_frac = static_cast<double>(hits) / curr_planes_world.size();
                }
                double persist_scale = std::max(0.0, 1.0 - S2S_PERSIST_ALPHA * persist_frac);

                // Motion gating: skip accumulation when stationary (matches sim)
                bool stationary = false;
                if (prev_scan_valid) {
                    V3D dp = t_curr - t_prev_s2s;
                    M3D dR = R_prev_s2s.transpose() * R_curr;
                    double dtheta = Eigen::AngleAxisd(dR).angle();
                    stationary = (dp.norm() < S2S_MIN_TRANS_M) &&
                                 (std::abs(dtheta) < S2S_MIN_ROT_RAD);
                }

                // Adaptive rejection: reject if trace(P_rel) > ratio * running median
                double trace_p_rel = P_rel.trace();
                bool reject = false;
                if (s2s_trace_window.size() >= S2S_ADAPTIVE_WINDOW && trace_p_rel > 0.0) {
                    std::vector<double> sorted(s2s_trace_window.begin(), s2s_trace_window.end());
                    size_t mid = sorted.size() / 2;
                    std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
                    double median = sorted[mid];
                    if (median > 0.0 && trace_p_rel > S2S_ADAPTIVE_REJECT_RATIO * median) {
                        reject = true;
                    }
                }

                if (!stationary && !reject && trace_p_rel > 0.0) {
                    // Rotate P_rel from prev-body tangent into world tangent before
                    // accumulating, so P_drift lives in a single consistent frame.
                    Eigen::Matrix<double, 6, 6> Adj_prev = Eigen::Matrix<double, 6, 6>::Zero();
                    Adj_prev.block<3,3>(0,0) = R_prev_s2s;
                    Adj_prev.block<3,3>(3,3) = R_prev_s2s;
                    Eigen::Matrix<double, 6, 6> P_rel_w = Adj_prev * P_rel * Adj_prev.transpose();
                    P_drift += persist_scale * P_rel_w;
                    s2s_trace_window.push_back(trace_p_rel);
                    if (s2s_trace_window.size() > S2S_ADAPTIVE_WINDOW) {
                        s2s_trace_window.pop_front();
                    }
                }
                if (!curr_planes_world.empty()) {
                    s2s_plane_history.push_back(std::move(curr_planes_world));
                    if (s2s_plane_history.size() > S2S_PERSIST_HISTORY) {
                        s2s_plane_history.pop_front();
                    }
                }

                *feats_down_body_prev = *feats_down_body;
                kdtree_prev_scan.setInputCloud(feats_down_body_prev);
                R_prev_s2s = R_curr;
                t_prev_s2s = t_curr;
                prev_scan_valid = true;
            }

            double t_update_end = omp_get_wtime();

            {
                std::lock_guard<std::mutex> lock(mtx_fwd_prop);
                if (lidar_end_time > fwd_prop_anchor.timestamp) {
                    fwd_prop_anchor.pos = state_point.pos;
                    fwd_prop_anchor.vel = state_point.vel;
                    fwd_prop_anchor.rot = state_point.rot.toRotationMatrix();
                    fwd_prop_anchor.timestamp = lidar_end_time;
                }
                fwd_prop_anchor.bg  = state_point.bg;
                fwd_prop_anchor.ba  = state_point.ba;
                fwd_prop_anchor.grav = V3D(state_point.grav[0],
                                           state_point.grav[1],
                                           state_point.grav[2]);
                fwd_prop_anchor.P = kf.get_P();
                if (use_scan_to_scan_cov && prev_scan_valid) {
                    // Rotate P_drift (world tangent) into the current body tangent before
                    // adding to the published covariance blocks.
                    M3D R_cur_body = state_point.rot.toRotationMatrix();
                    Eigen::Matrix<double, 6, 6> Adj_curT = Eigen::Matrix<double, 6, 6>::Zero();
                    Adj_curT.block<3,3>(0,0) = R_cur_body.transpose();
                    Adj_curT.block<3,3>(3,3) = R_cur_body.transpose();
                    Eigen::Matrix<double, 6, 6> P_drift_body = Adj_curT * P_drift * Adj_curT.transpose();
                    fwd_prop_anchor.P.block<3,3>(0,0) += P_drift_body.block<3,3>(0,0);
                    fwd_prop_anchor.P.block<3,3>(3,3) += P_drift_body.block<3,3>(3,3);
                }
                fwd_prop_anchor.valid = true;
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            /******* Publish points *******/
            if (scan_pub_en) publish_frame_world();
            if (scan_pub_en && scan_body_pub_en) publish_frame_body();
            if (effect_pub_en) publish_effect_world();

            if (use_scan_to_scan_cov && prev_scan_valid) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "S2S drift - P_filter: %.6f, P_drift_pos: %.6f, P_drift_rot: %.6f, P_published: %.6f",
                    kf.get_P().block<3,3>(0,0).trace(),
                    P_drift.block<3,3>(0,0).trace(),
                    P_drift.block<3,3>(3,3).trace(),
                    fwd_prop_anchor.P.block<3,3>(0,0).trace());
            }

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (map_pub_en) publish_map();
        if (use_shadow_map) publish_shadow_map();
    }

    /*
     * trigger_correction(corrected_poses)
     *
     * Member wrapper around the free `correct_map()` function. Calls the
     * core logic (tree rebuild + T_map_odom update) then publishes the
     * corrected artifacts on `/path_corrected` and `/cloud_map_corrected`.
     *
     * This is the entry point plan 3's LC thread will call. Plan 2's unit
     * tests call the free `correct_map()` directly to bypass publishing.
     */
    static gtsam::Pose3 iso_to_gtsam(const Eigen::Isometry3d& T)
    {
        return gtsam::Pose3(gtsam::Rot3(T.linear()), gtsam::Point3(T.translation()));
    }

    static Eigen::Isometry3d gtsam_to_iso(const gtsam::Pose3& P)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = P.rotation().matrix();
        T.translation() = P.translation();
        return T;
    }

    /*
     * lc_on_keyframe — runs on the LC worker thread. For each new keyframe:
     *  1. Query KeyframeDB for revisit candidates (radius + time-gap)
     *  2. (step 5) Verify via GICP between candidate scan and new scan
     *  3. (step 6) Add BetweenFactor to iSAM2, incremental update
     *  4. (step 7) If max pose delta > threshold, call trigger_correction()
     * Current (step 4): just the candidate search + logging.
     */
    // Pull the full corrected trajectory from iSAM2 and publish on
    // /path_corrected. Called after every keyframe (not just after LC
    // events) so the corrected path grows continuously. Before any LC
    // fires, corrected_poses mirrors keyframe_poses_orig in the
    // iSAM2 frame.
    void publish_corrected_path_from_isam()
    {
        std::vector<Eigen::Isometry3d> corrected_poses;
        {
            std::lock_guard<std::mutex> lock(mtx_isam);
            if (!lc_isam) return;
            gtsam::Values est = lc_isam->calculateEstimate();
            size_t n = keyframe_poses_orig.size();
            corrected_poses.reserve(n);
            for (size_t k = 0; k < n; ++k) {
                gtsam::Symbol key('x', static_cast<int>(k));
                if (est.exists(key)) {
                    corrected_poses.push_back(gtsam_to_iso(est.at<gtsam::Pose3>(key)));
                } else {
                    corrected_poses.push_back(keyframe_poses_orig[k]);
                }
            }
        }
        if (corrected_poses.empty()) return;
        auto stamp = this->get_clock()->now();
        publish_path_corrected_(pubPathCorrected_, frame_prefix_ + map_frame_, stamp, corrected_poses);
    }

    void lc_on_keyframe(const LCKeyframe& kf)
    {
        // Build up iSAM2 graph incrementally: on every keyframe, add a prior
        // (for first) or an odom edge (current <- previous) with that edge
        // initialized from the odom-frame pose of the current keyframe.
        {
            std::lock_guard<std::mutex> lock(mtx_isam);
            if (!lc_isam) {
                gtsam::ISAM2Params params;
                params.relinearizeThreshold = 0.01;
                params.relinearizeSkip = 1;
                lc_isam = std::make_unique<gtsam::ISAM2>(params);
            }
            gtsam::NonlinearFactorGraph graph;
            gtsam::Values values;
            gtsam::Symbol key_cur('x', kf.source_idx);

            if (lc_keyframe_db.size() == 1) {
                // First keyframe: add PriorFactor to anchor the graph.
                auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                    key_cur, iso_to_gtsam(kf.pose_odom), prior_noise));
                values.insert(key_cur, iso_to_gtsam(kf.pose_odom));
            } else {
                // Subsequent keyframes: add between-factor to previous keyframe.
                // The relative-pose measurement comes from the odom estimates.
                LCKeyframe prev = lc_keyframe_db.get(static_cast<int>(lc_keyframe_db.size()) - 2);
                Eigen::Isometry3d T_prev_cur = prev.pose_odom.inverse() * kf.pose_odom;
                auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << lc_odom_rot_sigma, lc_odom_rot_sigma, lc_odom_rot_sigma,
                                          lc_odom_pos_sigma, lc_odom_pos_sigma, lc_odom_pos_sigma).finished());
                gtsam::Symbol key_prev('x', prev.source_idx);
                graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    key_prev, key_cur, iso_to_gtsam(T_prev_cur), odom_noise));
                values.insert(key_cur, iso_to_gtsam(kf.pose_odom));
            }
            lc_isam->update(graph, values);
        }

        // Publish the current corrected path on every keyframe so rviz has
        // a continuously-growing trajectory even before the first LC fires.
        // Before any LC: corrected_poses ≈ keyframe_poses_orig (iSAM2 only
        // has odom between-factors). After LC: reflects the refinements.
        publish_corrected_path_from_isam();

        auto candidates = lc_keyframe_db.radius_search(
            kf.pose_odom.translation(),
            lc_radius, kf.timestamp, lc_min_time_gap);

        static rclcpp::Clock lc_log_clock(RCL_ROS_TIME);
        RCLCPP_INFO_THROTTLE(this->get_logger(), lc_log_clock, 2000,
            "lc_worker: kf_idx=%d db_size=%zu candidates=%zu processed=%zu dropped=%zu",
            kf.source_idx, lc_keyframe_db.size(), candidates.size(),
            lc_worker.processed_count(), lc_worker.dropped_count());

        if (candidates.empty()) return;
        if (kf.timestamp - lc_last_accepted_time < lc_min_spacing) return;

        // GICP verification: try to register the current keyframe scan against
        // the nearest candidate. Initial guess from relative odom poses. Reject
        // if ICP doesn't converge or the fitness score (mean sq residual on
        // correspondences) exceeds the threshold.
        int best_candidate = candidates.front();
        LCKeyframe cand = lc_keyframe_db.get(best_candidate);

        // Initial guess: T_cand^cur = T_cand^odom^{-1} · T_cur^odom
        Eigen::Isometry3d T_guess = cand.pose_odom.inverse() * kf.pose_odom;
        Eigen::Matrix4f init_guess = T_guess.matrix().cast<float>();

        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        gicp.setInputSource(kf.scan_downsampled);
        gicp.setInputTarget(cand.scan_downsampled);
        gicp.setMaxCorrespondenceDistance(lc_icp_max_dist);
        gicp.setMaximumIterations(lc_icp_max_iter);
        gicp.setTransformationEpsilon(1e-6);
        PointCloudXYZI aligned;
        gicp.align(aligned, init_guess);

        if (!gicp.hasConverged()) {
            RCLCPP_INFO(this->get_logger(),
                "lc: GICP no-converge kf=%d<-%d", kf.source_idx, best_candidate);
            return;
        }
        double fitness = gicp.getFitnessScore();
        if (fitness > lc_icp_fitness_thresh) {
            RCLCPP_INFO(this->get_logger(),
                "lc: GICP rejected kf=%d<-%d fitness=%.4f > thresh=%.4f",
                kf.source_idx, best_candidate, fitness, lc_icp_fitness_thresh);
            return;
        }

        Eigen::Matrix4f T_cand_cur_f = gicp.getFinalTransformation();
        Eigen::Vector3f t_rel = T_cand_cur_f.topRightCorner<3, 1>();

        // Perceptual-aliasing guard: corridor walls look similar from different
        // positions, so GICP can converge to wall-to-wall matches that aren't
        // the same physical region. A well-behaved LC with true revisit has
        // |t_rel| ~ drift magnitude (<< 1m typically). Reject anything larger.
        if (t_rel.norm() > lc_max_rel_t_m) {
            RCLCPP_INFO(this->get_logger(),
                "lc: rejected kf=%d<-%d |t_rel|=%.3f > max_rel_t=%.3f (likely perceptual aliasing)",
                kf.source_idx, best_candidate, t_rel.norm(), lc_max_rel_t_m);
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "lc: GICP accepted kf=%d<-%d fitness=%.4f |t_rel|=%.3f",
            kf.source_idx, best_candidate, fitness, t_rel.norm());
        lc_last_accepted_time = kf.timestamp;

        // Add LC BetweenFactor to iSAM2 and run an incremental update.
        Eigen::Isometry3d T_cand_cur = Eigen::Isometry3d::Identity();
        T_cand_cur.matrix() = T_cand_cur_f.cast<double>();
        {
            std::lock_guard<std::mutex> lock(mtx_isam);
            if (!lc_isam) return;  // shouldn't happen — graph was initialized above
            gtsam::NonlinearFactorGraph lc_graph;
            auto lc_noise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << lc_lc_rot_sigma, lc_lc_rot_sigma, lc_lc_rot_sigma,
                                      lc_lc_pos_sigma, lc_lc_pos_sigma, lc_lc_pos_sigma).finished());
            gtsam::Symbol key_cand('x', cand.source_idx);
            gtsam::Symbol key_cur('x', kf.source_idx);
            lc_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                key_cand, key_cur, iso_to_gtsam(T_cand_cur), lc_noise));
            lc_isam->update(lc_graph, gtsam::Values());
            // Extra relinearization steps help iSAM2 propagate the LC impact
            // through the full tree rather than just one tree hop.
            lc_isam->update();
            lc_isam->update();
        }

        // Pull the corrected trajectory out of iSAM2 and decide whether to
        // fire `trigger_correction`. We compare the new estimates against the
        // `keyframe_poses_orig` shared with the map-correction plan; if any
        // keyframe moved more than the position / rotation thresholds, fire.
        std::vector<Eigen::Isometry3d> corrected_poses;
        {
            std::lock_guard<std::mutex> lock(mtx_isam);
            gtsam::Values est = lc_isam->calculateEstimate();
            // Poses are indexed by source_idx, which matches
            // keyframe_poses_orig's order (sequential, one per map_incremental
            // call when use_map_correction is true).
            size_t n = keyframe_poses_orig.size();
            corrected_poses.reserve(n);
            for (size_t k = 0; k < n; ++k) {
                gtsam::Symbol key('x', static_cast<int>(k));
                if (!est.exists(key)) {
                    // Fallback: if iSAM2 doesn't know this key (shouldn't happen
                    // for keys created via map_incremental), preserve the prior.
                    corrected_poses.push_back(keyframe_poses_orig[k]);
                } else {
                    corrected_poses.push_back(gtsam_to_iso(est.at<gtsam::Pose3>(key)));
                }
            }
        }

        // Coalesce: if a correction is already in flight, skip firing another.
        // The next accepted LC will re-evaluate against the post-correction
        // keyframe_poses_orig (which trigger_correction replaces on success).
        if (lc_correction_in_flight.load()) {
            RCLCPP_INFO(this->get_logger(),
                "lc: correction already in flight, coalescing");
            return;
        }

        double max_dpos = 0.0;
        double max_drot = 0.0;
        for (size_t k = 0; k < corrected_poses.size(); ++k) {
            Eigen::Isometry3d d = corrected_poses[k].inverse() * keyframe_poses_orig[k];
            max_dpos = std::max(max_dpos, d.translation().norm());
            Eigen::AngleAxisd aa(d.linear());
            max_drot = std::max(max_drot, std::abs(aa.angle()));
        }
        RCLCPP_INFO(this->get_logger(),
            "lc: PGO update max_dpos=%.3f m, max_drot=%.4f rad (thresh: %.3f / %.4f)",
            max_dpos, max_drot, lc_trigger_pos_m, lc_trigger_rot_rad);

        if (max_dpos < lc_trigger_pos_m && max_drot < lc_trigger_rot_rad) {
            return;  // sub-threshold delta; skip the expensive rebuild
        }

        // Sanity gate: if PGO wants to shift keyframes by much more than the
        // GICP-reported relative translation, the LC is inconsistent with the
        // rest of the graph — likely a false positive that slipped past the
        // per-correspondence gates. A true revisit: max_dpos ≈ |t_rel|;
        // false positive: max_dpos >> |t_rel|. Threshold conservatively at 3x.
        if (max_dpos > 3.0 * static_cast<double>(t_rel.norm())
            && max_dpos > 1.0) {
            RCLCPP_WARN(this->get_logger(),
                "lc: rejecting correction — max_dpos=%.3f >> 3*|t_rel|=%.3f (likely false LC)",
                max_dpos, 3.0 * t_rel.norm());
            return;
        }

        lc_correction_in_flight.store(true);
        int n_corrected = trigger_correction(corrected_poses);
        lc_correction_in_flight.store(false);
        RCLCPP_INFO(this->get_logger(),
            "lc: trigger_correction returned %d (shadow points)", n_corrected);
    }

    int trigger_correction(const std::vector<Eigen::Isometry3d>& corrected_poses)
    {
        int n = correct_map(corrected_poses);
        if (n <= 0) return n;

        auto stamp = this->get_clock()->now();
        publish_path_corrected_(pubPathCorrected_, frame_prefix_ + map_frame_, stamp, corrected_poses);

        PointVector all_points;
        {
            std::lock_guard<std::mutex> lock(mtx_global_map);
            global_ikdtree.flatten(global_ikdtree.Root_Node, all_points, NOT_RECORD);
        }
        publish_cloud_corrected_(pubCloudMapCorrected_, frame_prefix_ + map_frame_, stamp, all_points);

        RCLCPP_INFO(this->get_logger(),
            "correct_map applied: %zu keyframes, %zu shadow points, |t_map_odom|=%.3f m",
            corrected_poses.size(), all_points.size(), T_map_odom.translation().norm());
        return n;
    }

    void publish_shadow_map()
    {
        PointVector all_points;
        {
            std::lock_guard<std::mutex> lock(mtx_global_map);
            global_ikdtree.flatten(global_ikdtree.Root_Node, all_points, NOT_RECORD);
        }
        if (all_points.empty()) return;
        PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
        cloud->points.assign(all_points.begin(), all_points.end());
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = false;

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = frame_prefix_ + odom_frame_;
        msg.header.stamp = this->get_clock()->now();
        pubShadowMap_->publish(msg);
    }

    void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
    {
        publish_count++;
        sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

        msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
        msg->linear_acceleration.x *= G_m_s2;
        msg->linear_acceleration.y *= G_m_s2;
        msg->linear_acceleration.z *= G_m_s2;
        if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
        {
            msg->header.stamp =
                rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
        }

        double timestamp = get_time_sec(msg->header.stamp);

        mtx_buffer.lock();
        if (timestamp < last_timestamp_imu)
        {
            std::cerr << "lidar loop back, clear buffer" << std::endl;
            imu_buffer.clear();
        }
        last_timestamp_imu = timestamp;
        imu_buffer.push_back(msg);
        mtx_buffer.unlock();
        sig_buffer.notify_all();

        ImuPropState anchor;
        {
            std::lock_guard<std::mutex> lock(mtx_fwd_prop);
            anchor = fwd_prop_anchor;
        }
        if (!anchor.valid) return;

        double imu_time = get_time_sec(msg->header.stamp);
        double dt = imu_time - anchor.timestamp;
        if (dt <= 0.0) return;

        V3D acc_cur(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
        V3D gyr_cur(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

        V3D omega = gyr_cur - anchor.bg;
        V3D a_body = acc_cur - anchor.ba;
        V3D a_world = anchor.rot * a_body + anchor.grav;

        V3D prop_pos = anchor.pos + anchor.vel * dt + 0.5 * a_world * dt * dt;
        V3D prop_vel = anchor.vel + a_world * dt;
        M3D prop_rot = anchor.rot * Exp(omega, dt);

        if (dt > 0.5) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "EKF anchor stale (%.2fs), chaining IMU propagation", dt);
            std::lock_guard<std::mutex> lock(mtx_fwd_prop);
            fwd_prop_anchor.pos = prop_pos;
            fwd_prop_anchor.vel = prop_vel;
            fwd_prop_anchor.rot = prop_rot;
            fwd_prop_anchor.timestamp = imu_time;
        }

        Eigen::Quaterniond q(prop_rot);
        q.normalize();
        if (!quat_filter_init_) {
            q_filtered_ = q;
            quat_filter_init_ = true;
        } else {
            q_filtered_ = q_filtered_.slerp(vel_filter_alpha_, q);
        }

        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = frame_prefix_ + odom_frame_;
        odom.child_frame_id = frame_prefix_ + body_frame_;
        odom.header.stamp = msg->header.stamp;
        odom.pose.pose.position.x = prop_pos(0);
        odom.pose.pose.position.y = prop_pos(1);
        odom.pose.pose.position.z = prop_pos(2);
        odom.pose.pose.orientation.x = q_filtered_.x();
        odom.pose.pose.orientation.y = q_filtered_.y();
        odom.pose.pose.orientation.z = q_filtered_.z();
        odom.pose.pose.orientation.w = q_filtered_.w();

        Eigen::Matrix<double, 6, 23> J = Eigen::Matrix<double, 6, 23>::Zero();
        J.block<3,3>(0, 0) = M3D::Identity();
        J.block<3,3>(0, 3) = -anchor.rot * skew_sym_mat(a_body) * 0.5 * dt * dt;
        J.block<3,3>(0, 12) = M3D::Identity() * dt;
        J.block<3,3>(0, 18) = -anchor.rot * 0.5 * dt * dt;
        J.block<3,3>(3, 3) = M3D::Identity();
        J.block<3,3>(3, 15) = -M3D::Identity() * dt;

        Eigen::Matrix<double, 6, 6> P_pose = J * anchor.P * J.transpose();

        constexpr double imu_dt = 0.005;
        int n_samples = std::max(1, static_cast<int>(std::round(dt / imu_dt)));
        double sample_dt = dt / n_samples;
        double gyr_noise = n_samples * sample_dt * sample_dt * gyr_cov;
        double acc_noise = acc_cov * sample_dt * dt * dt * dt / 3.0;
        P_pose(0,0) += acc_noise;
        P_pose(1,1) += acc_noise;
        P_pose(2,2) += acc_noise;
        P_pose(3,3) += gyr_noise;
        P_pose(4,4) += gyr_noise;
        P_pose(5,5) += gyr_noise;

        M3D R = anchor.rot;
        P_pose.block<3,3>(3, 3) = R * P_pose.block<3,3>(3, 3) * R.transpose();
        P_pose.block<3,3>(0, 3) = P_pose.block<3,3>(0, 3) * R.transpose();
        P_pose.block<3,3>(3, 0) = R * P_pose.block<3,3>(3, 0);

        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                odom.pose.covariance[i * 6 + j] = P_pose(i, j);
            }
        }

        V3D vel_body = prop_rot.transpose() * prop_vel;
        if (!vel_filter_init_) {
            vel_filtered_ = vel_body;
            vel_filter_init_ = true;
            RCLCPP_INFO(this->get_logger(), "Velocity filter alpha: %.4f", vel_filter_alpha_);
        } else {
            vel_filtered_ = vel_filter_alpha_ * vel_body + (1.0 - vel_filter_alpha_) * vel_filtered_;
        }
        odom.twist.twist.linear.x = vel_filtered_(0);
        odom.twist.twist.linear.y = vel_filtered_(1);
        odom.twist.twist.linear.z = vel_filtered_(2);
        if (!omega_filter_init_) {
            omega_filtered_ = omega;
            omega_filter_init_ = true;
        } else {
            omega_filtered_ = vel_filter_alpha_ * omega + (1.0 - vel_filter_alpha_) * omega_filtered_;
        }
        odom.twist.twist.angular.x = omega_filtered_(0);
        odom.twist.twist.angular.y = omega_filtered_(1);
        odom.twist.twist.angular.z = omega_filtered_(2);

        pubOdomAftMapped_->publish(odom);

        if (tf_pub_en_) {
            geometry_msgs::msg::TransformStamped trans;
            trans.header.frame_id = frame_prefix_ + odom_frame_;
            trans.child_frame_id = frame_prefix_ + body_frame_;
            trans.header.stamp = msg->header.stamp;
            trans.transform.translation.x = prop_pos(0);
            trans.transform.translation.y = prop_pos(1);
            trans.transform.translation.z = prop_pos(2);
            trans.transform.rotation.x = q.x();
            trans.transform.rotation.y = q.y();
            trans.transform.rotation.z = q.z();
            trans.transform.rotation.w = q.w();
            tf_broadcaster_->sendTransform(trans);
        }

        if (use_map_correction) {
            Eigen::Isometry3d T_odom_base = Eigen::Isometry3d::Identity();
            T_odom_base.linear() = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()).toRotationMatrix();
            T_odom_base.translation() = Eigen::Vector3d(prop_pos(0), prop_pos(1), prop_pos(2));
            Eigen::Isometry3d T_map_base = T_map_odom * T_odom_base;

            nav_msgs::msg::Odometry odom_map = odom;
            odom_map.header.frame_id = frame_prefix_ + map_frame_;
            odom_map.pose.pose.position.x = T_map_base.translation().x();
            odom_map.pose.pose.position.y = T_map_base.translation().y();
            odom_map.pose.pose.position.z = T_map_base.translation().z();
            Eigen::Quaterniond q_map(T_map_base.linear());
            q_map.normalize();
            odom_map.pose.pose.orientation.x = q_map.x();
            odom_map.pose.pose.orientation.y = q_map.y();
            odom_map.pose.pose.orientation.z = q_map.z();
            odom_map.pose.pose.orientation.w = q_map.w();
            pubOdomMap_->publish(odom_map);

            if (tf_pub_en_) {
                geometry_msgs::msg::TransformStamped tf_map_odom;
                tf_map_odom.header.frame_id = frame_prefix_ + map_frame_;
                tf_map_odom.child_frame_id = frame_prefix_ + odom_frame_;
                tf_map_odom.header.stamp = msg->header.stamp;
                tf_map_odom.transform.translation.x = T_map_odom.translation().x();
                tf_map_odom.transform.translation.y = T_map_odom.translation().y();
                tf_map_odom.transform.translation.z = T_map_odom.translation().z();
                Eigen::Quaterniond q_mo(T_map_odom.linear());
                q_mo.normalize();
                tf_map_odom.transform.rotation.x = q_mo.x();
                tf_map_odom.transform.rotation.y = q_mo.y();
                tf_map_odom.transform.rotation.z = q_mo.z();
                tf_map_odom.transform.rotation.w = q_mo.w();
                tf_broadcaster_->sendTransform(tf_map_odom);
            }
        }

        if (path_en && publish_count % 20 == 0) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = msg->header.stamp;
            pose.header.frame_id = frame_prefix_ + odom_frame_;
            pose.pose.position.x = prop_pos(0);
            pose.pose.position.y = prop_pos(1);
            pose.pose.position.z = prop_pos(2);
            pose.pose.orientation.x = q.x();
            pose.pose.orientation.y = q.y();
            pose.pose.orientation.z = q.z();
            pose.pose.orientation.w = q.w();
            path.poses.push_back(pose);
            pubPath_->publish(path);
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubShadowMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomMap_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPathCorrected_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudMapCorrected_;

    string frame_prefix_;
    string odom_frame_;
    string body_frame_;
    string map_frame_;
    bool tf_pub_en_ = true;
    double vel_filter_alpha_ = 1.0;
    V3D vel_filtered_ = V3D::Zero();
    V3D omega_filtered_ = V3D::Zero();
    bool vel_filter_init_ = false;
    bool omega_filter_init_ = false;
    Eigen::Quaterniond q_filtered_ = Eigen::Quaterniond::Identity();
    bool quat_filter_init_ = false;
    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    auto node = std::make_shared<LaserMappingNode>();

    rclcpp::executors::SingleThreadedExecutor imu_executor;
    rclcpp::executors::SingleThreadedExecutor main_executor;

    imu_executor.add_callback_group(
        node->get_imu_callback_group(),
        node->get_node_base_interface());
    main_executor.add_callback_group(
        node->get_node_base_interface()->get_default_callback_group(),
        node->get_node_base_interface());

    std::thread imu_thread([&imu_executor]() {
        imu_executor.spin();
    });

    main_executor.spin();

    imu_executor.cancel();
    if (imu_thread.joinable())
        imu_thread.join();

    if (rclcpp::ok())
        rclcpp::shutdown();
    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}

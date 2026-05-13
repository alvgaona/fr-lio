#include <rclcpp/rclcpp.hpp>
#include <mocap4r2_msgs/msg/rigid_bodies.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

class MocapConverter : public rclcpp::Node
{
public:
    MocapConverter() : Node("mocap_converter")
    {
        this->declare_parameter<std::string>("rigid_body_name", "");
        this->declare_parameter<std::string>("mocap_topic", "/mocap/rigid_bodies");
        this->declare_parameter<std::string>("odom_frame", "map");

        rigid_body_name_ = this->get_parameter("rigid_body_name").as_string();
        auto mocap_topic = this->get_parameter("mocap_topic").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();

        RCLCPP_INFO(this->get_logger(), "Mocap converter: rigid_body_name='%s', topic='%s', frame='%s'",
                     rigid_body_name_.c_str(), mocap_topic.c_str(), odom_frame_.c_str());

        sub_ = this->create_subscription<mocap4r2_msgs::msg::RigidBodies>(
            mocap_topic, 10,
            std::bind(&MocapConverter::callback, this, std::placeholders::_1));

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/ground_truth/odom", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/ground_truth/path", 10);

        path_msg_.header.frame_id = odom_frame_;
    }

private:
    void callback(const mocap4r2_msgs::msg::RigidBodies::SharedPtr msg)
    {
        const mocap4r2_msgs::msg::RigidBody* body = nullptr;

        if (rigid_body_name_.empty()) {
            if (msg->rigidbodies.empty()) return;
            body = &msg->rigidbodies[0];
        } else {
            for (const auto& rb : msg->rigidbodies) {
                if (rb.rigid_body_name == rigid_body_name_) {
                    body = &rb;
                    break;
                }
            }
            if (!body) {
                RCLCPP_WARN_ONCE(this->get_logger(), "Rigid body '%s' not found", rigid_body_name_.c_str());
                return;
            }
        }

        auto stamp = msg->header.stamp;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id = "ground_truth";
        odom.pose.pose = body->pose;

        const double t = rclcpp::Time(stamp).seconds();
        if (have_prev_) {
            const double dt = t - prev_t_;
            if (dt > 1e-4 && dt < 0.5) {
                const auto& p = body->pose.position;
                const auto& q = body->pose.orientation;

                const double dvx = (p.x - prev_p_[0]) / dt;
                const double dvy = (p.y - prev_p_[1]) / dt;
                const double dvz = (p.z - prev_p_[2]) / dt;

                // Rotate world-frame linear velocity into body frame (child_frame_id).
                // R^T * v, where R is built from the current quaternion.
                const double qx = q.x, qy = q.y, qz = q.z, qw = q.w;
                const double r00 = 1 - 2*(qy*qy + qz*qz);
                const double r01 = 2*(qx*qy - qz*qw);
                const double r02 = 2*(qx*qz + qy*qw);
                const double r10 = 2*(qx*qy + qz*qw);
                const double r11 = 1 - 2*(qx*qx + qz*qz);
                const double r12 = 2*(qy*qz - qx*qw);
                const double r20 = 2*(qx*qz - qy*qw);
                const double r21 = 2*(qy*qz + qx*qw);
                const double r22 = 1 - 2*(qx*qx + qy*qy);
                odom.twist.twist.linear.x = r00*dvx + r10*dvy + r20*dvz;
                odom.twist.twist.linear.y = r01*dvx + r11*dvy + r21*dvz;
                odom.twist.twist.linear.z = r02*dvx + r12*dvy + r22*dvz;

                // ω_body ≈ 2 · vec(q_prev* ⊗ q_curr) / dt
                const double pw = prev_q_[3], px = -prev_q_[0], py = -prev_q_[1], pz = -prev_q_[2];
                const double dqw = pw*qw - px*qx - py*qy - pz*qz;
                double dqx = pw*qx + px*qw + py*qz - pz*qy;
                double dqy = pw*qy - px*qz + py*qw + pz*qx;
                double dqz = pw*qz + px*qy - py*qx + pz*qw;
                if (dqw < 0) { dqx = -dqx; dqy = -dqy; dqz = -dqz; }
                odom.twist.twist.angular.x = 2.0 * dqx / dt;
                odom.twist.twist.angular.y = 2.0 * dqy / dt;
                odom.twist.twist.angular.z = 2.0 * dqz / dt;
            }
        }
        prev_p_[0] = body->pose.position.x;
        prev_p_[1] = body->pose.position.y;
        prev_p_[2] = body->pose.position.z;
        prev_q_[0] = body->pose.orientation.x;
        prev_q_[1] = body->pose.orientation.y;
        prev_q_[2] = body->pose.orientation.z;
        prev_q_[3] = body->pose.orientation.w;
        prev_t_ = t;
        have_prev_ = true;

        odom_pub_->publish(odom);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = stamp;
        pose_stamped.header.frame_id = odom_frame_;
        pose_stamped.pose = body->pose;
        path_msg_.poses.push_back(pose_stamped);
        path_msg_.header.stamp = stamp;
        path_pub_->publish(path_msg_);
    }

    rclcpp::Subscription<mocap4r2_msgs::msg::RigidBodies>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    nav_msgs::msg::Path path_msg_;
    std::string odom_frame_;
    std::string rigid_body_name_;

    bool have_prev_ = false;
    double prev_t_ = 0.0;
    double prev_p_[3] = {0, 0, 0};
    double prev_q_[4] = {0, 0, 0, 1};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MocapConverter>());
    rclcpp::shutdown();
    return 0;
}

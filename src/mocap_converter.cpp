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
        this->declare_parameter<int>("rigid_body_index", 0);
        this->declare_parameter<std::string>("mocap_topic", "/mocap/rigid_bodies");
        this->declare_parameter<std::string>("odom_frame", "map");

        rigid_body_index_ = this->get_parameter("rigid_body_index").as_int();
        auto mocap_topic = this->get_parameter("mocap_topic").as_string();
        odom_frame_ = this->get_parameter("odom_frame").as_string();

        RCLCPP_INFO(this->get_logger(), "Mocap converter: rigid_body_index=%d, topic='%s', frame='%s'",
                     rigid_body_index_, mocap_topic.c_str(), odom_frame_.c_str());

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
        if (rigid_body_index_ >= static_cast<int>(msg->rigidbodies.size())) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Rigid body index %d out of range (size: %zu)",
                             rigid_body_index_, msg->rigidbodies.size());
            return;
        }

        const auto& pose = msg->rigidbodies.at(rigid_body_index_).pose;
        auto stamp = msg->header.stamp;

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = odom_frame_;
        odom.child_frame_id = "ground_truth";
        odom.pose.pose = pose;
        odom_pub_->publish(odom);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = stamp;
        pose_stamped.header.frame_id = odom_frame_;
        pose_stamped.pose = pose;
        path_msg_.poses.push_back(pose_stamped);
        path_msg_.header.stamp = stamp;
        path_pub_->publish(path_msg_);
    }

    rclcpp::Subscription<mocap4r2_msgs::msg::RigidBodies>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    nav_msgs::msg::Path path_msg_;
    std::string odom_frame_;
    int rigid_body_index_ = 0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MocapConverter>());
    rclcpp::shutdown();
    return 0;
}

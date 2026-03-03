#include <rclcpp/rclcpp.hpp>
#include "livox_ros_driver2/msg/custom_msg.hpp"

class LidarAccumulator : public rclcpp::Node
{
public:
    LidarAccumulator() : Node("lidar_accumulator")
    {
        this->declare_parameter<int>("accumulate_count", 10);
        this->declare_parameter<std::string>("input_topic", "/livox/lidar");
        this->declare_parameter<std::string>("output_topic", "/livox/lidar_accumulated");

        accumulate_count_ = this->get_parameter("accumulate_count").as_int();
        auto input_topic = this->get_parameter("input_topic").as_string();
        auto output_topic = this->get_parameter("output_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "Accumulating %d scans from '%s' -> '%s'",
                     accumulate_count_, input_topic.c_str(), output_topic.c_str());

        pub_ = this->create_publisher<livox_ros_driver2::msg::CustomMsg>(output_topic, 20);
        sub_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            input_topic, rclcpp::SensorDataQoS(),
            std::bind(&LidarAccumulator::callback, this, std::placeholders::_1));
    }

private:
    void callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        if (count_ == 0) {
            accumulated_ = *msg;
        } else {
            uint64_t time_offset_ns = msg->timebase - accumulated_.timebase;
            uint32_t time_offset_ns32 = static_cast<uint32_t>(
                std::min(time_offset_ns, static_cast<uint64_t>(UINT32_MAX)));

            for (auto &pt : msg->points) {
                auto new_pt = pt;
                new_pt.offset_time += time_offset_ns32;
                accumulated_.points.push_back(new_pt);
            }
            accumulated_.point_num += msg->point_num;
        }

        count_++;
        if (count_ >= accumulate_count_) {
            pub_->publish(accumulated_);
            count_ = 0;
        }
    }

    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_;
    rclcpp::Publisher<livox_ros_driver2::msg::CustomMsg>::SharedPtr pub_;
    livox_ros_driver2::msg::CustomMsg accumulated_;
    int accumulate_count_ = 10;
    int count_ = 0;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarAccumulator>());
    rclcpp::shutdown();
    return 0;
}

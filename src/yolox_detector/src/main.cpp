#include "yolox_detector/yolox_detector_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolox_detector::YoloxDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

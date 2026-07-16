#include "collision_classifier/collision_classifier_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<collision_classifier::CollisionClassifierNode>());
  rclcpp::shutdown();
  return 0;
}

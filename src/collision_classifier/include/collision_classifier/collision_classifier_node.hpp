#pragma once
#include "collision_classifier/inference_backend.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/classification.hpp>
#include <image_transport/image_transport.hpp>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace collision_classifier {

// Runs a binary free/blocked image classifier (ResNet18 by default) through
// ExecuTorch and publishes the softmax probabilities as a
// vision_msgs/Classification on /collision/classification. Downstream, the
// collision_avoider controller drives the robot forward when free and turns in
// place when blocked.
class CollisionClassifierNode : public rclcpp::Node {
public:
  explicit CollisionClassifierNode(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  std::unique_ptr<InferenceBackend> backend_;

  // Class order matches torchvision ImageFolder (alphabetical): blocked, free.
  std::vector<std::string> labels_{"blocked", "free"};
  int blocked_index_{0};
  std::array<float, 3> mean_{{0.485f, 0.456f, 0.406f}};
  std::array<float, 3> std_{{0.229f, 0.224f, 0.225f}};

  image_transport::Subscriber image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Classification>::SharedPtr class_pub_;
  image_transport::Publisher debug_pub_;
};

}  // namespace collision_classifier

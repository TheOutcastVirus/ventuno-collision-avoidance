#pragma once
#include "yolox_detector/inference_backend.hpp"
#include "yolox_detector/postprocess.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>

namespace yolox_detector {

class YoloxDetectorNode : public rclcpp::Node {
public:
  explicit YoloxDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  std::unique_ptr<InferenceBackend> backend_;

  float score_threshold_{0.45f};
  float nms_threshold_{0.45f};
  int num_classes_{80};

  image_transport::Subscriber image_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
  image_transport::Publisher debug_pub_;
};

}  // namespace yolox_detector

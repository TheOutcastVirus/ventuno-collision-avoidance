#include "collision_classifier/collision_classifier_node.hpp"
#include "collision_classifier/cpu_backend.hpp"
#include "collision_classifier/preprocess.hpp"
#ifdef HAVE_QNN_BACKEND
#include "collision_classifier/qnn_backend.hpp"
#endif
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <vision_msgs/msg/object_hypothesis.hpp>
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace collision_classifier {

CollisionClassifierNode::CollisionClassifierNode(const rclcpp::NodeOptions & options)
: Node("collision_classifier", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────────
  this->declare_parameter("backend", std::string("cpu"));
  this->declare_parameter("model_path", std::string(""));
  this->declare_parameter("qnn_lib_dir", std::string(""));
  this->declare_parameter("input_width", 224);
  this->declare_parameter("input_height", 224);
  this->declare_parameter("image_topic", std::string("/oak/rgb/image_raw"));
  this->declare_parameter("classification_topic", std::string("/collision/classification"));
  this->declare_parameter("debug_image_topic", std::string("/collision/image"));
  this->declare_parameter("labels", std::vector<std::string>{"blocked", "free"});
  this->declare_parameter("blocked_index", 0);
  this->declare_parameter("mean", std::vector<double>{0.485, 0.456, 0.406});
  this->declare_parameter("std", std::vector<double>{0.229, 0.224, 0.225});

  auto backend_name = this->get_parameter("backend").as_string();
  auto model_path   = this->get_parameter("model_path").as_string();
  auto qnn_lib_dir  = this->get_parameter("qnn_lib_dir").as_string();
  int in_w = this->get_parameter("input_width").as_int();
  int in_h = this->get_parameter("input_height").as_int();
  labels_ = this->get_parameter("labels").as_string_array();
  blocked_index_ = this->get_parameter("blocked_index").as_int();

  auto mean_v = this->get_parameter("mean").as_double_array();
  auto std_v  = this->get_parameter("std").as_double_array();
  if (mean_v.size() != 3 || std_v.size() != 3) {
    throw std::invalid_argument("mean and std must each have 3 elements (RGB)");
  }
  for (int i = 0; i < 3; ++i) {
    mean_[i] = static_cast<float>(mean_v[i]);
    std_[i]  = static_cast<float>(std_v[i]);
  }
  if (labels_.empty()) {
    throw std::invalid_argument("labels must not be empty");
  }
  if (blocked_index_ < 0 || blocked_index_ >= static_cast<int>(labels_.size())) {
    throw std::invalid_argument("blocked_index is out of range for labels");
  }

  // ── Backend ──────────────────────────────────────────────────────────────────
  if (model_path.empty()) {
    if (backend_name == "cpu") {
      model_path = "models/collision_resnet18_xnnpack.pte";
    } else if (backend_name == "npu") {
      model_path = "models/collision_resnet18_qnn.pte";
    }
  }
  if (backend_name == "cpu") {
    backend_ = std::make_unique<CpuBackend>(in_w, in_h);
  } else if (backend_name == "npu") {
#ifdef HAVE_QNN_BACKEND
    backend_ = std::make_unique<QnnBackend>(in_w, in_h, qnn_lib_dir);
#else
    throw std::runtime_error(
      "collision_classifier was compiled without QNN backend support. "
      "Rebuild with -DBUILD_QNN_BACKEND=ON.");
#endif
  } else {
    throw std::invalid_argument("Unknown backend '" + backend_name +
                                "'. Choose 'cpu' or 'npu'.");
  }

  if (!backend_->load(model_path)) {
    throw std::runtime_error("Failed to load model from: " + model_path);
  }
  RCLCPP_INFO(this->get_logger(), "Loaded model '%s' on '%s' backend (%dx%d)",
              model_path.c_str(), backend_name.c_str(), in_w, in_h);

  // ── Publishers ──────────────────────────────────────────────────────────────
  auto class_topic = this->get_parameter("classification_topic").as_string();
  auto debug_topic = this->get_parameter("debug_image_topic").as_string();
  class_pub_ = this->create_publisher<vision_msgs::msg::Classification>(class_topic, 10);
  debug_pub_ = image_transport::create_publisher(this, debug_topic);

  // ── Subscriber ──────────────────────────────────────────────────────────────
  auto img_topic = this->get_parameter("image_topic").as_string();
  image_sub_ = image_transport::create_subscription(
    this, img_topic,
    std::bind(&CollisionClassifierNode::image_callback, this, std::placeholders::_1),
    "raw");
  RCLCPP_INFO(this->get_logger(), "Subscribing to '%s'", img_topic.c_str());
}

void CollisionClassifierNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  cv_bridge::CvImageConstPtr cv_img;
  try {
    cv_img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge: %s", e.what());
    return;
  }

  cv::Mat input = preprocess(cv_img->image, backend_->input_width(),
                             backend_->input_height(), mean_, std_);

  std::vector<float> logits;
  try {
    logits = backend_->infer(input);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
    return;
  }

  if (logits.size() != labels_.size()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Model produced %zu outputs but %zu labels are configured",
                 logits.size(), labels_.size());
    return;
  }

  std::vector<float> probs = softmax(logits);
  float prob_blocked = probs[blocked_index_];

  // ── Publish classification ──────────────────────────────────────────────────
  vision_msgs::msg::Classification cls;
  cls.header = msg->header;
  for (size_t i = 0; i < labels_.size(); ++i) {
    vision_msgs::msg::ObjectHypothesis hyp;
    hyp.class_id = labels_[i];
    hyp.score = probs[i];
    cls.results.push_back(hyp);
  }
  class_pub_->publish(cls);

  // ── Publish debug overlay ───────────────────────────────────────────────────
  if (debug_pub_.getNumSubscribers() > 0) {
    cv::Mat dbg = cv_img->image.clone();
    bool blocked = prob_blocked >= 0.5f;
    cv::Scalar color = blocked ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);
    char label[64];
    std::snprintf(label, sizeof(label), "%s %.0f%%",
                  blocked ? labels_[blocked_index_].c_str() : "free",
                  (blocked ? prob_blocked : (1.0f - prob_blocked)) * 100.f);
    cv::rectangle(dbg, cv::Point(0, 0), cv::Point(dbg.cols - 1, dbg.rows - 1),
                  color, 8);
    cv::putText(dbg, label, cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                color, 2, cv::LINE_AA);
    auto dbg_msg = cv_bridge::CvImage(msg->header, "bgr8", dbg).toImageMsg();
    debug_pub_.publish(*dbg_msg);
  }
}

}  // namespace collision_classifier

#include "collision_classifier/preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace collision_classifier {

cv::Mat preprocess(const cv::Mat & bgr, int out_w, int out_h,
                   const std::array<float, 3> & mean,
                   const std::array<float, 3> & std)
{
  // Resize to the model's input size (plain resize, matching torchvision
  // transforms.Resize used at training time — no aspect-ratio letterbox).
  cv::Mat resized;
  cv::resize(bgr, resized, {out_w, out_h}, 0, 0, cv::INTER_LINEAR);

  // BGR (OpenCV/ROS) → RGB (torchvision) and scale to [0, 1].
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // Per-channel ImageNet normalization: (x - mean) / std. After the BGR→RGB
  // conversion channel 0 = R, 1 = G, 2 = B, matching the mean/std order.
  std::vector<cv::Mat> channels(3);
  cv::split(rgb, channels);
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - mean[c]) / std[c];
  }
  cv::Mat out;
  cv::merge(channels, out);
  return out;
}

std::vector<float> softmax(const std::vector<float> & logits)
{
  std::vector<float> probs(logits.size());
  if (logits.empty()) return probs;

  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum > 0.f) {
    for (float & p : probs) p /= sum;
  }
  return probs;
}

}  // namespace collision_classifier

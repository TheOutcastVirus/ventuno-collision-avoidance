#pragma once
#include <array>
#include <opencv2/core.hpp>
#include <vector>

namespace collision_classifier {

// Preprocess a BGR8 camera frame into the tensor layout a torchvision
// classifier (e.g. ResNet18) expects, matching the training transforms in
// tools/train_collision_resnet18.py:
//   1. resize to (out_w × out_h)
//   2. BGR → RGB
//   3. scale to [0, 1] (divide by 255)
//   4. normalize per channel: (x - mean) / std   (ImageNet stats by default)
// Returns a CV_32FC3 image with channels in RGB order, ready for
// InferenceBackend::infer(). `mean`/`std` are in RGB order.
cv::Mat preprocess(const cv::Mat & bgr, int out_w, int out_h,
                   const std::array<float, 3> & mean,
                   const std::array<float, 3> & std);

// Numerically stable softmax over a small logit vector (the classifier's
// per-class outputs). Returns probabilities that sum to 1.
std::vector<float> softmax(const std::vector<float> & logits);

}  // namespace collision_classifier

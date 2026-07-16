#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace collision_classifier {

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;

  // Load model from .pte file. Returns true on success.
  virtual bool load(const std::string & model_path) = 0;

  // Run inference on a pre-processed float32 image (already resized and
  // normalized to the model's input size, with channels in RGB order).
  // Returns the raw output tensor values as a flat vector — for the collision
  // classifier these are the per-class logits [num_classes] (free, blocked).
  virtual std::vector<float> infer(const cv::Mat & input_f32) = 0;

  virtual int input_width() const = 0;
  virtual int input_height() const = 0;
};

}  // namespace collision_classifier

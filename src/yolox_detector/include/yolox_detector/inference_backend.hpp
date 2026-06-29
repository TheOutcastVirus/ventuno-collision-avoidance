#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace yolox_detector {

class InferenceBackend {
public:
  virtual ~InferenceBackend() = default;

  // Load model from .pte file. Returns true on success.
  virtual bool load(const std::string & model_path) = 0;

  // Run inference on a pre-processed BGR float32 image (already resized to
  // the model's input size).  Returns the raw output tensor values as a flat
  // vector in the layout [num_anchors, 5 + num_classes].
  virtual std::vector<float> infer(const cv::Mat & input_f32) = 0;

  virtual int input_width() const = 0;
  virtual int input_height() const = 0;
};

}  // namespace yolox_detector

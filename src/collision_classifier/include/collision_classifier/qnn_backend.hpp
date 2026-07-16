#pragma once
#include "collision_classifier/inference_backend.hpp"
#include <executorch/extension/module/module.h>
#include <memory>
#include <string>
#include <vector>

namespace collision_classifier {

// QTI HTP (Hexagon) backend via ExecuTorch QNN delegate.
// The model must have been exported with QnnPartitioner + QnnQuantizer.
// libQnnHtp.so and friends must be on LD_LIBRARY_PATH, or qnn_lib_dir must
// be passed to the constructor so we can prepend it at runtime.
class QnnBackend : public InferenceBackend {
public:
  QnnBackend(int width, int height, const std::string & qnn_lib_dir = "");

  bool load(const std::string & model_path) override;
  std::vector<float> infer(const cv::Mat & input_f32) override;

  int input_width() const override { return width_; }
  int input_height() const override { return height_; }

private:
  int width_;
  int height_;
  std::string qnn_lib_dir_;
  std::unique_ptr<executorch::extension::Module> module_;
};

}  // namespace collision_classifier

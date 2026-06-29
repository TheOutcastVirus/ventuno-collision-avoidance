#pragma once
#include "yolox_detector/inference_backend.hpp"
#include <executorch/extension/module/module.h>
#include <memory>
#include <string>
#include <vector>

namespace yolox_detector {

class CpuBackend : public InferenceBackend {
public:
  CpuBackend(int width, int height);

  bool load(const std::string & model_path) override;
  std::vector<float> infer(const cv::Mat & input_f32) override;

  int input_width() const override { return width_; }
  int input_height() const override { return height_; }

private:
  int width_;
  int height_;
  std::unique_ptr<executorch::extension::Module> module_;
};

}  // namespace yolox_detector

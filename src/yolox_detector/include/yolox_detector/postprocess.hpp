#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace yolox_detector {

struct Detection {
  float x1, y1, x2, y2;  // pixel coordinates in original image space
  float score;
  int class_id;
};

// Letterbox-resize src into a dst of (out_w × out_h).
// Fills border with 114 and stores the scale + offset used so detections can
// be mapped back to the original image.
cv::Mat letterbox(const cv::Mat & src, int out_w, int out_h,
                  float & scale, float & pad_left, float & pad_top);

// Convert the raw YOLOX output tensor (anchor-free, three strides: 8, 16, 32)
// into a list of Detections in original-image pixel coordinates.
//   raw_output  : flat float32 buffer [num_anchors × (5 + num_classes)]
//   num_anchors : total number of decoded anchors
//   num_classes : number of object classes
//   in_w, in_h  : model input size used during inference
//   scale       : scale factor returned by letterbox()
//   pad_left, pad_top : padding returned by letterbox()
//   score_thr   : objectness * class_score threshold
//   nms_thr     : IoU threshold for NMS
std::vector<Detection> decode_and_nms(
  const std::vector<float> & raw_output,
  int num_anchors, int num_classes,
  int in_w, int in_h,
  float scale, float pad_left, float pad_top,
  float score_thr, float nms_thr);

}  // namespace yolox_detector

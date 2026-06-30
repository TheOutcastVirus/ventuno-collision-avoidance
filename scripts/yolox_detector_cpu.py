#!/usr/bin/env python3
"""YOLOX-Tiny CPU detector — ROS 2 node using ONNX Runtime.

No ExecuTorch dependency. Subscribes to an Image topic, runs YOLOX-Tiny
inference via onnxruntime, and publishes Detection2DArray.

If the ONNX model file is missing, the node will automatically run
scripts/download_models.py to fetch and export it before starting.

See YOLOX_CPU_ONNX.md for full setup instructions.
"""
import os
import subprocess
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DOWNLOAD_SCRIPT = os.path.join(_SCRIPT_DIR, "download_models.py")


def _ensure_model(model_path: str) -> None:
    """Auto-download + export the ONNX model if it doesn't exist."""
    if os.path.exists(model_path):
        return
    print(f"[yolox_detector] Model not found: {model_path}", flush=True)
    print("[yolox_detector] Running download_models.py ...", flush=True)
    result = subprocess.run(
        [sys.executable, _DOWNLOAD_SCRIPT, "--models-dir",
         os.path.join(_REPO_ROOT, "models")],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Model download/export failed. "
            "Install export deps:\n"
            "  pip install torch torchvision\n"
            "  pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git\n"
            "Then re-run the node, or run scripts/download_models.py manually."
        )


class YoloxDetectorNode(Node):
    def __init__(self):
        super().__init__("yolox_detector")

        self.declare_parameter("model_path", os.path.join(_REPO_ROOT, "models", "yolox_tiny.onnx"))
        self.declare_parameter("input_width", 416)
        self.declare_parameter("input_height", 416)
        self.declare_parameter("score_threshold", 0.45)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("num_classes", 80)
        self.declare_parameter("image_topic", "/camera/rgb/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("debug_image_topic", "/detections/image")

        model_path = self.get_parameter("model_path").value
        self.input_w = self.get_parameter("input_width").value
        self.input_h = self.get_parameter("input_height").value
        self.score_thresh = self.get_parameter("score_threshold").value
        self.nms_thresh = self.get_parameter("nms_threshold").value

        _ensure_model(model_path)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        self.bridge = CvBridge()

        image_topic = self.get_parameter("image_topic").value
        det_topic = self.get_parameter("detections_topic").value
        dbg_topic = self.get_parameter("debug_image_topic").value

        self.sub = self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.det_pub = self.create_publisher(Detection2DArray, det_topic, 10)
        self.dbg_pub = self.create_publisher(Image, dbg_topic, 10)

        self.get_logger().info(f"Model loaded: {model_path}")

    def _letterbox(self, img, target_w, target_h):
        """Resize with aspect-ratio padding (gray fill 114/255)."""
        ih, iw = img.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_w - nw) // 2
        pad_y = (target_h - nh) // 2
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _decode(self, raw, scale, pad_x, pad_y, orig_w, orig_h):
        """
        Decode anchor-free YOLOX output (decode_in_inference=False).

        raw: [num_preds, 5 + num_classes]
          [0]: x offset in feature-map cells  [1]: y offset
          [2]: log width                       [3]: log height
          [4]: objectness logit                [5:]: class logits
        """
        strides = [8, 16, 32]
        detections = []
        offset = 0

        for stride in strides:
            gh = self.input_h // stride
            gw = self.input_w // stride
            n = gh * gw

            pred = raw[offset : offset + n]
            offset += n

            gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            gx = gx.ravel()
            gy = gy.ravel()

            cx = (pred[:, 0] + gx) * stride
            cy = (pred[:, 1] + gy) * stride
            w = np.exp(np.clip(pred[:, 2], -10, 10)) * stride
            h = np.exp(np.clip(pred[:, 3], -10, 10)) * stride

            obj = 1.0 / (1.0 + np.exp(-pred[:, 4]))
            cls_scores = 1.0 / (1.0 + np.exp(-pred[:, 5:]))

            scores = obj[:, None] * cls_scores
            class_ids = scores.argmax(axis=1)
            max_scores = scores[np.arange(n), class_ids]

            mask = max_scores > self.score_thresh
            if not mask.any():
                continue

            x1 = np.clip((cx[mask] - w[mask] / 2 - pad_x) / scale, 0, orig_w)
            y1 = np.clip((cy[mask] - h[mask] / 2 - pad_y) / scale, 0, orig_h)
            x2 = np.clip((cx[mask] + w[mask] / 2 - pad_x) / scale, 0, orig_w)
            y2 = np.clip((cy[mask] + h[mask] / 2 - pad_y) / scale, 0, orig_h)

            for i in range(int(mask.sum())):
                detections.append(
                    (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                     float(max_scores[mask][i]), int(class_ids[mask][i]))
                )

        return detections

    def _nms(self, detections):
        if not detections:
            return []
        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
        scores = np.array([d[4] for d in detections], dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order):
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= self.nms_thresh]
        return [detections[i] for i in keep]

    def _image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        orig_h, orig_w = frame.shape[:2]

        letterboxed, scale, pad_x, pad_y = self._letterbox(
            frame, self.input_w, self.input_h
        )
        # Normalize to [0,1], convert HWC→NCHW
        inp = (letterboxed.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        (raw_output,) = self.session.run(None, {self.input_name: inp})
        raw = raw_output[0]  # [num_preds, 5 + num_classes]

        detections = self._nms(
            self._decode(raw, scale, pad_x, pad_y, orig_w, orig_h)
        )

        det_array = Detection2DArray()
        det_array.header = msg.header
        for x1, y1, x2, y2, score, cls_id in detections:
            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(cls_id)
            hyp.hypothesis.score = score
            det.results.append(hyp)
            det_array.detections.append(det)
        self.det_pub.publish(det_array)

        if self.dbg_pub.get_subscription_count() > 0:
            dbg = frame.copy()
            for x1, y1, x2, y2, score, cls_id in detections:
                cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    dbg, f"{cls_id}:{score:.2f}",
                    (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


def main():
    rclpy.init()
    node = YoloxDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

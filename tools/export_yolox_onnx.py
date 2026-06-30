#!/usr/bin/env python3
"""Export YOLOX-Tiny to ONNX. No ExecuTorch dependency.

Usage:
    pip install torch torchvision
    pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git
    python3 tools/export_yolox_onnx.py --weights yolox_tiny.pth --output models/yolox_tiny.onnx
"""
import argparse
import os

import torch
from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN


def build_yolox_tiny() -> YOLOX:
    depth, width = 0.33, 0.375
    backbone = YOLOPAFPN(depth=depth, width=width)
    head = YOLOXHead(num_classes=80, width=width, decode_in_inference=False)
    return YOLOX(backbone=backbone, head=head)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOX-Tiny to ONNX")
    parser.add_argument("--weights", default="yolox_tiny.pth", help="Path to .pth checkpoint")
    parser.add_argument("--output", default="models/yolox_tiny.onnx", help="Output .onnx path")
    parser.add_argument("--input-size", default=416, type=int, help="Model input resolution (square)")
    parser.add_argument("--opset", default=11, type=int, help="ONNX opset version")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = build_yolox_tiny()
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, args.input_size, args.input_size)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["images"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch_size"}},
    )

    print(f"Saved: {args.output}")
    print(f"Input  shape : [1, 3, {args.input_size}, {args.input_size}]  (NCHW, float32, 0–1 range)")
    print("Output shape : [1, num_preds, 5+num_classes]  (anchor-free, decode_in_inference=False)")


if __name__ == "__main__":
    main()

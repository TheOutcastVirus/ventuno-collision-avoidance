#!/usr/bin/env python3
"""Export the trained collision-avoidance ResNet18 to ONNX.

Handy for inspecting the graph (e.g. in Netron) or running with onnxruntime.
The ExecuTorch runtime used on the board consumes the .pte files instead — see
export_resnet18_cpu.py (XNNPACK/CPU) and export_resnet18_qnn.py (QNN/NPU).

Usage:
    python3 tools/export_resnet18_onnx.py \
        --weights models/collision_resnet18.pth --output models/collision_resnet18.onnx
"""

import argparse
import pathlib

import torch

from collision_model import load_trained


def main():
    p = argparse.ArgumentParser(description="Export collision ResNet18 to ONNX")
    p.add_argument("--weights", default="models/collision_resnet18.pth")
    p.add_argument("--output", default="models/collision_resnet18.onnx")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-size", type=int, default=224)
    args = p.parse_args()

    model = load_trained(args.weights, num_classes=args.num_classes)
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=11,
        dynamic_axes={"images": {0: "batch_size"}},
        # Use the legacy TorchScript exporter: it embeds the weights and honors
        # opset_version. The newer dynamo exporter (default on torch >= 2.9)
        # writes weights as external data, yielding a tiny weightless .onnx.
        dynamo=False,
    )
    print(f"Saved: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Lower the collision-avoidance ResNet18 to ExecuTorch .pte for the XNNPACK (CPU) backend.

This is the CPU fallback used when the classifier runs with backend:=cpu. Run on
a machine with torch + executorch installed (host or the Ventuno):
    pip install torch torchvision
    pip install "executorch[xnnpack]"

Usage:
    python3 tools/export_resnet18_cpu.py \
        --weights models/collision_resnet18.pth \
        --output models/collision_resnet18_xnnpack.pte

Output: models/collision_resnet18_xnnpack.pte (fp32)
"""

import argparse
import pathlib

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

from collision_model import load_trained


def main():
    p = argparse.ArgumentParser(description="Export collision ResNet18 to XNNPACK .pte")
    p.add_argument("--weights", default="models/collision_resnet18.pth")
    p.add_argument("--output", default="models/collision_resnet18_xnnpack.pte")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-size", type=int, default=224)
    args = p.parse_args()

    print("Loading model ...")
    model = load_trained(args.weights, num_classes=args.num_classes)

    example_input = (torch.zeros(1, 3, args.input_size, args.input_size),)

    print("Exporting to ExecuTorch IR ...")
    ep = torch.export.export(model, example_input)

    print("Lowering with XNNPACK partitioner ...")
    et_program = to_edge_transform_and_lower(
        ep,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Saved: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

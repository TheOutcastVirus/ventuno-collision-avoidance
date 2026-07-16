#!/usr/bin/env python3
"""Lower the collision-avoidance ResNet18 to ExecuTorch .pte for the Qualcomm QNN HTP (NPU) backend.

Runs on the Ventuno Q with the local ExecuTorch checkout on PYTHONPATH and the
QAIRT/QNN environment sourced (scripts/install_ventuno_deps.sh sets this up; the
board environment file exports QNN_SDK_ROOT). The model is 8-bit quantized
(8a8w) for the Hexagon HTP.

CRITICAL: quantization calibration must see the SAME normalized inputs the C++
runtime feeds the model (BGR->RGB, /255, ImageNet mean/std). Point --calibration-dir
at your collected dataset so calibration uses real normalized frames; otherwise a
normalized-noise fallback is used, which quantizes less well.

Usage (on the board):
    python3 tools/export_resnet18_qnn.py \
        --weights models/collision_resnet18.pth \
        --output models/collision_resnet18_qnn.pte \
        --calibration-dir dataset --soc-model QCS8300

Output: models/collision_resnet18_qnn.pte (8-bit, HTP-delegated)
"""

import argparse
import os
import pathlib
import random
import sys

import torch

from collision_model import eval_transform, load_trained


def _calibration_tensors(num_batches, input_size, calibration_dir):
    """Yield normalized (1,3,H,W) calibration inputs.

    Prefers real images from `calibration_dir` (recursively), transformed with
    the same eval_transform used at inference. Falls back to normalized noise
    when no images are available.
    """
    if calibration_dir and os.path.isdir(calibration_dir):
        transform = eval_transform(input_size)
        from PIL import Image

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = [p for p in pathlib.Path(calibration_dir).rglob("*")
                 if p.suffix.lower() in exts]
        random.shuffle(paths)
        used = 0
        for path in paths:
            if used >= num_batches:
                break
            try:
                img = Image.open(path).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            yield (transform(img).unsqueeze(0),)
            used += 1
        if used > 0:
            return

    print("WARNING: no calibration images found; falling back to normalized noise. "
          "Pass --calibration-dir <dataset> for a better-quantized model.",
          file=sys.stderr)
    for _ in range(num_batches):
        yield (torch.randn(1, 3, input_size, input_size),)


def main():
    p = argparse.ArgumentParser(description="Export collision ResNet18 to QNN .pte")
    p.add_argument("--weights", default="models/collision_resnet18.pth")
    p.add_argument("--output", default="models/collision_resnet18_qnn.pte")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--soc-model", default="QCS8300",
                   help="SoC model for HTP lowering (QCS8300 on the Ventuno Q)")
    p.add_argument("--build-folder",
                   default=os.path.join(
                       os.environ.get("EXECUTORCH_ROOT", "/opt/executorch"), "build-x86"))
    p.add_argument("--calibration-dir", default="dataset",
                   help="Directory of images used for quantization calibration")
    p.add_argument("--calibration-batches", type=int, default=16)
    args = p.parse_args()

    if "QNN_SDK_ROOT" not in os.environ:
        raise EnvironmentError("QNN_SDK_ROOT is not set; source the board environment first")

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from executorch.backends.qualcomm.export_utils import (
        QnnConfig,
        build_executorch_binary,
        make_quantizer,
    )
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    model = load_trained(args.weights, num_classes=args.num_classes)

    qnn_config = QnnConfig(
        soc_model=args.soc_model,
        build_folder=args.build_folder,
        backend="htp",
        target="aarch64-oe-linux",
        compile_only=False,
        enable_x86_64=True,
    )

    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_8a8w,
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=args.soc_model,
    )
    calibration = list(_calibration_tensors(
        args.calibration_batches, args.input_size, args.calibration_dir))

    # build_executorch_binary appends .pte to file_name.
    file_stem = str(output_path.with_suffix(""))
    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=file_stem,
        dataset=calibration,
        custom_quantizer=quantizer,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    sys.exit(main())

"""Export YOLOX-tiny to ExecuTorch .pte format with QTI QNN HTP backend (NPU).

Run this on a dev machine (not the robot).  Requires:
  - Qualcomm AI Engine Direct SDK (QAIRT) — set QNN_SDK_ROOT env var
  - pip install torch torchvision yolox
  - ExecuTorch built with QNN backend:
      cmake -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT ...

Output: models/yolox_tiny_qnn.pte
"""

import pathlib
import os
import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.qnn_quantizer import (
    QnnQuantizer,
    get_default_16bit_qnn_ptq_config,
)
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig


def build_model(num_classes: int = 80) -> torch.nn.Module:
    depth, width = 0.33, 0.375  # YOLOX-tiny
    backbone = YOLOPAFPN(depth, width, in_channels=[256, 512, 1024])
    head = YOLOXHead(num_classes, width, in_channels=[256, 512, 1024])
    model = YOLOX(backbone, head)
    model.eval()
    return model


def calibration_data(n: int = 200, h: int = 416, w: int = 416):
    """Yields random calibration tensors — replace with real images for best accuracy."""
    for _ in range(n):
        yield (torch.rand(1, 3, h, w),)


def main() -> None:
    input_h, input_w = 416, 416
    num_classes = 80

    qnn_sdk = os.environ.get("QNN_SDK_ROOT")
    if not qnn_sdk:
        raise EnvironmentError(
            "QNN_SDK_ROOT env var not set. "
            "Download the Qualcomm AI Engine Direct SDK and set QNN_SDK_ROOT."
        )

    print("Building model ...")
    model = build_model(num_classes)

    # Optionally load pretrained weights:
    # ckpt = torch.load("yolox_tiny.pth", map_location="cpu")
    # model.load_state_dict(ckpt["model"])

    example_input = (torch.zeros(1, 3, input_h, input_w),)

    print("Exporting to FX IR for quantization ...")
    ep = torch.export.export(model, example_input)

    # ── PTQ quantization ────────────────────────────────────────────────────────
    print("Preparing model for PTQ ...")
    quantizer = QnnQuantizer()
    quantizer.set_global_op_quant_config(get_default_16bit_qnn_ptq_config())

    prepared = prepare_pt2e(ep.module(), quantizer)

    print("Running calibration (200 batches) ...")
    with torch.no_grad():
        for cal_input in calibration_data(n=200, h=input_h, w=input_w):
            prepared(*cal_input)

    quantized = convert_pt2e(prepared)
    print("Quantization complete.")

    # Re-export the quantized model
    ep_q = torch.export.export(quantized, example_input)

    print("Lowering with QNN partitioner ...")
    et_program = to_edge_transform_and_lower(
        ep_q,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[QnnPartitioner()],
    ).to_executorch()

    out_path = pathlib.Path("models/yolox_tiny_qnn.pte")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")
    print(
        "Deploy: copy this .pte file to the Ventuno Q and set backend: npu in detector.yaml"
    )


if __name__ == "__main__":
    main()

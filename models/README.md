# Models

Collision-avoidance classifier artifacts. These are **generated** by the
training + export pipeline (see the repo [README](../README.md) and `tools/`),
not committed pre-built — collect your own data and train for your environment.

| File | Produced by | Used by |
|---|---|---|
| `collision_resnet18.pth` | `tools/train_collision_resnet18.py` | export scripts (git-ignored, not tracked) |
| `collision_resnet18.onnx` | `tools/export_resnet18_onnx.py` | inspection / onnxruntime |
| `collision_resnet18_xnnpack.pte` | `tools/export_resnet18_cpu.py` | `collision_classifier` with `backend:=cpu` |
| `collision_resnet18_qnn.pte` | `tools/export_resnet18_qnn.py` | `collision_classifier` with `backend:=npu` (Hexagon HTP) |

Naming convention: `collision_resnet18[_<backend>].<ext>`.

The classifier expects a 224×224 RGB input normalized with ImageNet mean/std;
the same normalization is applied at runtime in
`src/collision_classifier/src/preprocess.cpp` and at train/export time via
`tools/collision_model.py`. Keep them in sync.

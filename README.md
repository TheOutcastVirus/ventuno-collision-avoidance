# Ventuno Collision Avoidance

A JetBot-style **collision-avoidance** robot demo built on a
[TurtleBot 4](https://clearpathrobotics.com/turtlebot-4/), with the stock Raspberry Pi replaced
by an **Arduino Ventuno Q** board. A small neural network learns to tell "free" from "blocked"
from the camera alone, and the robot uses it to wander without bumping into things — all running
on-device on Qualcomm hardware:

- **OAK-D Lite** camera streams RGB over ROS 2
- A **ResNet18** binary classifier (`free` / `blocked`) runs on the Ventuno Q's Hexagon NPU via
  [ExecuTorch](https://github.com/pytorch/executorch) and Qualcomm's QNN runtime
- A simple reactive controller drives the Create 3 base **forward** while the path is free and
  **turns in place** when it sees an obstacle, mirroring the classic
  [JetBot collision-avoidance](https://github.com/NVIDIA-AI-IOT/jetbot) behavior

Everything runs natively on the board, no internet required. This repo lives alongside the
object-following demo (`ventuno-object-tracking`) and shares the same board setup.

## Hardware

| Component | Role |
|---|---|
| Arduino Ventuno Q | Compute: Hexagon NPU, ARM CPU, replaces the TurtleBot 4's Raspberry Pi |
| Clearpath TurtleBot 4 Lite (Create 3 base) | Mobile base |
| OAK-D Lite | RGB camera |

## Software stack

- Ubuntu 24.04 (Qualcomm image) on the Ventuno Q
- ROS 2 Jazzy
- PyTorch ExecuTorch with the Qualcomm QNN HTP backend (NPU) or XNNPACK (CPU fallback)
- DepthAI for the OAK-D Lite
- PyTorch + torchvision (host or board) for training and model export

## Quickstart

On a fresh Ventuno Q (Ubuntu 24.04):

```bash
git clone https://github.com/TheOutcastVirus/ventuno-collision-avoidance.git ~/Documents/ventuno-collision-avoidance
cd ~/Documents/ventuno-collision-avoidance
bash scripts/install_ventuno_deps.sh
```

This installs ROS/system dependencies, the QAIRT/QNN SDK, builds ExecuTorch with the Qualcomm
backend, sets up the Create 3 USB-ethernet link, and builds the ROS workspace. See
[`.claude/skills/ventuno-setup/SKILL.md`](.claude/skills/ventuno-setup/SKILL.md) for what it does
step by step and how to debug a failed run.

The demo needs a trained model. The workflow is **collect → train → lower → run**; the four
stages are below.

---

## 1. Collect data

Drive/place the robot around your environment and label snapshots as `free` (safe to move
forward) or `blocked` (obstacle, wall, or ledge ahead). This launch brings up the camera and a
keyboard-driven collector:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch collision_avoider data_collection.launch.py
```

In that terminal:

| Key | Action |
|---|---|
| `f` | save the current frame as **free** |
| `b` | save the current frame as **blocked** |
| `q` | quit |

Images are written to `dataset/free/` and `dataset/blocked/` as 224×224 JPEGs, and the running
counts are printed after each save. Aim for **varied, roughly balanced** data (~100+ per class):
different orientations, lighting, obstacle types, and floor textures.

> If keypresses don't register (some setups don't forward an interactive terminal to a
> launched node — the collector logs a warning when this happens), bring up the camera only and
> run the collector directly in a terminal instead:
> ```bash
> ros2 launch collision_avoider data_collection.launch.py enable_collector:=false  # terminal 1
> ros2 run collision_avoider data_collection                                        # terminal 2
> ```

When done, zip it up to move to a training machine (optional — you can also train on the board):

```bash
zip -r -q dataset.zip dataset
```

## 2. Train

Transfer-learns a ResNet18 into the 2-class free/blocked classifier. Run it on a host GPU box or
on the Ventuno itself (it auto-selects CUDA when available, else CPU):

```bash
pip install -r tools/requirements-export.txt
python3 tools/train_collision_resnet18.py --dataset dataset \
    --output models/collision_resnet18.pth --epochs 30
```

This saves the best-by-test-accuracy weights to `models/collision_resnet18.pth`. Copy that file
back to the board's `models/` directory if you trained elsewhere.

## 3. Lower to ExecuTorch

Convert the trained `.pth` into ExecuTorch `.pte` files. For the **NPU (QNN/HTP)** model, run this
**on the Ventuno** with the board environment sourced (the installer set this up):

```bash
source ~/.ventuno_collision_avoidance_env
source ~/.venv/executorch/bin/activate

# NPU (Hexagon HTP), 8-bit quantized:
python3 tools/export_resnet18_qnn.py \
    --weights models/collision_resnet18.pth \
    --output models/collision_resnet18_qnn.pte \
    --calibration-dir dataset --soc-model QCS8300
```

> **Important:** QNN quantization calibration must see the same normalized inputs the runtime
> uses. Pass `--calibration-dir dataset` so it calibrates on your real frames — otherwise the
> quantized NPU model will load but predict garbage.

For the **CPU (XNNPACK)** fallback model (host or board):

```bash
python3 tools/export_resnet18_cpu.py \
    --weights models/collision_resnet18.pth \
    --output models/collision_resnet18_xnnpack.pte

# Optional ONNX export for inspection / onnxruntime:
python3 tools/export_resnet18_onnx.py --weights models/collision_resnet18.pth
```

See [`.claude/skills/ventuno-setup/references/executorch-qnn.md`](.claude/skills/ventuno-setup/references/executorch-qnn.md)
for the full QNN/HTP export and debugging details.

## 4. Run the demo

First sanity-check the classifier offline on the bundled sample images (no camera or robot):

```bash
ros2 launch collision_classifier dataset_classifier.launch.py backend:=cpu
ros2 topic echo /collision/classification    # in another shell
```

Then run the full collision-avoidance demo on the robot:

```bash
ros2 launch launch/collision_avoidance.launch.py                        # NPU backend
ros2 launch launch/collision_avoidance.launch.py backend:=cpu \
    model_path:=models/collision_resnet18_xnnpack.pte                   # CPU fallback
ros2 launch launch/collision_avoidance.launch.py publish_cmd_vel:=false # dry run, no motion
ros2 launch launch/collision_avoidance.launch.py blocked_threshold:=0.6 # more cautious
```

The robot drives forward while `free` and turns in place when `blocked`. Tune `base_speed`,
`turn_speed`, and `blocked_threshold` (launch args or `src/collision_avoider/config/avoider.yaml`)
for your robot and space. To watch the classifier's live view, subscribe to `/collision/image`.

Handy checks:

```bash
ros2 topic hz /collision/classification   # ~5 Hz on the NPU
ros2 launch collision_avoider movement_test.launch.py   # open-loop drive test (no model)
```

## Repo layout

```
src/oak_camera/            ROS 2 driver for the OAK-D Lite (DepthAI)
src/collision_classifier/  ResNet18 free/blocked classifier node (CPU/XNNPACK or NPU/QNN)
src/collision_avoider/     Reactive controller + keyboard data-collection + movement test
tools/                     Training and ExecuTorch export scripts (.pth -> .onnx / .pte)
models/                    Generated model artifacts (see models/README.md)
launch/                    Top-level collision_avoidance.launch.py
datasets/sample_images/    A few images for the offline classifier test
scripts/                   Board install, Create 3 USB gadget, DDS profile
```

## Docs

Board setup and debugging knowledge lives in the `ventuno-setup` agent skill (plain markdown,
readable by any coding agent or human):

- [`.claude/skills/ventuno-setup/SKILL.md`](.claude/skills/ventuno-setup/SKILL.md) — overview,
  troubleshooting index, and end-to-end verification
- [`.claude/skills/ventuno-setup/references/executorch-qnn.md`](.claude/skills/ventuno-setup/references/executorch-qnn.md) —
  full ExecuTorch/QNN setup, from board identification to a running NPU classifier
- [`.claude/skills/ventuno-setup/references/create3-connection.md`](.claude/skills/ventuno-setup/references/create3-connection.md) —
  wiring and bring-up for the Create 3 USB link
- [`.claude/skills/ventuno-setup/references/ros-networking.md`](.claude/skills/ventuno-setup/references/ros-networking.md) —
  DDS/network tuning notes

See also [`AGENTS.md`](AGENTS.md) for the Codex-compatible pointer to the same content.

## License

MIT — see [LICENSE](LICENSE).

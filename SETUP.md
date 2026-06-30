# Setup Guide — Ventuno Object Tracking

Target platform: **Arduino Ventuno Q** (Qualcomm Snapdragon SoC, Ubuntu 24.04 ARM64)

---

## Overview

Two machines are involved **only if you need NPU (QNN) inference**. For CPU inference (XNNPACK), everything runs on the Ventuno alone.

| Machine | Role | Required for |
|---------|------|-------------|
| **Host** — x86_64 Ubuntu 22.04 | Export `.pte` models with QNN backend | NPU path only |
| **Ventuno Q** — ARM64 Ubuntu 24.04 | C++ inference runtime, ROS 2 nodes, camera driver | Always |

The QNN AOT compiler (model export) is distributed for x86_64 Linux only. Everything else — the C++ runtime, ROS 2 workspace, and XNNPACK model export — runs natively on the Ventuno.

---

## Table of Contents

**CPU path (Ventuno only)**
1. [System prerequisites (Ventuno)](#1-system-prerequisites-ventuno)
2. [ROS 2 Jazzy](#2-ros-2-jazzy)
3. [OAK-D Lite / DepthAI SDK](#3-oak-d-lite--depthai-sdk)
4. [Build ExecuTorch C++ runtime (XNNPACK)](#4-build-executorch-c-runtime-xnnpack)
5. [Export XNNPACK model (on Ventuno)](#5-export-xnnpack-model-on-ventuno)
6. [Build this workspace](#6-build-this-workspace)
7. [Running the nodes](#7-running-the-nodes)

**NPU path (adds x86 host steps)**
8. [System prerequisites (host)](#8-system-prerequisites-host)
9. [Clone ExecuTorch (host)](#9-clone-executorch-host)
10. [Python virtualenv (host)](#10-python-virtualenv-host)
11. [Build QNN AOT tools (host)](#11-build-qnn-aot-tools-host)
12. [Export QNN model (host)](#12-export-qnn-model-host)
13. [Copy QNN runtime libs to Ventuno](#13-copy-qnn-runtime-libs-to-ventuno)

---

## VENTUNO Q

---

## 1. System prerequisites (Ventuno)

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  clang \
  libclang-dev \
  git \
  wget \
  curl \
  python3-pip \
  python3-venv \
  libssl-dev \
  libffi-dev \
  pkg-config \
  patchelf \
  flatbuffers-compiler \
  libflatbuffers-dev \
  libopencv-dev
```

Set clang as the default compiler (ExecuTorch prefers it on ARM):

```bash
sudo update-alternatives --install /usr/bin/cc  cc  /usr/bin/clang   100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
```

---

## 2. ROS 2 Jazzy

```bash
# Locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS 2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update && sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-vision-msgs \
  ros-jazzy-image-transport \
  ros-jazzy-cv-bridge \
  ros-jazzy-camera-info-manager \
  python3-colcon-common-extensions \
  python3-rosdep

sudo rosdep init && rosdep update
```

> Do **not** add `source /opt/ros/jazzy/setup.bash` to `~/.bashrc`. Source it
> explicitly only in terminals where you need ROS, to avoid leaking ROS environment
> variables into ExecuTorch cmake builds.

---

## 3. OAK-D Lite / DepthAI SDK

```bash
sudo curl -fL https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/public.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/depthai.gpg

echo "deb [signed-by=/usr/share/keyrings/depthai.gpg] \
  https://artifacts.luxonis.com/artifactory/luxonis-depthai-ubuntu noble main" \
  | sudo tee /etc/apt/sources.list.d/depthai.list

sudo apt update && sudo apt install -y libdepthai-dev
```

---

## 4. Build ExecuTorch C++ runtime (XNNPACK)

Build natively on the Ventuno — no cross-compilation toolchain needed.

**In a clean terminal without ROS sourced:**

```bash
git clone https://github.com/pytorch/executorch.git ~/executorch
cd ~/executorch
git submodule update --init --recursive

python3 -m venv ~/.venv/executorch
source ~/.venv/executorch/bin/activate
pip install --upgrade pip
./install_requirements.sh
```

> `install_requirements.sh` is required before cmake — it generates the flatbuffers-derived
> C++ headers that the build system needs.

```bash
cmake \
  -S ~/executorch \
  -B ~/executorch/cmake-out \
  --preset linux \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/executorch

cmake --build ~/executorch/cmake-out -j$(nproc)
sudo cmake --install ~/executorch/cmake-out --prefix /opt/executorch
```

Verify:

```bash
ls /opt/executorch/lib/libexecutorch*.a
# Expected: libexecutorch.a  libexecutorch_core.a
#           libxnnpack_backend.a  libextension_module_static.a
```

---

## 5. Export XNNPACK model (on Ventuno)

With the venv active:

```bash
source ~/.venv/executorch/bin/activate
export PYTHONPATH=~/executorch:$PYTHONPATH

cd ~/ventuno-object-tracking
python tools/export_yolox_cpu.py
# Output: models/yolox_tiny_xnnpack.pte
```

---

## 6. Build this workspace

Open a new terminal and source ROS 2:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/ventuno-object-tracking
rosdep install --from-paths src --ignore-src -r -y

colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

---

## 7. Running the nodes

```bash
source /opt/ros/jazzy/setup.bash
source ~/ventuno-object-tracking/install/setup.bash
```

### Camera + detector together (recommended)

```bash
ros2 launch launch/object_tracking.launch.py \
  backend:=cpu \
  model_path:=$(pwd)/models/yolox_tiny_xnnpack.pte
```

### Camera only

```bash
ros2 launch src/oak_camera/launch/camera.launch.py
```

### Detector only (camera already running)

```bash
ros2 launch src/yolox_detector/launch/detector.launch.py \
  backend:=cpu \
  model_path:=$(pwd)/models/yolox_tiny_xnnpack.pte
```

### Monitoring

```bash
ros2 topic hz /detections
ros2 topic echo /detections
rviz2
```

---

## NPU PATH (HOST MACHINE)

The steps below are only needed if you want QNN/NPU inference. The host must be
**x86_64 Ubuntu 22.04** — this is where model export to QNN format happens.

---

## 8. System prerequisites (host)

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  clang \
  git \
  wget \
  curl \
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  python3-pip \
  libssl-dev \
  pkg-config \
  flatbuffers-compiler \
  libflatbuffers-dev
```

---

## 9. Clone ExecuTorch (host)

```bash
git clone https://github.com/pytorch/executorch.git ~/executorch
cd ~/executorch
git submodule update --init --recursive
```

---

## 10. Python virtualenv (host)

```bash
python3.11 -m venv ~/.venv/executorch
source ~/.venv/executorch/bin/activate
pip install --upgrade pip
./install_requirements.sh
pip install -r backends/qualcomm/requirements.txt
```

---

## 11. Build QNN AOT tools (host)

The QNN SDK and Android NDK are **auto-downloaded** if `QNN_SDK_ROOT` is not set.
Use `--skip_linux_android` to skip the Android build — only the x86_64 AOT tools are needed.

```bash
cd ~/executorch
source ~/.venv/executorch/bin/activate

./backends/qualcomm/scripts/build.sh --release --skip_linux_android
# Output: build-x86/ with x86_64 AOT tooling
```

If you have the SDK already installed at `/opt/qairt`, set `QNN_SDK_ROOT` to skip the
download:

```bash
export QNN_SDK_ROOT=/opt/qairt
./backends/qualcomm/scripts/build.sh --release --skip_linux_android
```

---

## 12. Export QNN model (host)

```bash
source ~/.venv/executorch/bin/activate
export PYTHONPATH=~/executorch:$PYTHONPATH

cd ~/ventuno-object-tracking
python tools/export_yolox_qnn.py
# Output: models/yolox_tiny_qnn.pte
```

> Replace the random calibration data in `export_yolox_qnn.py` with real images from
> your deployment environment for better quantization accuracy.

Copy the model to the Ventuno:

```bash
scp models/yolox_tiny_qnn.pte ventuno@<robot-ip>:~/ventuno-object-tracking/models/
```

---

## 13. Copy QNN runtime libs to Ventuno

The Ventuno needs the QNN `.so` runtime libraries at inference time. Copy the aarch64
binaries from the SDK on the host:

```bash
# Find the correct aarch64 lib directory for your SDK version:
ls $QNN_SDK_ROOT/lib/ | grep aarch64

# Copy runtime libs (adjust directory name to match your SDK version):
scp $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc<version>/libQnnHtp.so \
  ventuno@<robot-ip>:/usr/local/lib/
scp $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc<version>/libQnnSystem.so \
  ventuno@<robot-ip>:/usr/local/lib/

# Run on Ventuno after copying:
ssh ventuno@<robot-ip> sudo ldconfig
```

Then rebuild the workspace on the Ventuno with QNN enabled:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ventuno-object-tracking

colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DBUILD_QNN_BACKEND=ON \
    -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

Launch with NPU backend:

```bash
ros2 launch launch/object_tracking.launch.py \
  backend:=npu \
  model_path:=$(pwd)/models/yolox_tiny_qnn.pte
```

---

## Troubleshooting

**`torch` not found during cmake configure**
The venv was not active when cmake ran. Activate it and re-run cmake from a clean
terminal without ROS sourced:
```bash
source ~/.venv/executorch/bin/activate
unset CMAKE_PREFIX_PATH PYTHONPATH
cmake -S ~/executorch -B ~/executorch/cmake-out ...
```

**`use of executorch build extension module requires executorch build extension data map`**
Stale install. Clean and retry:
```bash
rm -rf ~/executorch/cmake-out
pip uninstall executorch -y
./install_requirements.sh
cmake ...
```

**`find_package(executorch REQUIRED)` fails during colcon build**
Pass `-DEXECUTORCH_INSTALL_DIR=/opt/executorch` explicitly, or add `/opt/executorch`
to `CMAKE_PREFIX_PATH`.

**`Failed to load QNN lib: libQnnHtp.so: cannot open shared object file`**
The QNN runtime `.so` files are missing or not in `LD_LIBRARY_PATH`. Re-run step 13
and verify `sudo ldconfig` was run on the Ventuno.

**ROS environment leaking into ExecuTorch builds**
Never source `/opt/ros/jazzy/setup.bash` in the terminal used to build ExecuTorch.
Open a fresh terminal.

**Node crashes with `Failed to load model`**
Check that the `.pte` path is correct and matches the backend (XNNPACK model on `npu`
backend will fail, and vice-versa).

**Low detection rate / missed objects**
- Lower `score_threshold` in `config/detector.yaml` (default 0.45)
- Use real calibration images in `export_yolox_qnn.py` before re-exporting

---

## Expected throughput (rough targets — YOLOX-tiny at 416×416)

| Backend | Typical latency |
|---------|----------------|
| CPU (XNNPACK) | ~80–150 ms/frame |
| NPU (QNN HTP) | ~15–40 ms/frame |

Actual numbers depend on the Snapdragon variant on your Ventuno Q.

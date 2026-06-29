# YOLOX Detector — Setup Guide

Target platform: **Arduino Ventuno Q** (Qualcomm Snapdragon SoC, Ubuntu 24.04 ARM64)

---

## Table of Contents

1. [System prerequisites](#1-system-prerequisites)
2. [ROS 2 Jazzy](#2-ros-2-jazzy)
3. [OpenCV](#3-opencv)
4. [ExecuTorch — CPU / XNNPACK backend](#4-executorch--cpu--xnnpack-backend)
5. [Qualcomm AI Engine Direct SDK (QAIRT) — NPU backend](#5-qualcomm-ai-engine-direct-sdk-qairt--npu-backend)
6. [ExecuTorch — rebuild with QNN backend](#6-executorch--rebuild-with-qnn-backend)
7. [OAK-D Lite / DepthAI dependency (oak_camera)](#7-oak-d-lite--depthai-dependency-oak_camera)
8. [Build this workspace](#8-build-this-workspace)
9. [Model export (dev machine)](#9-model-export-dev-machine)
10. [Running the nodes](#10-running-the-nodes)
11. [Verifying the NPU path](#11-verifying-the-npu-path)

---

## 1. System prerequisites

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
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  python3-pip \
  libssl-dev \
  libffi-dev \
  pkg-config \
  patchelf \
  flatbuffers-compiler \
  libflatbuffers-dev
```

Set clang as the default compiler (ExecuTorch prefers it on ARM):

```bash
sudo update-alternatives --install /usr/bin/cc  cc  /usr/bin/clang   100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
```

---

## 2. ROS 2 Jazzy

Follow the official installation guide.  The steps below are a condensed version for Ubuntu 24.04.

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

Add to `~/.bashrc`:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 3. OpenCV

ROS 2 Jazzy ships with OpenCV 4.  If it is not already present:

```bash
sudo apt install -y libopencv-dev
```

Verify:

```bash
pkg-config --modversion opencv4
# expected: 4.x.x
```

---

## 4. ExecuTorch — CPU / XNNPACK backend

Build ExecuTorch from source and install it to `/opt/executorch`.

```bash
# Clone — check https://github.com/pytorch/executorch/releases for the latest tag
git clone --branch v0.4.0 https://github.com/pytorch/executorch.git ~/executorch
cd ~/executorch
git submodule update --init --recursive
```

### Python virtualenv and install_requirements.sh

> **This step is mandatory.** `install_requirements.sh` generates the flatbuffers-derived
> C++ headers that the cmake build needs. Skipping it causes configure-time failures
> with missing includes.

```bash
python3.11 -m venv ~/.venv/executorch
source ~/.venv/executorch/bin/activate
pip install --upgrade pip

# Install ExecuTorch Python tools and their dependencies.
# This replaces a bare "pip install torch": it also runs codegen for flatbuffers.
cd ~/executorch
./install_requirements.sh

# XNNPACK python bindings (needed for the export scripts in step 9)
./install_requirements.sh --pybind xnnpack
```

### Build the C++ runtime

Use explicit `-S`/`-B` flags so there is no ambiguity about which directory contains
`CMakeLists.txt` (the repo root, not the build directory).

```bash
cmake \
  -S ~/executorch \
  -B ~/executorch/cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/executorch \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -GNinja

cmake --build ~/executorch/cmake-out -j$(nproc)
sudo cmake --install ~/executorch/cmake-out --prefix /opt/executorch
```

Check that the key libraries are present:

```bash
ls /opt/executorch/lib/libexecutorch*.a
# Should list: libexecutorch.a  libexecutorch_core.a
#              libxnnpack_backend.a  libextension_module_static.a
```

---

## 5. Qualcomm AI Engine Direct SDK (QAIRT) — NPU backend

> **Skip this section if you only need the CPU path.**

The QAIRT SDK ships the QNN runtime libraries (`libQnnHtp.so`, `libQnnSystem.so`, etc.) that ExecuTorch links against.

1. Create a free account at [developer.qualcomm.com](https://developer.qualcomm.com).
2. Download the **Qualcomm AI Engine Direct SDK** (QAIRT) for Linux ARM64 from:
   `Software → Qualcomm AI Engine Direct SDK`
3. Extract to `/opt/qairt`:

```bash
sudo mkdir -p /opt/qairt
sudo tar -xf qairt-sdk-*.tar.gz -C /opt/qairt --strip-components=1
```

4. Export the path so subsequent builds and the node runtime can find the libraries:

```bash
echo 'export QNN_SDK_ROOT=/opt/qairt' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify the HTP library is present:

```bash
ls $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so
```

---

## 6. ExecuTorch — rebuild with QNN backend

> **Skip this section if you only need the CPU path.**

Re-run cmake against the same source tree, adding the QNN flags.  The existing
`cmake-out/` directory is reused — cmake incremental builds will only recompile
what changed.

```bash
cmake \
  -S ~/executorch \
  -B ~/executorch/cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/executorch \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
  -GNinja

cmake --build ~/executorch/cmake-out -j$(nproc)
sudo cmake --install ~/executorch/cmake-out --prefix /opt/executorch
```

Confirm the QNN backend library was installed:

```bash
ls /opt/executorch/lib/libqnn_executorch_backend.a
```

---

## 7. OAK-D Lite / DepthAI dependency (oak_camera)

The `oak_camera` package depends on the DepthAI v3 C++ SDK.

```bash
# Install the official DepthAI apt repository
sudo curl -fL https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/public.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/depthai.gpg

echo "deb [signed-by=/usr/share/keyrings/depthai.gpg] \
  https://artifacts.luxonis.com/artifactory/luxonis-depthai-ubuntu noble main" \
  | sudo tee /etc/apt/sources.list.d/depthai.list

sudo apt update && sudo apt install -y libdepthai-dev
```

---

## 8. Build this workspace

```bash
cd ~/ventuno-object-tracking   # or wherever you cloned this repo

# Install any remaining ROS 2 deps declared in package.xml files
rosdep install --from-paths src --ignore-src -r -y

# CPU-only build (no QNN)
colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DCMAKE_BUILD_TYPE=Release

# ── OR ── NPU build (requires steps 5–6 above)
colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DBUILD_QNN_BACKEND=ON \
    -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

---

## 9. Model export (dev machine)

Run the export scripts **on a development machine**, not on the robot.  Copy the resulting `.pte` files to the robot.

### Prerequisites (dev machine)

If you ran the build steps on this machine you already have the right virtualenv.
Activate it and install the remaining Python deps:

```bash
source ~/.venv/executorch/bin/activate   # created in step 4

# YOLOX Python package (provides the model definition)
pip install yolox

# QNN Python bindings (only needed for export_yolox_qnn.py)
cd ~/executorch
./install_requirements.sh --pybind qnn
```

> The `install_requirements.sh` script pins the exact PyTorch version that
> matches your ExecuTorch checkout, so do **not** install torch separately with
> a version override — it will conflict.

### CPU / XNNPACK export

```bash
cd ~/ventuno-object-tracking
python tools/export_yolox_cpu.py
# Output: models/yolox_tiny_xnnpack.pte
```

### NPU / QNN export

```bash
export QNN_SDK_ROOT=/opt/qairt  # must also be set on dev machine

python tools/export_yolox_qnn.py
# Output: models/yolox_tiny_qnn.pte
```

> **Tip:** replace the random calibration data in `export_yolox_qnn.py` with real
> images from your deployment environment for better quantization accuracy.

### Copy models to the robot

```bash
scp models/yolox_tiny_xnnpack.pte ventuno@<robot-ip>:~/ventuno-object-tracking/models/
scp models/yolox_tiny_qnn.pte     ventuno@<robot-ip>:~/ventuno-object-tracking/models/
```

---

## 10. Running the nodes

Source the workspace first:

```bash
source ~/ventuno-object-tracking/install/setup.bash
```

### Camera + detector together (recommended)

```bash
# CPU backend
ros2 launch launch/object_tracking.launch.py \
  backend:=cpu \
  model_path:=$(pwd)/models/yolox_tiny_xnnpack.pte

# NPU backend
ros2 launch launch/object_tracking.launch.py \
  backend:=npu \
  model_path:=$(pwd)/models/yolox_tiny_qnn.pte
```

### Detector only (camera already running separately)

```bash
ros2 launch yolox_detector detector.launch.py \
  backend:=cpu \
  model_path:=$(pwd)/models/yolox_tiny_xnnpack.pte
```

### Useful monitoring commands

```bash
# Detection rate
ros2 topic hz /detections

# View detections
ros2 topic echo /detections

# Visualise bounding boxes in RViz2 — add Image display on /detections/image
rviz2
```

---

## 11. Verifying the NPU path

After launching with `backend:=npu`, confirm that inference is actually running on the HTP:

```bash
# Check the node started without errors
ros2 node info /yolox_detector

# Confirm QNN libs are loaded in the node process
PID=$(ros2 node list --full-node-enclave-names | grep yolox | xargs -I{} ros2 run --prefix 'pgrep -n' yolox_detector yolox_detector_node 2>/dev/null || pgrep -n yolox_detector_node)
cat /proc/$PID/maps | grep -E "libQnnHtp|libQnnSystem"
# Should show the QNN .so files mapped into the process

# Latency comparison
ros2 topic hz /detections   # should be noticeably higher than CPU baseline
```

Expected throughput on YOLOX-tiny at 416×416 (rough targets — actual numbers depend on Snapdragon variant):

| Backend | Typical latency |
|---------|----------------|
| CPU (XNNPACK) | ~80–150 ms/frame |
| NPU (QNN HTP) | ~15–40 ms/frame |

---

## Troubleshooting

**`CMake Error: The source directory does not appear to contain CMakeLists.txt`**
You ran `cmake ..` from inside a subdirectory instead of pointing at the repo root.
Use the explicit `-S`/`-B` form shown in sections 4 and 6:
```bash
cmake -S ~/executorch -B ~/executorch/cmake-out ...
```

**`find_package(executorch REQUIRED)` fails during colcon build**
Pass `-DEXECUTORCH_INSTALL_DIR=/opt/executorch` explicitly, or add `/opt/executorch` to `CMAKE_PREFIX_PATH`.

**`Failed to load QNN lib: libQnnHtp.so: cannot open shared object file`**
Ensure `LD_LIBRARY_PATH` includes `$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4` and that you sourced `~/.bashrc` in the terminal running the node.

**Node crashes immediately with `Failed to load model`**
Check that the `.pte` file path is correct and that the model was exported with the right backend (CPU model on `npu` backend or vice-versa will fail at partition dispatch).

**Low detection rate / missed objects**
- Lower `score_threshold` in `config/detector.yaml` (default 0.45)
- Replace random calibration data in `export_yolox_qnn.py` with real images before re-exporting the QNN model

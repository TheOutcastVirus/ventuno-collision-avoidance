# Setup Guide — Ventuno Object Tracking

Target platform: **Arduino Ventuno Q** (Qualcomm Snapdragon SoC, Ubuntu 24.04 ARM64)

---

## Overview

Two machines are involved in this setup:

| Machine | Role |
|---------|------|
| **Host** — x86_64 Ubuntu 22.04 Linux | Python AOT environment: export `.pte` model files, build the QNN x86_64 AOT tooling |
| **Ventuno Q** — ARM64 Ubuntu 24.04 | C++ inference runtime, ROS 2 nodes, camera driver |

Model export (Python, torch.export) **must** run on the host because the QNN AOT compiler is only distributed for x86_64 Linux. The resulting `.pte` files are then copied to the Ventuno for inference.

The XNNPACK C++ runtime and the ROS 2 workspace are built **natively on the Ventuno** (it is a full Ubuntu system, not an MCU).

---

## Table of Contents

**Host machine**
1. [System prerequisites (host)](#1-system-prerequisites-host)
2. [Clone ExecuTorch and init submodules](#2-clone-executorch-and-init-submodules)
3. [Python virtualenv and install_requirements.sh](#3-python-virtualenv-and-install_requirementssh)
4. [Qualcomm AI Engine Direct SDK (QAIRT)](#4-qualcomm-ai-engine-direct-sdk-qairt)
5. [Build ExecuTorch AOT tools + QNN backend (host)](#5-build-executorch-aot-tools--qnn-backend-host)
6. [Export models to .pte](#6-export-models-to-pte)

**Ventuno Q**
7. [System prerequisites (Ventuno)](#7-system-prerequisites-ventuno)
8. [ROS 2 Jazzy](#8-ros-2-jazzy)
9. [OAK-D Lite / DepthAI SDK](#9-oak-d-lite--depthai-sdk)
10. [Build ExecuTorch C++ runtime (XNNPACK)](#10-build-executorch-c-runtime-xnnpack)
11. [Copy model files to Ventuno](#11-copy-model-files-to-ventuno)
12. [Build this workspace](#12-build-this-workspace)
13. [Running the nodes](#13-running-the-nodes)

---

## HOST MACHINE

---

## 1. System prerequisites (host)

The host must be **x86_64 Ubuntu 22.04**. Ubuntu 24.04 is not officially supported by the QNN SDK for the AOT host tooling.

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

## 2. Clone ExecuTorch and init submodules

```bash
git clone https://github.com/pytorch/executorch.git ~/executorch
cd ~/executorch
git submodule update --init --recursive
```

---

## 3. Python virtualenv and install_requirements.sh

> **This step is mandatory before any cmake build.** `install_requirements.sh` installs
> PyTorch nightly and generates the flatbuffers-derived C++ headers the cmake build needs.

```bash
python3.11 -m venv ~/.venv/executorch
source ~/.venv/executorch/bin/activate
pip install --upgrade pip

cd ~/executorch
./install_requirements.sh
```

Valid flags for `install_requirements.sh`:
- `--use-pt-pinned-commit` — build against the exact pinned PyTorch commit instead of nightly
- `--example` — also install torchvision, torchaudio, and example script dependencies

For QNN Python tools, also install:

```bash
pip install -r backends/qualcomm/requirements.txt
```

> Keep the venv active for all subsequent host steps.

---

## 4. Qualcomm AI Engine Direct SDK (QAIRT)

> **Skip this section if you only need the CPU/XNNPACK path.**

1. Create a free account at [developer.qualcomm.com](https://developer.qualcomm.com).
2. Download the **Qualcomm AI Engine Direct SDK** (QAIRT) for Linux from the Software Center.
   Verified stable version: **QNN 2.37.0**
3. Extract to a permanent location:

```bash
sudo mkdir -p /opt/qairt
sudo unzip qairt-sdk-*.zip -d /opt/qairt --strip-components=1
# or for tar archives:
# sudo tar -xf qairt-sdk-*.tar.gz -C /opt/qairt --strip-components=1
```

4. Source the SDK environment setup script. This sets `LD_LIBRARY_PATH`, `PATH`, and other
   required variables:

```bash
source /opt/qairt/bin/envsetup.sh
```

Add to `~/.bashrc` so it persists across terminals:

```bash
echo 'export QNN_SDK_ROOT=/opt/qairt' >> ~/.bashrc
echo 'source $QNN_SDK_ROOT/bin/envsetup.sh' >> ~/.bashrc
```

Verify the HTP library is present:

```bash
ls $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so
```

---

## 5. Build ExecuTorch AOT tools + QNN backend (host)

Use the official build script from the ExecuTorch repo. It builds both the x86_64 AOT
tools (needed to compile models to `.pte`) and can optionally cross-compile the ARM64
Linux runtime.

```bash
cd ~/executorch
source ~/.venv/executorch/bin/activate
source $QNN_SDK_ROOT/bin/envsetup.sh

# Build x86_64 AOT tools + QNN backend (mandatory for model export)
./backends/qualcomm/scripts/build.sh --release
```

This produces `build-x86/` with the AOT tooling.

To also cross-compile the ARM64 Linux runtime for the Ventuno (requires an OE Linux
toolchain — skip if building natively on the Ventuno in step 10):

```bash
export TOOLCHAIN_ROOT_HOST=/path/to/sysroots/x86_64-qtisdk-linux
export TOOLCHAIN_ROOT_TARGET=/path/to/sysroots/armv8a-oe-linux

./backends/qualcomm/scripts/build.sh --enable_linux_embedded --release
# Output: build-oe-linux/
```

After the build, copy the QNN Python bindings so the export scripts can import them:

```bash
# Already done by build.sh — verify:
ls ~/executorch/backends/qualcomm/python/
```

---

## 6. Export models to .pte

With the venv active and QNN env sourced:

```bash
source ~/.venv/executorch/bin/activate
source $QNN_SDK_ROOT/bin/envsetup.sh
export PYTHONPATH=~/executorch/..:$PYTHONPATH

cd ~/ventuno-object-tracking
```

### CPU / XNNPACK export

```bash
python tools/export_yolox_cpu.py
# Output: models/yolox_tiny_xnnpack.pte
```

### NPU / QNN export

```bash
python tools/export_yolox_qnn.py
# Output: models/yolox_tiny_qnn.pte
```

> **Tip:** replace the random calibration data in `export_yolox_qnn.py` with real
> images from your deployment environment for better quantization accuracy.

---

## VENTUNO Q

---

## 7. System prerequisites (Ventuno)

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

## 8. ROS 2 Jazzy

Follow the official installation guide. The steps below are a condensed version for Ubuntu 24.04.

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
> variables into non-ROS builds (e.g. ExecuTorch cmake builds).

---

## 9. OAK-D Lite / DepthAI SDK

The `oak_camera` package requires the DepthAI v3 C++ SDK.

```bash
sudo curl -fL https://artifacts.luxonis.com/artifactory/luxonis-depthai-data-local/public.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/depthai.gpg

echo "deb [signed-by=/usr/share/keyrings/depthai.gpg] \
  https://artifacts.luxonis.com/artifactory/luxonis-depthai-ubuntu noble main" \
  | sudo tee /etc/apt/sources.list.d/depthai.list

sudo apt update && sudo apt install -y libdepthai-dev
```

---

## 10. Build ExecuTorch C++ runtime (XNNPACK)

Build ExecuTorch natively on the Ventuno. The Ventuno runs full Ubuntu 24.04 so no
cross-compilation toolchain is needed for the XNNPACK (CPU) backend.

First, clone ExecuTorch and set up the Python environment:

```bash
git clone https://github.com/pytorch/executorch.git ~/executorch
cd ~/executorch
git submodule update --init --recursive

python3 -m venv ~/.venv/executorch
source ~/.venv/executorch/bin/activate
pip install --upgrade pip
./install_requirements.sh
```

Then build the C++ runtime. Run this in a **clean terminal without ROS sourced**:

```bash
source ~/.venv/executorch/bin/activate

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

Verify key libraries are present:

```bash
ls /opt/executorch/lib/libexecutorch*.a
# Should list: libexecutorch.a  libexecutorch_core.a
#              libxnnpack_backend.a  libextension_module_static.a
```

### QNN runtime on Ventuno (optional)

If you cross-compiled with `--enable_linux_embedded` on the host in step 5, copy the
output to the Ventuno:

```bash
# On host:
scp -r ~/executorch/build-oe-linux/lib/ ventuno@<robot-ip>:/opt/executorch/
scp -r ~/executorch/build-oe-linux/include/ ventuno@<robot-ip>:/opt/executorch/
```

Also copy the QNN SDK runtime libraries (the `.so` files for the target) to the Ventuno:

```bash
# On host — the exact lib subdirectory depends on your QAIRT version and OE toolchain:
scp $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc<version>/libQnnHtp.so ventuno@<robot-ip>:/usr/local/lib/
scp $QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc<version>/libQnnSystem.so ventuno@<robot-ip>:/usr/local/lib/
sudo ldconfig  # run on Ventuno after copying
```

> Check `ls $QNN_SDK_ROOT/lib/` on the host to find the correct aarch64 subdirectory
> name for your SDK version.

---

## 11. Copy model files to Ventuno

From the host machine:

```bash
scp models/yolox_tiny_xnnpack.pte ventuno@<robot-ip>:~/ventuno-object-tracking/models/
scp models/yolox_tiny_qnn.pte     ventuno@<robot-ip>:~/ventuno-object-tracking/models/
```

---

## 12. Build this workspace

On the Ventuno, source ROS 2 first:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/ventuno-object-tracking

# Install any remaining ROS 2 deps declared in package.xml files
rosdep install --from-paths src --ignore-src -r -y

# CPU-only build (XNNPACK)
colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DCMAKE_BUILD_TYPE=Release

# ── OR ── NPU build (requires QNN runtime libraries installed in step 10)
colcon build \
  --cmake-args \
    -DEXECUTORCH_INSTALL_DIR=/opt/executorch \
    -DBUILD_QNN_BACKEND=ON \
    -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

---

## 13. Running the nodes

Source both ROS 2 and the workspace:

```bash
source /opt/ros/jazzy/setup.bash
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

## Troubleshooting

**`torch` not found during cmake configure**
The venv was not active when cmake ran, or `install_requirements.sh` was not run.
Activate the venv and re-run cmake from a clean terminal without ROS sourced:
```bash
source ~/.venv/executorch/bin/activate
unset CMAKE_PREFIX_PATH PYTHONPATH
cmake -S ~/executorch -B ~/executorch/cmake-out ...
```

**`use of executorch build extension module requires executorch build extension data map`**
A partial or stale executorch install is being imported during the build. Clean and retry:
```bash
rm -rf ~/executorch/cmake-out ~/executorch/pip-out
pip uninstall executorch -y
./install_requirements.sh
cmake ...
```

**`CMake Error: The source directory does not appear to contain CMakeLists.txt`**
Use the explicit `-S`/`-B` form — never run `cmake ..` from inside the build directory:
```bash
cmake -S ~/executorch -B ~/executorch/cmake-out ...
```

**`find_package(executorch REQUIRED)` fails during colcon build**
Pass `-DEXECUTORCH_INSTALL_DIR=/opt/executorch` explicitly, or add `/opt/executorch`
to `CMAKE_PREFIX_PATH`.

**`Failed to load QNN lib: libQnnHtp.so: cannot open shared object file`**
The QNN runtime `.so` files are not in `LD_LIBRARY_PATH`. Add the directory containing
them and run `sudo ldconfig`.

**ROS environment leaking into ExecuTorch builds**
Never source `/opt/ros/jazzy/setup.bash` in the same terminal you use to build
ExecuTorch. Open a fresh terminal and do not add the ROS source line to `~/.bashrc`.

**Node crashes immediately with `Failed to load model`**
Check that the `.pte` file path is correct and that the model was exported with the
matching backend (an XNNPACK model will fail on the `npu` backend and vice-versa).

**Low detection rate / missed objects**
- Lower `score_threshold` in `config/detector.yaml` (default 0.45)
- Use real calibration images in `export_yolox_qnn.py` instead of random data before
  re-exporting the QNN model

---

## Expected throughput (rough targets — YOLOX-tiny at 416×416)

| Backend | Typical latency |
|---------|----------------|
| CPU (XNNPACK) | ~80–150 ms/frame |
| NPU (QNN HTP) | ~15–40 ms/frame |

Actual numbers depend on the Snapdragon variant on your Ventuno Q.

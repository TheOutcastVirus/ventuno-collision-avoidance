# ExecuTorch + QNN HTP setup and debugging

Full working reference target: Ventuno Q, Ubuntu 24.04 Qualcomm image, SoC `QCS8275`
(soc_id `675`), QAIRT `/opt/qcom/aistack/qairt/2.48.0.260626`, HTP skel libraries
`hexagon-v75/unsigned`, ROS Jazzy. Verified path: a ResNet18 free/blocked classifier exported
to an ExecuTorch `.pte` with QNN HTP delegation, publishing `/collision/classification` at
~5 Hz on sample images, with `/collision/image` for annotated frames. (The same QNN path was
originally brought up with YOLOX-Tiny; only the model and pre/post-processing differ.)

The working path end to end:

1. Export the ResNet18 classifier to an ExecuTorch `.pte` using the Qualcomm backend (`backend="htp"`).
2. Build the ROS package with `-DBUILD_QNN_BACKEND=ON` so `QnnBackend` is compiled in.
3. Launch the classifier with `backend:=npu` and the QNN-exported model.
4. The node selects `QnnBackend`, preloads QNN HTP libraries, loads the `.pte`, and runs
   `module_->execute("forward", ...)`.

Key code locations:
- `src/collision_classifier/launch/dataset_classifier.launch.py` — declares `backend` (default `cpu`).
- `src/collision_classifier/src/collision_classifier_node.cpp` — maps `backend == "npu"` to `QnnBackend`.
- `src/collision_classifier/src/qnn_backend.cpp` — preloads `libQnnSystem.so`, `libQnnHtp.so`,
  `libQnnHtpPrepare.so`, loads/executes the model.
- `tools/export_resnet18_qnn.py` — exports with `backend="htp"`, `QnnExecuTorchBackendType.kHtpBackend`.

## Board identification and HTP target

```bash
cat /sys/devices/soc0/machine   # QCS8275
cat /sys/devices/soc0/soc_id    # 675
```

Don't assume a newer HTP arch (e.g. V81) works — validate with:

```bash
$QNN_SDK_ROOT/bin/aarch64-oe-linux-gcc11.2/qnn-platform-validator \
  --backend dsp --coreVersion --testBackend --debug --targetPath /tmp/qnn_validator
```

This board validates as **HTP V75**. Working paths:
- DSP libs: `$QNN_SDK_ROOT/lib/hexagon-v75/unsigned`
- ARM QNN libs: `$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2`

## ExecuTorch chipset patch (why and what)

ExecuTorch has no chipset enum entry for this board. The fix used throughout this project
is a `QCS8300 = 82` compatibility entry mapped to `HtpArch.V75` with 8 MB VTCM, applied to
the ExecuTorch checkout at `/opt/executorch`. `scripts/install_ventuno_deps.sh`'s
`patch_executorch_for_ventuno()` applies this automatically and is idempotent — it checks
for `QCS8300` in `qc_schema.py` and `utils.py` before reapplying, so it's safe to rerun.

Files touched:
- `backends/qualcomm/serialization/qc_compiler_spec.fbs` — add `QCS8300 = 82` to the enum.
- `backends/qualcomm/serialization/qc_schema.py` — add `QCS8300 = 82` to `QcomChipset`, and
  `QcomChipset.QCS8300: SocInfo(QcomChipset.QCS8300, HtpInfo(HtpArch.V75, 8))` to `_soc_info_table`.
- `backends/qualcomm/utils/utils.py` — add `"QCS8300": HtpArch.V75` to
  `get_soc_to_htp_arch_map()` and `"QCS8300": QcomChipset.QCS8300` to `get_soc_to_chipset_map()`.

The exact patch is in `scripts/install_ventuno_deps.sh` (`patch_executorch_for_ventuno`).
If a newer ExecuTorch revision shifts the diff context so `git apply` fails there, make the
four edits above manually instead of fighting the patch.

Export models with `--soc-model QCS8300` (see `tools/export_resnet18_qnn.py` usage below).

## Building ExecuTorch

Done natively on the board (no cross-compilation) into `$EXECUTORCH_ROOT/build-x86` — the
directory name is just inherited from the Qualcomm build scripts; on the Ventuno it holds
native ARM artifacts. `scripts/install_ventuno_deps.sh` does this in `build_executorch()`.
To do it by hand:

```bash
cd "$EXECUTORCH_ROOT"
git submodule update --init --recursive
python3 -m venv "$HOME/.venv/executorch"
source "$HOME/.venv/executorch/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r backends/qualcomm/requirements.txt

cmake -S . -B build-x86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$EXECUTORCH_ROOT/build-x86" \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_USE_DL=ON \
  -DPYTHON_EXECUTABLE="$HOME/.venv/executorch/bin/python"

cmake --build build-x86 --target install -j"$(nproc)"

# Copy artifacts the export path needs back into the source tree:
mkdir -p backends/qualcomm/python exir/_serialize kernels/quantized
cp -fv build-x86/backends/qualcomm/Py* backends/qualcomm/python/
cp -fv schema/program.fbs exir/_serialize/program.fbs
cp -fv schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
[ -f build-x86/kernels/quantized/libquantized_ops_aot_lib.so ] && \
  cp -fv build-x86/kernels/quantized/libquantized_ops_aot_lib.so kernels/quantized/
```

Verify:

```bash
test -f "$EXECUTORCH_ROOT/build-x86/lib/cmake/ExecuTorch/executorch-config.cmake"
test -f "$EXECUTORCH_ROOT/build-x86/lib/cmake/ExecuTorch/ExecuTorchTargets.cmake"
test -f "$EXECUTORCH_ROOT/build-x86/lib/executorch/backends/qualcomm/libqnn_executorch_backend.so"
```

The ROS package links `executorch`, `executorch_core`, `extension_module_static`,
`extension_tensor`, `portable_ops_lib`, `qnn_executorch_backend`.

## Exporting a model

```bash
source "$HOME/.venv/executorch/bin/activate"
python tools/export_resnet18_qnn.py \
  --weights models/collision_resnet18.pth \
  --output models/collision_resnet18_qnn.pte \
  --soc-model QCS8300 \
  --calibration-dir dataset \
  --calibration-batches 16
deactivate
```

Success looks like `Starting stage: Graph Preparation Initializing` ... `Saved: models/collision_resnet18_qnn.pte`.
Calibration must feed the same ImageNet-normalized inputs the C++ node uses at runtime
(`--calibration-dir` points at real collected frames); mismatched calibration yields a
model that loads but predicts garbage.

## Building the ROS package with QNN

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select collision_classifier --cmake-args \
  -Dexecutorch_DIR=$EXECUTORCH_ROOT/build-x86/lib/cmake/ExecuTorch \
  -DBUILD_QNN_BACKEND=ON
```

## Known issues and fixes

**FastRPC / DSP libraries missing** — symptom: QNN validation or runtime fails around
DSP/FastRPC loading (`libcdsprpc.so` missing, no `/dev/fastrpc-*` access).
Fix: `sudo apt-get install -y qcom-fastrpc1 qcom-fastrpc-dev`.

**`ADSP_LIBRARY_PATH` separator** — FastRPC needs **semicolons**, not the usual Linux colon:
```bash
export ADSP_LIBRARY_PATH="$QAIRT_LIB/hexagon-v75/unsigned;/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp/cdsp;/dsp"
```

**Python export fails on missing schema resources** — copy `schema/program.fbs` and
`schema/scalar_type.fbs` from the ExecuTorch checkout into `exir/_serialize/` (done by the
build steps above; repeat if export starts failing again after an ExecuTorch update).

**`cv_bridge` include on Jazzy** — use `#include <cv_bridge/cv_bridge.hpp>` (not the old
non-`.hpp` header).

**ExecuTorch tensor shape type** — use `executorch::aten::SizesType` for shape vectors in
backend C++ code.

**ExecuTorch `Module` API** — construct `Module(model_path, Module::LoadMode::MmapUseMlock)`
then call `module_->load()` with no argument.

**`find_package(executorch REQUIRED)` fails on optional tokenizer targets** — ExecuTorch's
CMake package config references optional tokenizer targets this project doesn't use.
`src/collision_classifier/CMakeLists.txt` predefines those as imported interface libraries before
`find_package(executorch REQUIRED)` to work around it.

**Garbage classification at runtime (node loads, `/collision/classification` publishes, but
scores are meaningless)** — the usual root cause is a preprocessing/calibration mismatch. The
classifier expects the torchvision pipeline: BGR→RGB, scale `1/255`, then ImageNet mean/std
normalization (`src/collision_classifier/src/preprocess.cpp` and `tools/collision_model.py`).
Make sure the QNN calibration during export feeds the **same** normalized inputs
(`export_resnet18_qnn.py --calibration-dir dataset`, not normalized noise), and that the
`mean`/`std` params match between the node and the training transform.

## Runtime verification

```bash
ros2 node list
ros2 topic list
ros2 topic hz /collision/classification
ros2 topic echo --once /collision/classification

# Confirm QNN/FastRPC libraries actually loaded into the running process:
PID=$(pgrep -f collision_classifier_node)
grep -E 'libQnn|libcdsprpc|libxdsprpc|hexagon|fastrpc' /proc/$PID/maps | sort -u
```

Saving annotated frames (`/collision/image` only publishes with a subscriber):

```bash
mkdir -p artifacts/annotated && cd artifacts/annotated
ros2 run image_view image_saver --ros-args -r image:=/collision/image
```

## Replicating on a different/new Ventuno Q

`scripts/install_ventuno_deps.sh` does all of this. If doing it manually or debugging why
the script diverges from a known-good board:

1. Confirm identity: `cat /sys/devices/soc0/machine && cat /sys/devices/soc0/soc_id`. If it's
   not `QCS8275`/`675`, re-validate the HTP arch with `qnn-platform-validator` rather than
   assuming V75 still applies.
2. `sudo apt-get install -y qcom-fastrpc1 qcom-fastrpc-dev`.
3. Confirm QAIRT/QNN SDK path and libraries exist; update `QNN_SDK_ROOT` in
   `~/.ventuno_collision_avoidance_env` if the version differs from `2.48.0.260626`.
4. Confirm `~/.ventuno_collision_avoidance_env` is created and sourced from `~/.bashrc` before
   any non-interactive-shell early return (SSH one-liners need it).
5. ExecuTorch checkout at `/opt/executorch`; apply the `QCS8300` patch above.
6. Build ExecuTorch, export the model, build the ROS package, launch — same commands as above.

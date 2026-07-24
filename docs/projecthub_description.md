# A TurtleBot 4 That Teaches Itself Not to Crash, Powered by an Arduino Ventuno Q

> **Subheading:** A JetBot-style collision-avoidance robot. You drive it around, label a
> few hundred camera frames "free" or "blocked," and a small neural network learns your
> space. It then wanders on its own without bumping into things — the classifier running on
> the Arduino Ventuno Q's Hexagon NPU, no cloud, no laptop, no depth sensor.

![The robot driving forward on a clear path and turning away when it sees an obstacle](images/collision-avoidance-demo.gif)

---

## Metadata for the Project Hub form

| Field | Value |
|---|---|
| Type | Showcase |
| License | MIT |
| Categories | Robotics · Machine Learning / AI · Embedded · Computer Vision |
| Difficulty | Intermediate–Advanced |

### Components and supplies

| Qty | Component |
|---|---|
| 1 | Arduino Ventuno Q |
| 1 | Clearpath TurtleBot 4 Lite, iRobot Create 3 base |
| 1 | Luxonis OAK-D Lite camera |
| 1 | USB-C cable for the Create 3 USB-ethernet link |

### Apps and platforms

Ubuntu 24.04 (Qualcomm image) · ROS 2 Jazzy · PyTorch ExecuTorch · Qualcomm QNN / QAIRT
SDK · DepthAI · PyTorch + torchvision · C++ / Python 3

---

## Intro

There are two ways to keep a robot from driving into a wall. The obvious one is to give it
a distance sensor — lidar, ultrasonics, a depth camera — and stop when something gets close.
The more interesting one, and the one NVIDIA's [JetBot](https://github.com/NVIDIA-AI-IOT/jetbot)
made famous, is to skip the sensor entirely and let the robot *learn what trouble looks
like* from a plain camera.

This project is that second approach, running on an **Arduino Ventuno Q**.

Like our [object-following demo](https://github.com/TheOutcastVirus/ventuno-object-tracking),
it starts with a TurtleBot 4 whose Raspberry Pi has been pulled out and replaced with a
Ventuno Q — an ARM board with a Qualcomm **Hexagon NPU** for neural network inference. But
where that project detected and chased objects, this one does something conceptually
simpler and, in a way, more fun: it looks at the single image in front of it and answers one
yes/no question — *is the path ahead free, or blocked?* — thirty times a second, and steers
on the answer.

The catch, and the charm, is that **you have to teach it.** There's no pre-trained model to
download. You drive the robot around your actual room, tag frames as free or blocked, and
train a classifier on your floor, your furniture, your lighting. An hour of data collection
and a short training run later, the robot knows your space.

## What it does

- **Sees with one camera.** An OAK-D Lite streams RGB into ROS 2. No depth, no lidar — just
  the picture.
- **Classifies free vs. blocked on the NPU.** A ResNet18, transfer-learned into a two-class
  classifier, runs through ExecuTorch on the Hexagon NPU via Qualcomm's QNN HTP backend at
  around 5 Hz.
- **Drives reactively.** A dead-simple controller drives the Create 3 base forward while the
  path reads *free*, and turns in place when it reads *blocked* — exactly the JetBot
  behavior.
- **Smooths its own nerves.** The blocked probability is EMA-filtered so one jumpy frame
  doesn't send the robot spinning.
- **Learns your environment.** A keyboard-driven data collector (`f` = free, `b` = blocked)
  makes building a dataset a few minutes of driving around.
- **Fails safe.** No fresh classification within half a second and the robot stops;
  `publish_cmd_vel:=false` runs the whole stack without moving a wheel.

## Why the Ventuno Q

**The NPU does the seeing.** A ResNet18 forward pass on every frame is real work for a small
CPU. Putting it on the Hexagon NPU keeps the classifier fast and leaves the ARM cores free
for ROS 2, the camera driver, and the control loop.

**It drops into the Pi's slot.** The TurtleBot 4's compute bay is sized and powered for a
single-board computer, so the Ventuno Q is a swap, not a rebuild.

**It's self-contained.** The model lives on the robot's disk and runs on the robot's
silicon. Train it once, unplug the network, and it still avoids your furniture.

## How it works

```
OAK-D Lite ──RGB──▶ oak_camera ──/oak/rgb/image_raw──▶ collision_classifier
                                                          (Hexagon NPU)
                                                              │
                                                   /collision/classification
                                                       (P(free), P(blocked))
                                                              ▼
                                                       collision_avoider
                                                              │
                                                          /cmd_vel
                                                              ▼
                                                        Create 3 base
```

<!-- IMAGE: replace the ASCII sketch with a proper block diagram before publishing -->

Three ROS 2 nodes, launched together by `launch/collision_avoidance.launch.py`. Notably,
there's **no republisher node** here — unlike some TurtleBot 4 compute swaps, the Create 3
sits at the root namespace and takes `/cmd_vel` directly, so nothing has to bridge the
`_do_not_use` topics. The DDS environment does still have to match the base
(`ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`).

### `oak_camera` — the eye

A DepthAI driver publishing the OAK-D Lite's RGB stream. That's all this project needs from
it — no stereo depth, no alignment. One camera, one job.

### `collision_classifier` — the judgment

A C++ node running a ResNet18 binary classifier through ExecuTorch, with two interchangeable
backends chosen by a `backend` parameter: `npu` (QNN HTP) or `cpu` (XNNPACK). Each frame is
resized to 224×224, converted BGR→RGB, scaled to [0,1], and normalized with the standard
ImageNet mean/std before inference — and that normalization has to match what training did
*exactly*, or the model sees inputs it was never trained on.

There's a subtle ordering trap the code is careful about. torchvision's `ImageFolder` sorts
class folders alphabetically, so `dataset/{blocked,free}` makes **index 0 = blocked, index 1
= free** — not the order you'd guess from the JetBot convention. The node keeps an explicit
`blocked_index` so the controller always reads the right probability regardless of how the
labels happen to sort. It publishes both class scores on `/collision/classification`, plus
an annotated `/collision/image` you can watch live.

### `collision_avoider` — the reflex

A Python node, and deliberately about as simple as a controller can be. It pulls P(blocked)
out of each classification, smooths it with an EMA so a single noisy frame can't jerk the
robot around, and then:

```python
if self.prob_blocked < self.blocked_threshold:
    twist.linear.x = clamp(self.base_speed, self.max_linear_speed)   # clear -> forward
else:
    twist.angular.z = clamp(                                          # blocked -> turn
        self.turn_direction * self.turn_speed, self.max_angular_speed)
```

Forward when free, turn in place when blocked, look again. It runs at 10 Hz, clamps every
command to safe indoor speed caps, and — the important part — stops dead if no classification
has arrived in the last half second. A perception stack that silently dies should leave you
with a stopped robot, not a runaway one.

## Teaching it: collect → train → lower → run

This is the part that makes the project. There's no model in the box; you make one.

### 1. Collect

Bring up the camera and the keyboard collector, then drive the robot around and label what
it sees:

```bash
ros2 launch collision_avoider data_collection.launch.py
```

`f` saves the current frame as **free**, `b` saves it as **blocked**, `q` quits. Frames land
in `dataset/free/` and `dataset/blocked/` as 224×224 JPEGs. Aim for roughly balanced classes,
~100+ each, across different orientations, lighting, obstacles, and floor textures — the more
your dataset looks like the messy reality of your room, the better the robot behaves in it.

### 2. Train

Transfer-learn a ResNet18 into the free/blocked classifier. Runs on a host GPU box or on the
Ventuno itself (it picks CUDA if it finds it, else CPU):

```bash
python3 tools/train_collision_resnet18.py --dataset dataset \
    --output models/collision_resnet18.pth --epochs 30
```

### 3. Lower to ExecuTorch

Convert the trained weights to an on-device `.pte`. For the NPU model, this runs **on the
board**, and there's one gotcha that will waste an afternoon if you miss it:

```bash
python3 tools/export_resnet18_qnn.py \
    --weights models/collision_resnet18.pth \
    --output models/collision_resnet18_qnn.pte \
    --calibration-dir dataset --soc-model QCS8300
```

The `--calibration-dir dataset` is not optional. QNN's 8-bit quantization calibrates its
numeric ranges by running real inputs through the model, and those inputs have to be
normalized the same way the runtime normalizes them. Skip it and the quantized model loads
happily and then predicts **garbage** — a failure that looks like a bug everywhere except
where it actually is. (`--soc-model QCS8300` is the other board-specific detail: the Ventuno's
chipset validates as HTP V75, which the project maps onto ExecuTorch's `QCS8300` target.)

There's an XNNPACK export for the CPU fallback too (`tools/export_resnet18_cpu.py`), and an
ONNX export for inspection.

### 4. Run

Sanity-check the classifier offline first, on the bundled sample images — no camera, no robot:

```bash
ros2 launch collision_classifier dataset_classifier.launch.py backend:=cpu
ros2 topic echo /collision/classification
```

Then, on the robot, dry-run it before you let it move:

```bash
ros2 launch launch/collision_avoidance.launch.py publish_cmd_vel:=false
```

Walk an obstacle in and out of frame, watch the logged `P(blocked)` and the commands it
*would* send. When that looks right, let it drive:

```bash
ros2 launch launch/collision_avoidance.launch.py
ros2 launch launch/collision_avoidance.launch.py blocked_threshold:=0.6   # more cautious
```

## Tuning

Everything's in `src/collision_avoider/config/avoider.yaml`. The three that matter:

| Parameter | Default | What it does |
|---|---|---|
| `blocked_threshold` | 0.5 | P(blocked) above which it turns instead of driving; raise it to be bolder, lower to be timid |
| `base_speed` | 0.15 | Forward speed when free (m/s) |
| `turn_speed` | 0.6 | In-place rotation speed when blocked (rad/s) |
| `prob_smoothing` | 0.5 | EMA weight on each new P(blocked); lower = steadier, higher = twitchier |
| `turn_direction` | +1 | Which way it turns when blocked (+1 left / −1 right) |

`blocked_threshold` is the personality knob. At 0.5 the robot commits the moment it's more
likely blocked than not; push it to 0.6–0.7 and it turns away earlier and drives into fewer
things, at the cost of occasionally shying away from clear paths.

## What was hard

**QNN calibration silently poisoning the model.** The single nastiest bug in the project:
export the NPU model without pointing calibration at real, correctly-normalized frames and it
runs fine and predicts nonsense. No crash, no error — just a robot that thinks the open floor
is a wall. Getting calibration inputs to match runtime preprocessing exactly is the whole
game with quantized NPU models.

**Keeping preprocessing in lockstep in three places.** The ImageNet normalization has to be
identical at training time (`tools/collision_model.py`), at export-calibration time, and at
runtime in C++ (`preprocess.cpp`). Any drift between them and accuracy quietly craters.

**The label-ordering trap.** `ImageFolder` sorting `blocked` before `free` alphabetically
means index 0 is *blocked*, which is the opposite of what most people assume. An explicit
`blocked_index` threaded through the classifier and controller keeps everyone honest.

**Standing up ExecuTorch + QNN on the board at all.** Shared with the object-tracking
project and written up in `.claude/skills/ventuno-setup/references/executorch-qnn.md` —
unrecognised chipset, HTP version, FastRPC/DSP paths, on-device ExecuTorch build.

## What's next

- More than two classes — a "ledge/dropoff" class, or per-direction free/blocked so it can
  steer rather than just spin
- A short memory so it doesn't oscillate in a corner
- Combining the learned classifier with the object-follower for a robot that chases you *and*
  dodges furniture
- Active data collection: let the robot flag frames it's unsure about for you to label

## Links

- **Repo:** https://github.com/TheOutcastVirus/ventuno-collision-avoidance
- **Companion project:** [ventuno-object-tracking](https://github.com/TheOutcastVirus/ventuno-object-tracking)
  — same board, same setup, object-following instead of obstacle-avoidance
- **Board setup and debugging notes:** `.claude/skills/ventuno-setup/` — full ExecuTorch/QNN
  bring-up, the Create 3 connection, and DDS tuning, written as a plain-markdown agent skill
  that a coding agent (or a human) can read to debug the board.

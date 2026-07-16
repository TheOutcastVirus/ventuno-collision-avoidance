"""Shared model + preprocessing definitions for the collision-avoidance classifier.

Used by the training and export scripts so the architecture, class count, and
input normalization stay identical across train -> ONNX -> XNNPACK -> QNN. The
same normalization is applied at runtime in the C++ node
(src/collision_classifier/src/preprocess.cpp); keep the two in sync.

Design note: model construction is deliberately **torch-only** for the export
path. Lowering to ExecuTorch (.pte) runs on the Ventuno inside the ExecuTorch
venv, which is pinned to a specific torch and has no compatible torchvision.
So the ResNet18 used for export is inlined here with the *exact same
state_dict keys* as torchvision.models.resnet18 — a checkpoint trained on a
host with torchvision loads straight into it. torchvision is only needed for
training (ImageFolder/transforms and ImageNet-pretrained weights) and is
imported lazily.
"""

import torch
import torch.nn as nn

# ImageNet statistics (RGB), applied after scaling pixels to [0, 1].
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision.datasets.ImageFolder sorts class folders alphabetically, so
# dataset/{blocked,free} maps to index 0 = blocked, 1 = free. The runtime node
# uses blocked_index=0 to match.
CLASS_NAMES = ["blocked", "free"]


# ── Inlined ResNet18 (torch-only) ────────────────────────────────────────────
# Layer/attribute names mirror torchvision.models.resnet exactly so the produced
# state_dict keys are identical (conv1, bn1, layer{1..4}.{0,1}.conv/bn[/downsample],
# fc). This lets a torchvision-trained checkpoint load with strict=True.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _torchvision_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    import torchvision  # lazy: only needed on the training host

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Build the ResNet18 classifier.

    pretrained=True (training): uses torchvision to get ImageNet weights.
    pretrained=False (export/inference): uses the torch-only inlined ResNet18,
    so lowering runs on the board without torchvision. Weights come from the
    trained checkpoint via load_trained().
    """
    if pretrained:
        return _torchvision_resnet18(num_classes, pretrained=True)
    return ResNet18(num_classes=num_classes)


def load_trained(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Build the (torch-only) model and load trained weights."""
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def eval_transform(image_size: int = 224):
    """Torch-only PIL->tensor transform matching the C++ runtime preprocessing.

    Returns a callable ``pil_image -> normalized CHW float tensor``. Avoids
    torchvision.transforms so QNN calibration can run in the board's export
    venv (needs only Pillow + numpy).
    """
    import numpy as np

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _transform(pil_image):
        img = pil_image.convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        chw = torch.from_numpy(arr).permute(2, 0, 1)
        return (chw - mean) / std

    return _transform

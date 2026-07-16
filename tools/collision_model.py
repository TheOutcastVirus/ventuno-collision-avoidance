"""Shared model + preprocessing definitions for the collision-avoidance classifier.

Used by the training and export scripts so the architecture, class count, and
input normalization stay identical across train -> ONNX -> XNNPACK -> QNN. The
same normalization is applied at runtime in the C++ node
(src/collision_classifier/src/preprocess.cpp); keep the two in sync.
"""

import torch
import torchvision

# ImageNet statistics (RGB), applied after scaling pixels to [0, 1].
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# torchvision.datasets.ImageFolder sorts class folders alphabetically, so
# dataset/{blocked,free} maps to index 0 = blocked, 1 = free. The runtime node
# uses blocked_index=0 to match.
CLASS_NAMES = ["blocked", "free"]


def build_model(num_classes: int = 2, pretrained: bool = False) -> torch.nn.Module:
    """ResNet18 with the final layer replaced for `num_classes` outputs."""
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    else:
        model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained(weights_path: str, num_classes: int = 2) -> torch.nn.Module:
    """Build the model and load trained weights saved by the training script."""
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def eval_transform(image_size: int = 224):
    """Inference/calibration transform matching the C++ runtime preprocessing."""
    import torchvision.transforms as T

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

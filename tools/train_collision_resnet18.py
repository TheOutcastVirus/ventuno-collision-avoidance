#!/usr/bin/env python3
"""Train the free/blocked collision-avoidance classifier (ResNet18).

Transfer-learns a ResNet18 (ImageNet-pretrained) into a 2-class free/blocked
classifier from a folder of labeled images, mirroring the JetBot
train_model_resnet18.ipynb notebook but as a CLI script. Runs on a host GPU box
or on the Ventuno itself (auto-selects CUDA when available, else CPU).

Dataset layout (produced by `ros2 launch collision_avoider data_collection.launch.py`):
    <dataset>/
        free/     *.jpg
        blocked/  *.jpg

Usage:
    python3 tools/train_collision_resnet18.py --dataset dataset \
        --output models/collision_resnet18.pth --epochs 30

The output .pth (a plain state_dict) is then lowered to ExecuTorch with
tools/export_resnet18_cpu.py (CPU) and tools/export_resnet18_qnn.py (NPU).
"""

import argparse
import pathlib

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

from collision_model import IMAGENET_MEAN, IMAGENET_STD, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train the free/blocked ResNet18 classifier")
    p.add_argument("--dataset", default="dataset",
                   help="ImageFolder root containing free/ and blocked/ subfolders")
    p.add_argument("--output", default="models/collision_resnet18.pth",
                   help="Where to save the best model state_dict")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--test-size", type=int, default=50,
                   help="Number of images held out for the test split")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ColorJitter augmentation as in the JetBot notebook, then the standard
    # ImageNet normalization the model expects.
    transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = datasets.ImageFolder(args.dataset, transform)
    print(f"Loaded {len(dataset)} images, classes: {dataset.classes}")
    if len(dataset) <= args.test_size:
        raise SystemExit(
            f"Dataset has {len(dataset)} images but --test-size is {args.test_size}; "
            "collect more data or lower --test-size.")

    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset) - args.test_size, args.test_size])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=len(dataset.classes), pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        errors = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                errors += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        accuracy = 1.0 - errors / len(test_set)
        print(f"epoch {epoch:3d}: test accuracy {accuracy:.3f}")

        if accuracy >= best_accuracy:
            torch.save(model.state_dict(), output_path)
            best_accuracy = accuracy

    print(f"Best test accuracy {best_accuracy:.3f}; saved {output_path}")


if __name__ == "__main__":
    main()

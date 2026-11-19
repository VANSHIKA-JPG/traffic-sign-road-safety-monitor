import sys
from pathlib import Path

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from resnet_dataset import test_loader
from models.resnet_model import TrafficSignResNet


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Create same architecture used during training
model = TrafficSignResNet(
    num_classes=43,
    freeze_backbone=False
).to(device)


# Load trained unfrozen-ResNet weights
MODEL_PATH = "outputs/models/resnet18_finetuned.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

print("Model loaded successfully.")

model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


average_test_loss = (
    test_loss / len(test_loader)
)

test_accuracy = (
    100 * correct / total
)


print("\nFine-Tuned ResNet-18 Test Results")
print("--------------------------------")

print(
    f"Test Loss: {average_test_loss:.4f}"
)

print(
    f"Correct Predictions: {correct}"
)

print(
    f"Total Test Images: {total}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)
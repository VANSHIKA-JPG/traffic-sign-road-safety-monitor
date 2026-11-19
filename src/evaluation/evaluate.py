import sys
from pathlib import Path

import torch
import torch.nn as nn

# --------------------------------------------------
# ALLOW IMPORTS FROM src/
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from dataset import test_loader
from models.simple_cnn import SimpleCNN


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD MODEL ARCHITECTURE
# --------------------------------------------------

model = SimpleCNN().to(device)


# --------------------------------------------------
# LOAD TRAINED WEIGHTS
# --------------------------------------------------

MODEL_PATH = "outputs/models/simple_cnn_augmented.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

print("Model loaded successfully.")


# --------------------------------------------------
# EVALUATION MODE
# --------------------------------------------------

model.eval()

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0


# --------------------------------------------------
# TEST LOOP
# --------------------------------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate test loss
        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        # Find class with highest logit
        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

average_test_loss = (
    test_loss / len(test_loader)
)

test_accuracy = (
    100 * correct / total
)


print("\nTest Results")
print("-----------------------------")

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
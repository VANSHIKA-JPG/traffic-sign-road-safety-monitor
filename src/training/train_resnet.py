import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from resnet_dataset import train_loader, val_loader
from models.resnet_model import TrafficSignResNet


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = TrafficSignResNet(
    num_classes=43,
    freeze_backbone=True
).to(device)


# --------------------------------------------------
# LOSS
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# Only train parameters where requires_grad=True
# --------------------------------------------------

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.001
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUM_EPOCHS = 3


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)}"
            )

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_SAVE_PATH = (
    "outputs/models/resnet18_frozen.pth"
)

Path("outputs/models").mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print("Training complete.")
print(
    f"Model saved to {MODEL_SAVE_PATH}"
)
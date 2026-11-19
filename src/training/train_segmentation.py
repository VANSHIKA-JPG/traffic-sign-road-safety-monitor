import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from segmentation.segmentation_dataset import train_loader, val_loader
from models.segmentation_model import UNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "unet_drivable_best.pth"


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

model = UNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# IoU FUNCTION
# --------------------------------------------------

def calculate_iou(logits, masks):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    intersection = (
        predictions * masks
    ).sum(dim=(1, 2, 3))

    union = (
        predictions + masks
        - predictions * masks
    ).sum(dim=(1, 2, 3))

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return iou.mean().item()


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_val_iou = 0.0


for epoch in range(EPOCHS):

    # ==================================================
    # TRAIN
    # ==================================================

    model.train()

    train_loss = 0.0
    train_iou = 0.0


    for batch_index, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)


        optimizer.zero_grad()


        logits = model(images)


        loss = criterion(
            logits,
            masks
        )


        loss.backward()

        optimizer.step()


        train_loss += loss.item()

        train_iou += calculate_iou(
            logits.detach(),
            masks
        )


        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)}"
            )


    train_loss /= len(train_loader)

    train_iou /= len(train_loader)


    # ==================================================
    # VALIDATION
    # ==================================================

    model.eval()

    val_loss = 0.0
    val_iou = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)


            logits = model(images)


            loss = criterion(
                logits,
                masks
            )


            val_loss += loss.item()

            val_iou += calculate_iou(
                logits,
                masks
            )


    val_loss /= len(val_loader)

    val_iou /= len(val_loader)


    # ==================================================
    # RESULTS
    # ==================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train IoU: {train_iou:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val IoU: {val_iou:.4f}"
    )


    # ==================================================
    # SAVE BEST MODEL
    # ==================================================

    if val_iou > best_val_iou:

        best_val_iou = val_iou

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best model saved "
            f"(Val IoU: {best_val_iou:.4f})"
        )


print("\nTraining complete.")

print(
    "Best Validation IoU:",
    f"{best_val_iou:.4f}"
)

print(
    "Best model:",
    MODEL_PATH
)
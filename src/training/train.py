import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from dataset import train_loader, val_loader
from models.simple_cnn import SimpleCNN


# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. CREATE MODEL
# --------------------------------------------------

model = SimpleCNN().to(device)


# --------------------------------------------------
# 3. LOSS FUNCTION
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 4. OPTIMIZER
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 5. TRAINING SETTINGS
# --------------------------------------------------

NUM_EPOCHS = 3


# --------------------------------------------------
# 6. TRAINING LOOP
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Clear gradients from previous batch
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(outputs, labels)


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Add batch loss
        running_loss += loss.item()


        # Find predicted class
        _, predicted = torch.max(
            outputs,
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total


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


    val_loss = val_loss / len(val_loader)

    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


# --------------------------------------------------
# 7. SAVE TRAINED MODEL
# --------------------------------------------------

Path("outputs/models").mkdir(
    parents=True,
    exist_ok=True
)

MODEL_SAVE_PATH = "outputs/models/simple_cnn_augmented.pth"

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print("Training complete.")
print(
    f"Model saved to {MODEL_SAVE_PATH}"
)

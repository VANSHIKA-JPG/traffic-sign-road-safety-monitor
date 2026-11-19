import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from resnet_dataset import test_loader
from models.resnet_model import TrafficSignResNet
from class_names import CLASS_NAMES


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD FINE-TUNED RESNET
# --------------------------------------------------

model = TrafficSignResNet(
    num_classes=43,
    freeze_backbone=False
).to(device)

MODEL_PATH = "outputs/models/resnet18_finetuned.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# COLLECT TEST PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# PER-CLASS ACCURACY
# --------------------------------------------------

class_results = []

for class_id in range(len(CLASS_NAMES)):

    total = cm[class_id].sum()

    correct = cm[
        class_id,
        class_id
    ]

    if total > 0:

        accuracy = (
            100 * correct / total
        )

    else:
        accuracy = 0

    class_results.append(
        (
            class_id,
            accuracy,
            correct,
            total
        )
    )


# --------------------------------------------------
# SORT FROM WORST TO BEST
# --------------------------------------------------

class_results.sort(
    key=lambda x: x[1]
)


print("\nFine-Tuned ResNet - Per-Class Accuracy")
print("--------------------------------------------------------------")

for class_id, accuracy, correct, total in class_results:

    print(
        f"{CLASS_NAMES[class_id]:45s} "
        f"{accuracy:6.2f}% "
        f"({correct}/{total})"
    )


# --------------------------------------------------
# FIVE WORST CLASSES
# --------------------------------------------------

print("\n5 Worst Performing Classes")
print("---------------------------------------------")

for class_id, accuracy, correct, total in class_results[:5]:

    print(
        f"{CLASS_NAMES[class_id]:45s} "
        f"{accuracy:6.2f}% "
        f"({correct}/{total})"
    )
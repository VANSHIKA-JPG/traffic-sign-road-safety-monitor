import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from dataset import test_loader
from models.simple_cnn import SimpleCNN
from class_names import CLASS_NAMES


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = SimpleCNN().to(device)

MODEL_PATH = "outputs/models/simple_cnn_augmented.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# COLLECT PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
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

print("\nPer-Class Accuracy")
print("---------------------------------------------")

class_accuracies = []

for class_id in range(len(CLASS_NAMES)):

    total_class_samples = cm[class_id].sum()

    correct_class_predictions = cm[
        class_id,
        class_id
    ]

    if total_class_samples > 0:

        accuracy = (
            100
            * correct_class_predictions
            / total_class_samples
        )

    else:
        accuracy = 0

    class_accuracies.append(
        (class_id, accuracy)
    )

    print(
        f"{class_id:2d} "
        f"{CLASS_NAMES[class_id]:45s} "
        f"{accuracy:6.2f}%"
    )


# --------------------------------------------------
# FIND WORST CLASSES
# --------------------------------------------------

class_accuracies.sort(
    key=lambda x: x[1]
)

print("\n5 Worst Performing Classes")
print("---------------------------------------------")

for class_id, accuracy in class_accuracies[:5]:

    print(
        f"{CLASS_NAMES[class_id]:45s} "
        f"{accuracy:.2f}%"
    )


# --------------------------------------------------
# DISPLAY CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(18, 18)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(len(CLASS_NAMES)))
)

display.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
    values_format="d"
)

ax.set_title(
    "Augmented Simple CNN - GTSRB Test Confusion Matrix"
)

plt.tight_layout()


# --------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------

output_path = Path(
    "outputs/figures"
)

output_path.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output_path / "simple_cnn_augmented_confusion_matrix.png",
    dpi=200
)

print(
    "\nConfusion matrix saved to "
    "outputs/figures/simple_cnn_confusion_matrix.png"
)

plt.show()
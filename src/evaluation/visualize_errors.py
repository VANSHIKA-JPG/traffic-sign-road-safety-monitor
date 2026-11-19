import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

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

model.load_state_dict(
    torch.load(
        "outputs/models/simple_cnn.pth",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# COLLECT MISCLASSIFIED IMAGES
# --------------------------------------------------

errors = []

MAX_ERRORS = 12

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Convert logits into probabilities
        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

        for i in range(len(labels)):

            if predicted[i] != labels[i]:

                errors.append(
                    (
                        images[i].cpu(),
                        labels[i].item(),
                        predicted[i].item(),
                        confidence[i].item()
                    )
                )

                if len(errors) >= MAX_ERRORS:
                    break

        if len(errors) >= MAX_ERRORS:
            break


# --------------------------------------------------
# VISUALIZE ERRORS
# --------------------------------------------------

fig, axes = plt.subplots(
    3,
    4,
    figsize=(14, 10)
)

for ax, error in zip(
    axes.flat,
    errors
):

    image, actual, predicted, confidence = error

    image = image.permute(
        1,
        2,
        0
    )

    ax.imshow(image)

    ax.set_title(
        f"Actual: {CLASS_NAMES[actual]}\n"
        f"Predicted: {CLASS_NAMES[predicted]}\n"
        f"Confidence: {confidence * 100:.1f}%",
        fontsize=8
    )

    ax.axis("off")


plt.suptitle(
    "Simple CNN - Misclassified GTSRB Test Images",
    fontsize=14
)

plt.tight_layout()


# --------------------------------------------------
# SAVE FIGURE
# --------------------------------------------------

output_dir = Path(
    "outputs/figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    output_dir / "simple_cnn_errors.png",
    dpi=200,
    bbox_inches="tight"
)

print(
    "Error visualization saved to "
    "outputs/figures/simple_cnn_errors.png"
)

plt.show()
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from segmentation.segmentation_dataset import val_dataset
from models.segmentation_model import UNet


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD BEST MODEL
# --------------------------------------------------

model = UNet().to(device)

MODEL_PATH = "outputs/models/unet_drivable_best.pth"

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Best U-Net model loaded.")


# --------------------------------------------------
# SELECT VALIDATION SAMPLES
# --------------------------------------------------

sample_indices = [0, 100, 500, 1000, 1500]


for sample_number, index in enumerate(
    sample_indices,
    start=1
):

    image, ground_truth = val_dataset[index]

    input_tensor = (
        image.unsqueeze(0).to(device)
    )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(input_tensor)

        probability = torch.sigmoid(logits)

        prediction = (
            probability > 0.5
        ).float()


    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )

    ground_truth = (
        ground_truth
        .squeeze()
        .numpy()
    )


    # --------------------------------------------------
    # RESTORE IMAGE FOR DISPLAY
    # --------------------------------------------------

    display_image = image.clone()

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    display_image = (
        display_image * std + mean
    )

    display_image = (
        display_image
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )


    # --------------------------------------------------
    # CALCULATE SAMPLE IoU
    # --------------------------------------------------

    pred_tensor = torch.tensor(prediction)
    gt_tensor = torch.tensor(ground_truth)

    intersection = (
        pred_tensor * gt_tensor
    ).sum()

    union = (
        pred_tensor
        + gt_tensor
        - pred_tensor * gt_tensor
    ).sum()

    iou = (
        (intersection + 1e-6)
        / (union + 1e-6)
    ).item()


    print(
        f"Sample {sample_number} IoU: "
        f"{iou:.4f}"
    )


    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )


    axes[0].imshow(display_image)
    axes[0].set_title("Road Image")


    axes[1].imshow(
        ground_truth,
        cmap="gray"
    )
    axes[1].set_title("Ground Truth")


    axes[2].imshow(
        prediction,
        cmap="gray"
    )
    axes[2].set_title(
        f"U-Net Prediction\nIoU = {iou:.3f}"
    )


    axes[3].imshow(display_image)

    axes[3].imshow(
        prediction,
        cmap="jet",
        alpha=0.35
    )

    axes[3].set_title(
        "Predicted Road Overlay"
    )


    for ax in axes:
        ax.axis("off")


    plt.tight_layout()

    plt.show()
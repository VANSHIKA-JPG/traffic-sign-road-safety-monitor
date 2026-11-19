import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from segmentation.segmentation_dataset import val_dataset
from models.segmentation_model import UNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "outputs/models/unet_drivable_best.pth"
BATCH_SIZE = 8
THRESHOLD = 0.5


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Best U-Net model loaded.")


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

iou_sum = 0.0
iou_count = 0

empty_mask_count = 0
correct_empty_count = 0

false_positive_pixels = 0
empty_total_pixels = 0


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

with torch.no_grad():

    for images, masks in val_loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities > THRESHOLD
        ).float()


        # Evaluate each image separately
        for prediction, mask in zip(
            predictions,
            masks
        ):

            ground_truth_pixels = mask.sum().item()

            predicted_pixels = prediction.sum().item()


            # --------------------------------------
            # CASE 1:
            # Ground truth contains drivable road
            # --------------------------------------

            if ground_truth_pixels > 0:

                intersection = (
                    prediction * mask
                ).sum().item()

                union = (
                    prediction
                    + mask
                    - prediction * mask
                ).sum().item()

                if union > 0:

                    iou = intersection / union

                    iou_sum += iou
                    iou_count += 1


            # --------------------------------------
            # CASE 2:
            # Ground truth contains NO drivable road
            # --------------------------------------

            else:

                empty_mask_count += 1

                empty_total_pixels += mask.numel()

                false_positive_pixels += (
                    predicted_pixels
                )

                # Perfect only if model also predicts
                # no drivable pixels
                if predicted_pixels == 0:

                    correct_empty_count += 1


# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

mean_non_empty_iou = (
    iou_sum / iou_count
    if iou_count > 0
    else 0.0
)


empty_mask_accuracy = (
    correct_empty_count / empty_mask_count
    if empty_mask_count > 0
    else 0.0
)


empty_false_positive_rate = (
    false_positive_pixels / empty_total_pixels
    if empty_total_pixels > 0
    else 0.0
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nSegmentation Evaluation")
print("--------------------------------")

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Images with drivable area:",
    iou_count
)

print(
    "Images with empty masks:",
    empty_mask_count
)

print(
    "\nMean IoU on non-empty masks:",
    f"{mean_non_empty_iou:.4f}"
)

print(
    "Empty-mask exact accuracy:",
    f"{empty_mask_accuracy:.4f}"
)

print(
    "False-positive pixel rate "
    "on empty masks:",
    f"{empty_false_positive_rate:.4f}"
)
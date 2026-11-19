import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ANNOTATION_PATH = Path(
    "data/bdd100k/samples.json"
)

DATASET_ROOT = Path(
    "data/bdd100k"
)


# --------------------------------------------------
# LOAD SAMPLE INFORMATION
# --------------------------------------------------

with open(
    ANNOTATION_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


sample = data["samples"][0]


image_path = (
    DATASET_ROOT
    / sample["filepath"]
)

mask_path = (
    DATASET_ROOT
    / sample["drivable"]["mask_path"]
)


print("Image:", image_path)
print("Mask:", mask_path)


# --------------------------------------------------
# LOAD IMAGE AND MASK
# --------------------------------------------------

image = Image.open(
    image_path
).convert("RGB")

mask = Image.open(
    mask_path
)


print("Image size:", image.size)
print("Mask size:", mask.size)
print("Mask mode:", mask.mode)


# --------------------------------------------------
# INSPECT MASK VALUES
# --------------------------------------------------

mask_array = np.array(mask)

print(
    "Unique mask values:",
    np.unique(mask_array)
)


# --------------------------------------------------
# VISUALIZE ORIGINAL IMAGE
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(image)

plt.title(
    "Original BDD100K Road Image"
)

plt.axis("off")

plt.show()


# --------------------------------------------------
# VISUALIZE SEGMENTATION MASK
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(
    mask_array,
    cmap="gray"
)

plt.title(
    "BDD100K Drivable-Area Mask"
)

plt.axis("off")

plt.show()


# --------------------------------------------------
# OVERLAY MASK ON IMAGE
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(image)

plt.imshow(
    mask_array,
    cmap="jet",
    alpha=0.35
)

plt.title(
    "Drivable-Area Mask Overlay"
)

plt.axis("off")

plt.show()
# --------------------------------------------------
# VISUALIZE EACH MASK VALUE SEPARATELY
# --------------------------------------------------

for value in [0, 1, 2]:

    binary_mask = (
        mask_array == value
    )

    plt.figure(figsize=(12, 7))

    plt.imshow(
        binary_mask,
        cmap="gray"
    )

    plt.title(
        f"Pixels with Mask Value = {value}"
    )

    plt.axis("off")

    plt.show()
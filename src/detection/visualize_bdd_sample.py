import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
# LOAD ANNOTATIONS
# --------------------------------------------------

print("Loading annotations...")

with open(
    ANNOTATION_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


sample = data["samples"][0]


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

relative_path = sample["filepath"]

image_path = DATASET_ROOT / relative_path


print("Image:", image_path)


image = Image.open(
    image_path
).convert("RGB")


image_width, image_height = image.size


print(
    "Image size:",
    image_width,
    "x",
    image_height
)


# --------------------------------------------------
# GET DETECTIONS
# --------------------------------------------------

detections = sample[
    "detections"
]["detections"]


print(
    "Number of objects:",
    len(detections)
)


# --------------------------------------------------
# DRAW IMAGE
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 9)
)

ax.imshow(image)


# --------------------------------------------------
# DRAW BOUNDING BOXES
# --------------------------------------------------

for detection in detections:

    label = detection["label"]

    x, y, width, height = detection[
        "bounding_box"
    ]


    # FiftyOne bounding boxes are normalized.
    # Convert them into pixel coordinates.

    x_pixel = x * image_width

    y_pixel = y * image_height

    width_pixel = width * image_width

    height_pixel = height * image_height


    rectangle = patches.Rectangle(
        (x_pixel, y_pixel),
        width_pixel,
        height_pixel,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none"
    )


    ax.add_patch(rectangle)


    ax.text(
        x_pixel,
        y_pixel,
        label,
        fontsize=7,
        backgroundcolor="white"
    )


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

ax.axis("off")

plt.title(
    "BDD100K Original Detection Annotations"
)

plt.tight_layout()

plt.show()
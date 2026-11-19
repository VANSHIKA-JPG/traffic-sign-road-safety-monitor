import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# --------------------------------------------------
# PATHS
# --------------------------------------------------

IMAGE_DIR = Path(
    "data/bdd100k_yolo/images/train"
)

LABEL_DIR = Path(
    "data/bdd100k_yolo/labels/train"
)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "motorcycle",
    6: "bicycle",
    7: "traffic light",
    8: "traffic sign",
}


# --------------------------------------------------
# SELECT RANDOM IMAGE
# --------------------------------------------------

image_files = list(
    IMAGE_DIR.glob("*.jpg")
)

image_path = random.choice(
    image_files
)

label_path = (
    LABEL_DIR
    / f"{image_path.stem}.txt"
)


print("Image:", image_path)
print("Label:", label_path)


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = Image.open(
    image_path
).convert("RGB")

image_width, image_height = image.size


# --------------------------------------------------
# DISPLAY IMAGE
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 9)
)

ax.imshow(image)


# --------------------------------------------------
# LOAD YOLO LABELS
# --------------------------------------------------

if label_path.exists():

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()


    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue


        class_id = int(
            values[0]
        )

        x_center = float(
            values[1]
        )

        y_center = float(
            values[2]
        )

        width = float(
            values[3]
        )

        height = float(
            values[4]
        )


        # ------------------------------------------
        # CONVERT YOLO NORMALIZED COORDINATES
        # BACK TO PIXEL COORDINATES
        # ------------------------------------------

        box_width = (
            width * image_width
        )

        box_height = (
            height * image_height
        )


        x_pixel = (
            x_center * image_width
            - box_width / 2
        )

        y_pixel = (
            y_center * image_height
            - box_height / 2
        )


        # ------------------------------------------
        # DRAW BOX
        # ------------------------------------------

        rectangle = patches.Rectangle(
            (x_pixel, y_pixel),
            box_width,
            box_height,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(
            rectangle
        )


        # ------------------------------------------
        # DRAW CLASS NAME
        # ------------------------------------------

        class_name = CLASS_NAMES[
            class_id
        ]

        ax.text(
            x_pixel,
            y_pixel,
            class_name,
            fontsize=7,
            backgroundcolor="white"
        )


ax.axis("off")

plt.title(
    "Converted BDD100K YOLO Labels"
)

plt.tight_layout()

plt.show()
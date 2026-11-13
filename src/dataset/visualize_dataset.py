import sys
import random
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------
# ALLOW IMPORTS FROM src/
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

sys.path.append(
    str(SRC_DIR)
)


from dataset import train_dataset
from class_names import CLASS_NAMES


# --------------------------------------------------
# VISUALIZE RANDOM TRAFFIC SIGNS
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)


for ax in axes.flat:

    index = random.randint(
        0,
        len(train_dataset) - 1
    )

    image, label = train_dataset[index]


    # PyTorch:
    # [C, H, W]
    #
    # Matplotlib:
    # [H, W, C]

    image = image.permute(
        1,
        2,
        0
    )


    ax.imshow(image)

    ax.set_title(
        CLASS_NAMES[label],
        fontsize=8
    )

    ax.axis("off")


plt.tight_layout()

plt.show()
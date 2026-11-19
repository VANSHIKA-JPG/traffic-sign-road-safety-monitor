from pathlib import Path

import torch
from ultralytics import YOLO


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = 0 if torch.cuda.is_available() else "cpu"

print(
    "Using device:",
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)


# --------------------------------------------------
# LOAD PRETRAINED YOLO
# --------------------------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------------------------
# INPUT IMAGE
# --------------------------------------------------

IMAGE_PATH = "data/sample_images/road.jpg"


# --------------------------------------------------
# RUN DETECTION
# --------------------------------------------------

results = model.predict(
    source=IMAGE_PATH,
    device=device,
    conf=0.25,
    save=True,
    project="outputs/detection",
    name="first_test"
)


# --------------------------------------------------
# PRINT DETECTIONS
# --------------------------------------------------

result = results[0]

print("\nDetected Objects")
print("--------------------------------")


for box in result.boxes:

    class_id = int(
        box.cls.item()
    )

    confidence = float(
        box.conf.item()
    )

    class_name = model.names[
        class_id
    ]

    print(
        f"{class_name:20s} "
        f"{confidence * 100:.2f}%"
    )


print(
    "\nAnnotated image saved in "
    "outputs/detection/first_test/"
)
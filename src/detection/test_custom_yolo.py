from pathlib import Path
import random
from ultralytics import YOLO


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = (
    "runs/detect/outputs/yolo/"
    "bdd100k_yolo11n/weights/best.pt"
)

VAL_IMAGE_DIR = Path(
    "data/bdd100k_yolo/images/val"
)


# --------------------------------------------------
# LOAD OUR CUSTOM MODEL
# --------------------------------------------------

model = YOLO(MODEL_PATH)

print("Custom YOLO model loaded.")
print("Classes:", model.names)


# --------------------------------------------------
# RANDOM VALIDATION IMAGES
# --------------------------------------------------

images = list(
    VAL_IMAGE_DIR.glob("*.jpg")
)

random.seed(42)

selected_images = random.sample(
    images,
    5
)


# --------------------------------------------------
# RUN DETECTION
# --------------------------------------------------

for image_path in selected_images:

    print("\nProcessing:", image_path.name)

    model.predict(
        source=str(image_path),
        conf=0.25,
        device=0,
        save=True,

        project="outputs/yolo_visual_test",
        name="predictions",

        exist_ok=True
    )


print("\nVisual test complete.")
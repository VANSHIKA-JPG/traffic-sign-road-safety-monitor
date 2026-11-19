import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.segmentation_model import UNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

YOLO_MODEL_PATH = (
    "runs/detect/outputs/yolo/"
    "bdd100k_yolo11n/weights/best.pt"
)

UNET_MODEL_PATH = (
    "outputs/models/unet_drivable_best.pth"
)

IMAGE_PATH = (
    "data/bdd100k_yolo/images/val/"
    "b4c195ae-4dfb18d9.jpg"
)

OUTPUT_DIR = Path("outputs/spatial_integration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "yolo_unet_result.jpg"

SEGMENTATION_SIZE = (256, 256)

# Objects relevant to immediate road risk
RISK_CLASSES = {
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("Loading YOLO...")

yolo = YOLO(YOLO_MODEL_PATH)

print("Loading U-Net...")

unet = UNet().to(device)

unet.load_state_dict(
    torch.load(
        UNET_MODEL_PATH,
        map_location=device
    )
)

unet.eval()

print("Models loaded.")


# --------------------------------------------------
# SEGMENTATION TRANSFORM
# --------------------------------------------------

segmentation_transform = transforms.Compose([
    transforms.Resize(SEGMENTATION_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

height, width = image_bgr.shape[:2]

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

pil_image = Image.fromarray(image_rgb)


# --------------------------------------------------
# U-NET ROAD SEGMENTATION
# --------------------------------------------------

seg_input = segmentation_transform(
    pil_image
).unsqueeze(0).to(device)


with torch.no_grad():

    logits = unet(seg_input)

    probability = torch.sigmoid(logits)

    road_mask = (
        probability > 0.5
    ).float()


road_mask = (
    road_mask
    .squeeze()
    .cpu()
    .numpy()
    .astype(np.uint8)
)


# Resize mask back to original image resolution
road_mask = cv2.resize(
    road_mask,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)


# --------------------------------------------------
# YOLO DETECTION
# --------------------------------------------------

results = yolo.predict(
    source=IMAGE_PATH,
    conf=0.25,
    device=0,
    verbose=False
)

result = results[0]


print("\nDetected Objects")
print("---------------------------------------------")


# --------------------------------------------------
# ANALYZE OBJECT POSITION
# --------------------------------------------------

for box in result.boxes:

    class_id = int(box.cls.item())

    class_name = yolo.names[class_id]

    confidence = float(box.conf.item())


    if class_name not in RISK_CLASSES:
        continue


    x1, y1, x2, y2 = box.xyxy[0].tolist()

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))


    # --------------------------------------------------
    # OBJECT GROUND POINT
    # --------------------------------------------------
    #
    # We use the bottom-center of the bounding box.
    # For road objects, this approximates where the
    # object touches the ground.
    # --------------------------------------------------

    ground_x = int(
        (x1 + x2) / 2
    )

    ground_y = y2


    # --------------------------------------------------
    # CHECK ROAD MASK
    # --------------------------------------------------

    is_on_road = (
        road_mask[
            ground_y,
            ground_x
        ] == 1
    )


    status = (
        "ON ROAD"
        if is_on_road
        else "OFF ROAD"
    )


    print(
        f"{class_name:12s} "
        f"{confidence * 100:6.2f}% "
        f"{status}"
    )


    # --------------------------------------------------
    # DRAW BOX
    # --------------------------------------------------

    if is_on_road:
        box_color = (0, 0, 255)
    else:
        box_color = (0, 255, 0)


    cv2.rectangle(
        image_bgr,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )


    label = (
        f"{class_name} "
        f"{confidence:.2f} "
        f"{status}"
    )


    cv2.putText(
        image_bgr,
        label,
        (
            x1,
            max(20, y1 - 8)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        box_color,
        1
    )


    # Ground-contact point
    cv2.circle(
        image_bgr,
        (ground_x, ground_y),
        4,
        box_color,
        -1
    )


# --------------------------------------------------
# DRAW ROAD OVERLAY
# --------------------------------------------------

overlay = image_bgr.copy()

road_layer = np.zeros_like(image_bgr)

road_layer[
    road_mask == 1
] = (255, 0, 0)


overlay = cv2.addWeighted(
    overlay,
    1.0,
    road_layer,
    0.30,
    0
)


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

cv2.imwrite(
    str(OUTPUT_PATH),
    overlay
)


print("\nLegend:")
print("RED box   = ON ROAD")
print("GREEN box = OFF ROAD")
print("BLUE area = predicted drivable road")

print(
    "\nResult saved to:",
    OUTPUT_PATH
)
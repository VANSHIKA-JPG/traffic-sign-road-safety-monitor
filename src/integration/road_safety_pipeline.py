import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.resnet_model import TrafficSignResNet
from class_names import CLASS_NAMES


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

YOLO_MODEL_PATH = (
    "runs/detect/outputs/yolo/"
    "bdd100k_yolo11n/weights/best.pt"
)

RESNET_MODEL_PATH = (
    "outputs/models/resnet18_finetuned.pth"
)


# --------------------------------------------------
# TEST IMAGE
# --------------------------------------------------

IMAGE_PATH = (
    "data/bdd100k_yolo/images/val/"
    "c482a775-8e9c09c8.jpg"
)


# --------------------------------------------------
# OUTPUT FOLDERS
# --------------------------------------------------

OUTPUT_DIR = Path(
    "outputs/integration"
)

CROP_DIR = OUTPUT_DIR / "sign_crops"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD YOLO
# --------------------------------------------------

print("Loading YOLO...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

print("YOLO loaded.")


# --------------------------------------------------
# LOAD RESNET
# --------------------------------------------------

print("Loading ResNet...")

resnet_model = TrafficSignResNet(
    num_classes=43,
    freeze_backbone=False
)

resnet_model.load_state_dict(
    torch.load(
        RESNET_MODEL_PATH,
        map_location=device
    )
)

resnet_model = resnet_model.to(device)

resnet_model.eval()

print("ResNet loaded.")


# --------------------------------------------------
# RESNET PREPROCESSING
# --------------------------------------------------

resnet_transform = transforms.Compose([

    transforms.Resize(
        (128, 128)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# --------------------------------------------------
# LOAD ROAD IMAGE
# --------------------------------------------------

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# --------------------------------------------------
# YOLO DETECTION
# --------------------------------------------------

print("\nRunning YOLO...")

results = yolo_model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    device=0,
    verbose=False
)

result = results[0]


# --------------------------------------------------
# PROCESS TRAFFIC SIGNS
# --------------------------------------------------

sign_count = 0


for box in result.boxes:

    class_id = int(
        box.cls.item()
    )

    yolo_confidence = float(
        box.conf.item()
    )

    class_name = yolo_model.names[
        class_id
    ]


    # We only send traffic signs to ResNet
    if class_name != "traffic sign":
        continue


    sign_count += 1


    # --------------------------------------------------
    # GET BOUNDING BOX
    # --------------------------------------------------

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    x1 = max(
        0,
        int(x1)
    )

    y1 = max(
        0,
        int(y1)
    )

    x2 = min(
        image.shape[1],
        int(x2)
    )

    y2 = min(
        image.shape[0],
        int(y2)
    )


    # --------------------------------------------------
    # CROP TRAFFIC SIGN
    # --------------------------------------------------

    sign_crop = image[
        y1:y2,
        x1:x2
    ]


    if sign_crop.size == 0:
        continue


    crop_path = (
        CROP_DIR
        / f"sign_{sign_count}.jpg"
    )

    cv2.imwrite(
        str(crop_path),
        sign_crop
    )


    # --------------------------------------------------
    # OPENCV BGR → RGB
    # --------------------------------------------------

    sign_rgb = cv2.cvtColor(
        sign_crop,
        cv2.COLOR_BGR2RGB
    )

    sign_pil = Image.fromarray(
        sign_rgb
    )


    # --------------------------------------------------
    # RESNET INPUT
    # --------------------------------------------------

    input_tensor = resnet_transform(
        sign_pil
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------
    # RESNET PREDICTION
    # --------------------------------------------------

    with torch.no_grad():

        outputs = resnet_model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )


    predicted_class = int(
        predicted.item()
    )

    resnet_confidence = float(
        confidence.item()
    )

    sign_name = CLASS_NAMES[
        predicted_class
    ]


    # --------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------

    print(
        f"\nTraffic Sign #{sign_count}"
    )

    print(
        f"YOLO confidence: "
        f"{yolo_confidence * 100:.2f}%"
    )

    print(
        f"ResNet prediction: "
        f"{sign_name}"
    )

    print(
        f"ResNet confidence: "
        f"{resnet_confidence * 100:.2f}%"
    )

    print(
        f"Crop saved: {crop_path}"
    )


    # --------------------------------------------------
    # DRAW RESULT
    # --------------------------------------------------

    label = (
        f"{sign_name} "
        f"{resnet_confidence * 100:.1f}%"
    )


    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    cv2.putText(
        image,
        label,
        (
            x1,
            max(20, y1 - 10)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# SAVE FINAL IMAGE
# --------------------------------------------------

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "integrated_result.jpg"
)

cv2.imwrite(
    str(OUTPUT_IMAGE),
    image
)


print(
    "\n--------------------------------"
)

print(
    "Traffic signs detected:",
    sign_count
)

print(
    "Final image saved to:",
    OUTPUT_IMAGE
)
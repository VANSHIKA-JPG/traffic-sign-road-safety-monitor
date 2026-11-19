import sys
import csv
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
# PATHS
# --------------------------------------------------

VIDEO_PATH = "data/sample_videos/road_test.mp4"

YOLO_MODEL_PATH = (
    "runs/detect/outputs/yolo/"
    "bdd100k_yolo11n/weights/best.pt"
)

UNET_MODEL_PATH = (
    "outputs/models/unet_drivable_best.pth"
)

OUTPUT_DIR = Path("outputs/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "road_test_features.csv"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

SEGMENTATION_SIZE = (256, 256)

CONFIDENCE_THRESHOLD = 0.30

# First 10 seconds for initial verification
TEST_SECONDS = 10


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
    transforms.Resize(
        SEGMENTATION_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

capture = cv2.VideoCapture(
    VIDEO_PATH
)

if not capture.isOpened():

    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )


video_fps = capture.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

max_frames = min(
    total_frames,
    int(video_fps * TEST_SECONDS)
)


print("\nVideo Information")
print("--------------------------------")

print("FPS:", video_fps)

print(
    "Total frames:",
    total_frames
)

print(
    "Frames to analyze:",
    max_frames
)


# --------------------------------------------------
# CSV HEADER
# --------------------------------------------------

fieldnames = [

    "frame",
    "timestamp_sec",

    "car_count",
    "pedestrian_count",
    "rider_count",
    "motorcycle_count",
    "bicycle_count",

    "truck_count",
    "bus_count",
    "heavy_vehicle_count",

    "total_risk_objects",

    "on_road_object_count",
    "on_road_pedestrian_count",

    "largest_object_area_ratio",

    "average_confidence",

    "drivable_area_ratio"
]


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

processed_frames = 0


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    while processed_frames < max_frames:

        success, frame = capture.read()

        if not success:
            break


        # --------------------------------------------------
        # RESIZE FRAME
        # --------------------------------------------------

        frame = cv2.resize(
            frame,
            (
                PROCESS_WIDTH,
                PROCESS_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        height, width = frame.shape[:2]

        frame_area = (
            width * height
        )


        # ==================================================
        # U-NET SEGMENTATION
        # ==================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_frame = Image.fromarray(
            rgb_frame
        )

        seg_input = segmentation_transform(
            pil_frame
        )

        seg_input = (
            seg_input
            .unsqueeze(0)
            .to(device)
        )


        with torch.no_grad():

            logits = unet(
                seg_input
            )

            probabilities = torch.sigmoid(
                logits
            )

            road_mask = (
                probabilities > 0.5
            ).float()


        road_mask = (
            road_mask
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )


        road_mask = cv2.resize(
            road_mask,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
        )


        # Percentage of frame predicted
        # as drivable road
        drivable_area_ratio = (
            road_mask.sum()
            / road_mask.size
        )


        # ==================================================
        # YOLO DETECTION
        # ==================================================

        results = yolo.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False
        )


        result = results[0]


        # ==================================================
        # FRAME STATISTICS
        # ==================================================

        counts = {
            "car": 0,
            "pedestrian": 0,
            "rider": 0,
            "motorcycle": 0,
            "bicycle": 0,
            "truck": 0,
            "bus": 0,
        }


        total_risk_objects = 0

        on_road_object_count = 0

        on_road_pedestrian_count = 0

        object_area_ratios = []

        confidences = []


        # ==================================================
        # ANALYZE DETECTIONS
        # ==================================================

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            class_name = yolo.names[
                class_id
            ]

            confidence = float(
                box.conf.item()
            )


            if class_name not in RISK_CLASSES:
                continue


            counts[class_name] += 1

            total_risk_objects += 1

            confidences.append(
                confidence
            )


            # --------------------------------------------------
            # BOUNDING BOX
            # --------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )


            x1 = max(
                0,
                int(x1)
            )

            y1 = max(
                0,
                int(y1)
            )

            x2 = min(
                width - 1,
                int(x2)
            )

            y2 = min(
                height - 1,
                int(y2)
            )


            # --------------------------------------------------
            # OBJECT SIZE
            # --------------------------------------------------

            object_width = max(
                0,
                x2 - x1
            )

            object_height = max(
                0,
                y2 - y1
            )

            object_area = (
                object_width
                * object_height
            )

            area_ratio = (
                object_area
                / frame_area
            )

            object_area_ratios.append(
                area_ratio
            )


            # --------------------------------------------------
            # GROUND-CONTACT POINT
            # --------------------------------------------------

            ground_x = int(
                (x1 + x2) / 2
            )

            ground_y = y2


            # --------------------------------------------------
            # ON-ROAD CHECK
            # --------------------------------------------------

            is_on_road = (
                road_mask[
                    ground_y,
                    ground_x
                ] == 1
            )


            if is_on_road:

                on_road_object_count += 1


                if class_name == "pedestrian":

                    on_road_pedestrian_count += 1


        # ==================================================
        # DERIVED FEATURES
        # ==================================================

        heavy_vehicle_count = (
            counts["truck"]
            + counts["bus"]
        )


        largest_object_area_ratio = (

            max(object_area_ratios)

            if object_area_ratios

            else 0.0
        )


        average_confidence = (

            sum(confidences)
            / len(confidences)

            if confidences

            else 0.0
        )


        timestamp_sec = (
            processed_frames
            / video_fps
        )


        # ==================================================
        # SAVE FRAME FEATURES
        # ==================================================

        writer.writerow({

            "frame":
                processed_frames,

            "timestamp_sec":
                round(timestamp_sec, 4),

            "car_count":
                counts["car"],

            "pedestrian_count":
                counts["pedestrian"],

            "rider_count":
                counts["rider"],

            "motorcycle_count":
                counts["motorcycle"],

            "bicycle_count":
                counts["bicycle"],

            "truck_count":
                counts["truck"],

            "bus_count":
                counts["bus"],

            "heavy_vehicle_count":
                heavy_vehicle_count,

            "total_risk_objects":
                total_risk_objects,

            "on_road_object_count":
                on_road_object_count,

            "on_road_pedestrian_count":
                on_road_pedestrian_count,

            "largest_object_area_ratio":
                round(
                    largest_object_area_ratio,
                    6
                ),

            "average_confidence":
                round(
                    average_confidence,
                    6
                ),

            "drivable_area_ratio":
                round(
                    float(drivable_area_ratio),
                    6
                )
        })


        processed_frames += 1


        if processed_frames % 25 == 0:

            print(
                f"Processed "
                f"{processed_frames}/"
                f"{max_frames} frames"
            )


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

capture.release()


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\nTemporal Feature Extraction Complete")
print("--------------------------------")

print(
    "Frames processed:",
    processed_frames
)

print(
    "Features per frame:",
    len(fieldnames) - 2
)

print(
    "CSV saved to:",
    CSV_PATH
)
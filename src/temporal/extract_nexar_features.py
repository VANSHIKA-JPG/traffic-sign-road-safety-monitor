import sys
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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

METADATA_PATH = Path(
    "data/nexar/experiment1_metadata.csv"
)

VIDEO_ROOT = Path(
    "data/nexar/experiment1"
)

YOLO_MODEL_PATH = (
    "runs/detect/outputs/yolo/"
    "bdd100k_yolo11n/weights/best.pt"
)

UNET_MODEL_PATH = (
    "outputs/models/unet_drivable_best.pth"
)

OUTPUT_ROOT = Path(
    "outputs/temporal/nexar"
)

POSITIVE_OUTPUT = OUTPUT_ROOT / "positive"
NEGATIVE_OUTPUT = OUTPUT_ROOT / "negative"

POSITIVE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

NEGATIVE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

SEGMENTATION_SIZE = (256, 256)

TARGET_FPS = 5

CONFIDENCE_THRESHOLD = 0.30


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD MODELS ONCE
# --------------------------------------------------

print("Loading YOLO...")

yolo = YOLO(
    YOLO_MODEL_PATH
)

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
# U-NET TRANSFORM
# --------------------------------------------------

segmentation_transform = transforms.Compose([

    transforms.Resize(
        SEGMENTATION_SIZE
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
# CSV COLUMNS
# --------------------------------------------------

fieldnames = [

    "frame",
    "timestamp_sec",

    "label",
    "time_of_alert",
    "time_of_event",

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
# LOAD EXPERIMENT METADATA
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "Experiment videos:",
    len(metadata)
)


# --------------------------------------------------
# PROCESS EACH VIDEO
# --------------------------------------------------

completed_videos = 0
failed_videos = 0

overall_start = time.perf_counter()


for video_index, row in metadata.iterrows():

    file_name = row["file_name"]

    category = row["category"]

    label = int(
        row["label"]
    )


    # --------------------------------------------------
    # VIDEO PATH
    # --------------------------------------------------

    video_path = (
        VIDEO_ROOT
        / category
        / file_name
    )


    # --------------------------------------------------
    # OUTPUT CSV
    # --------------------------------------------------

    if category == "positive":

        output_dir = POSITIVE_OUTPUT

    else:

        output_dir = NEGATIVE_OUTPUT


    csv_path = (
        output_dir
        / f"{Path(file_name).stem}.csv"
    )


    # --------------------------------------------------
    # RESTART SAFE
    # --------------------------------------------------

    if csv_path.exists():

        print(
            f"\n[{video_index + 1}/"
            f"{len(metadata)}] "
            f"Skipping completed: "
            f"{file_name}"
        )

        completed_videos += 1

        continue


    print(
        f"\n[{video_index + 1}/"
        f"{len(metadata)}] "
        f"Processing {category}: "
        f"{file_name}"
    )


    # --------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------

    capture = cv2.VideoCapture(
        str(video_path)
    )


    if not capture.isOpened():

        print(
            "FAILED TO OPEN:",
            video_path
        )

        failed_videos += 1

        continue


    source_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if source_fps <= 0:

        print(
            "Invalid FPS:",
            source_fps
        )

        capture.release()

        failed_videos += 1

        continue


    # Nexar is normally 30 FPS.
    # At TARGET_FPS = 5:
    #
    # 30 / 5 = every 6th frame

    frame_step = max(
        1,
        round(
            source_fps
            / TARGET_FPS
        )
    )


    print(
        "Source FPS:",
        source_fps
    )

    print(
        "Frame step:",
        frame_step
    )

    print(
        "Raw frames:",
        total_frames
    )


    # --------------------------------------------------
    # TEMPORAL METADATA
    # --------------------------------------------------

    time_of_alert = row[
        "time_of_alert"
    ]

    time_of_event = row[
        "time_of_event"
    ]


    video_start = time.perf_counter()

    raw_frame_index = 0

    sampled_frame_count = 0


    try:

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames
            )

            writer.writeheader()


            while True:

                success, frame = (
                    capture.read()
                )


                if not success:
                    break


                # --------------------------------------
                # ONLY PROCESS SELECTED FRAMES
                # --------------------------------------

                if (
                    raw_frame_index
                    % frame_step
                    != 0
                ):

                    raw_frame_index += 1

                    continue


                # --------------------------------------
                # TRUE VIDEO TIMESTAMP
                # --------------------------------------

                timestamp_sec = (
                    raw_frame_index
                    / source_fps
                )


                # --------------------------------------
                # RESIZE
                # --------------------------------------

                frame = cv2.resize(
                    frame,
                    (
                        PROCESS_WIDTH,
                        PROCESS_HEIGHT
                    ),
                    interpolation=cv2.INTER_AREA
                )


                height, width = (
                    frame.shape[:2]
                )

                frame_area = (
                    width * height
                )


                # ======================================
                # U-NET
                # ======================================

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                pil_frame = Image.fromarray(
                    rgb_frame
                )


                seg_input = (
                    segmentation_transform(
                        pil_frame
                    )
                    .unsqueeze(0)
                    .to(device)
                )


                with torch.no_grad():

                    logits = unet(
                        seg_input
                    )

                    probability = (
                        torch.sigmoid(
                            logits
                        )
                    )

                    road_mask = (
                        probability
                        > 0.5
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
                    interpolation=(
                        cv2.INTER_NEAREST
                    )
                )


                drivable_area_ratio = (
                    road_mask.sum()
                    / road_mask.size
                )


                # ======================================
                # YOLO
                # ======================================

                results = yolo.predict(

                    source=frame,

                    conf=(
                        CONFIDENCE_THRESHOLD
                    ),

                    device=0,

                    verbose=False
                )


                result = results[0]


                # ======================================
                # FEATURE STORAGE
                # ======================================

                counts = {

                    "car": 0,

                    "pedestrian": 0,

                    "rider": 0,

                    "motorcycle": 0,

                    "bicycle": 0,

                    "truck": 0,

                    "bus": 0
                }


                total_risk_objects = 0

                on_road_object_count = 0

                on_road_pedestrian_count = 0

                object_area_ratios = []

                confidences = []


                # ======================================
                # DETECTIONS
                # ======================================

                for box in result.boxes:

                    class_id = int(
                        box.cls.item()
                    )


                    class_name = (
                        yolo.names[
                            class_id
                        ]
                    )


                    confidence = float(
                        box.conf.item()
                    )


                    if (
                        class_name
                        not in RISK_CLASSES
                    ):

                        continue


                    counts[
                        class_name
                    ] += 1


                    total_risk_objects += 1


                    confidences.append(
                        confidence
                    )


                    x1, y1, x2, y2 = (
                        box.xyxy[
                            0
                        ].tolist()
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


                    # ----------------------------------
                    # OBJECT AREA
                    # ----------------------------------

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


                    # ----------------------------------
                    # GROUND POINT
                    # ----------------------------------

                    ground_x = int(
                        (x1 + x2)
                        / 2
                    )

                    ground_y = y2


                    # ----------------------------------
                    # ON ROAD?
                    # ----------------------------------

                    is_on_road = (

                        road_mask[
                            ground_y,
                            ground_x
                        ]
                        == 1
                    )


                    if is_on_road:

                        on_road_object_count += 1


                        if (
                            class_name
                            == "pedestrian"
                        ):

                            on_road_pedestrian_count += 1


                # ======================================
                # DERIVED FEATURES
                # ======================================

                heavy_vehicle_count = (

                    counts["truck"]

                    + counts["bus"]
                )


                largest_object_area_ratio = (

                    max(
                        object_area_ratios
                    )

                    if object_area_ratios

                    else 0.0
                )


                average_confidence = (

                    sum(
                        confidences
                    )
                    / len(
                        confidences
                    )

                    if confidences

                    else 0.0
                )


                # ======================================
                # WRITE ROW
                # ======================================

                writer.writerow({

                    "frame":
                        raw_frame_index,

                    "timestamp_sec":
                        round(
                            timestamp_sec,
                            4
                        ),

                    "label":
                        label,

                    "time_of_alert":
                        (
                            round(
                                float(
                                    time_of_alert
                                ),
                                4
                            )
                            if pd.notna(
                                time_of_alert
                            )
                            else ""
                        ),

                    "time_of_event":
                        (
                            round(
                                float(
                                    time_of_event
                                ),
                                4
                            )
                            if pd.notna(
                                time_of_event
                            )
                            else ""
                        ),

                    "car_count":
                        counts["car"],

                    "pedestrian_count":
                        counts[
                            "pedestrian"
                        ],

                    "rider_count":
                        counts["rider"],

                    "motorcycle_count":
                        counts[
                            "motorcycle"
                        ],

                    "bicycle_count":
                        counts[
                            "bicycle"
                        ],

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
                            float(
                                drivable_area_ratio
                            ),
                            6
                        )
                })


                sampled_frame_count += 1


                if (
                    sampled_frame_count
                    % 50
                    == 0
                ):

                    print(
                        "Sampled frames:",
                        sampled_frame_count
                    )


                raw_frame_index += 1


    except Exception as error:

        print(
            "FAILED DURING PROCESSING:",
            file_name
        )

        print(
            "Reason:",
            error
        )

        # Remove incomplete CSV so that
        # rerunning will process it again.

        if csv_path.exists():

            csv_path.unlink()


        failed_videos += 1

        capture.release()

        continue


    finally:

        capture.release()


    video_time = (
        time.perf_counter()
        - video_start
    )


    completed_videos += 1


    print(
        "Completed:",
        file_name
    )

    print(
        "Sampled frames:",
        sampled_frame_count
    )

    print(
        "Processing time:",
        f"{video_time:.2f} sec"
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

overall_time = (
    time.perf_counter()
    - overall_start
)


print(
    "\n================================"
)

print(
    "NEXAR FEATURE EXTRACTION COMPLETE"
)

print(
    "================================"
)

print(
    "Completed videos:",
    completed_videos
)

print(
    "Failed videos:",
    failed_videos
)

print(
    "Total time:",
    f"{overall_time / 60:.2f} minutes"
)

print(
    "Output folder:",
    OUTPUT_ROOT
)
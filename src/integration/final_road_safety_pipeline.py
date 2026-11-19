import sys
import time
from collections import deque
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
from models.lstm_model import RiskLSTM
from integration.risk_engine import RiskEngine


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

LSTM_MODEL_PATH = (
    "outputs/models/risk_lstm_best.pth"
)

NORMALIZATION_PATH = (
    "data/nexar/lstm_dataset/normalization.pt"
)

OUTPUT_DIR = Path(
    "outputs/final_pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "road_safety_final.mp4"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

SEGMENTATION_SIZE = (
    256,
    256
)

CONFIDENCE_THRESHOLD = 0.30

# LSTM was trained using:
# 5 observations / second
TARGET_TEMPORAL_FPS = 5

SEQUENCE_LENGTH = 15


RISK_CLASSES = {
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


FEATURE_COLUMNS = [
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
    "drivable_area_ratio",
]


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# --------------------------------------------------
# LOAD YOLO
# --------------------------------------------------

print(
    "Loading YOLO..."
)

yolo = YOLO(
    YOLO_MODEL_PATH
)

print(
    "YOLO loaded."
)


# --------------------------------------------------
# LOAD U-NET
# --------------------------------------------------

print(
    "Loading U-Net..."
)

unet = UNet().to(
    device
)

unet.load_state_dict(
    torch.load(
        UNET_MODEL_PATH,
        map_location=device
    )
)

unet.eval()

print(
    "U-Net loaded."
)


# --------------------------------------------------
# LOAD LSTM
# --------------------------------------------------

print(
    "Loading LSTM..."
)

lstm_checkpoint = torch.load(
    LSTM_MODEL_PATH,
    map_location=device
)


lstm = RiskLSTM(
    input_size=lstm_checkpoint[
        "input_size"
    ],
    hidden_size=lstm_checkpoint[
        "hidden_size"
    ],
    num_layers=lstm_checkpoint[
        "num_layers"
    ],
    num_classes=lstm_checkpoint[
        "num_classes"
    ],
    dropout=0.3
).to(device)


lstm.load_state_dict(
    lstm_checkpoint[
        "model_state_dict"
    ]
)

lstm.eval()


print(
    "LSTM loaded."
)

print(
    "LSTM checkpoint epoch:",
    lstm_checkpoint["epoch"]
)


# --------------------------------------------------
# LOAD NORMALIZATION
# --------------------------------------------------

normalization_data = torch.load(
    NORMALIZATION_PATH,
    map_location="cpu"
)


feature_mean = (
    normalization_data["mean"]
    .numpy()
)

feature_std = (
    normalization_data["std"]
    .numpy()
)


print(
    "Normalization loaded."
)


# --------------------------------------------------
# RISK ENGINE
# --------------------------------------------------

risk_engine = RiskEngine(
    medium_threshold=0.35,
    high_threshold=0.65
)


# --------------------------------------------------
# SEGMENTATION TRANSFORM
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
# OPEN VIDEO
# --------------------------------------------------

capture = cv2.VideoCapture(
    VIDEO_PATH
)


if not capture.isOpened():

    raise FileNotFoundError(
        f"Could not open video: "
        f"{VIDEO_PATH}"
    )


source_fps = capture.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


if source_fps <= 0:

    capture.release()

    raise RuntimeError(
        "Invalid input video FPS."
    )


# We only update temporal features
# at approximately 5 FPS.

temporal_frame_step = max(
    1,
    round(
        source_fps
        / TARGET_TEMPORAL_FPS
    )
)


print("\nInput Video")
print("--------------------------------")

print(
    "FPS:",
    source_fps
)

print(
    "Frames:",
    total_frames
)

print(
    "Temporal frame step:",
    temporal_frame_step
)


# --------------------------------------------------
# VIDEO WRITER
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    source_fps,
    (
        PROCESS_WIDTH,
        PROCESS_HEIGHT
    )
)


if not writer.isOpened():

    capture.release()

    raise RuntimeError(
        "Could not create output video."
    )


# --------------------------------------------------
# TEMPORAL BUFFER
# --------------------------------------------------

feature_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# --------------------------------------------------
# CURRENT RISK STATE
# --------------------------------------------------

current_risk_probability = 0.0

current_risk_result = {
    "level": "BUILDING",
    "risk_probability": 0.0,
    "risk_percent": 0.0,
    "reasons": [
        "Building temporal context"
    ]
}


# --------------------------------------------------
# PROCESSING
# --------------------------------------------------

frame_index = 0
processed_frames = 0

start_time = time.perf_counter()


try:

    while True:

        success, frame = capture.read()


        if not success:
            break


        frame_start = time.perf_counter()


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


        height, width = (
            frame.shape[:2]
        )

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


        seg_input = (
            segmentation_transform(
                pil_frame
            )
            .unsqueeze(0)
            .to(device)
        )


        with torch.no_grad():

            segmentation_logits = unet(
                seg_input
            )

            segmentation_probability = (
                torch.sigmoid(
                    segmentation_logits
                )
            )

            road_mask = (
                segmentation_probability
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
            interpolation=cv2.INTER_NEAREST
        )


        drivable_area_ratio = (
            road_mask.sum()
            / road_mask.size
        )


        # ==================================================
        # YOLO
        # ==================================================

        results = yolo.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False
        )


        result = results[0]


        # ==================================================
        # FEATURE STORAGE
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
        # ROAD OVERLAY
        # ==================================================

        road_layer = np.zeros_like(
            frame
        )


        road_layer[
            road_mask == 1
        ] = (
            255,
            0,
            0
        )


        output_frame = cv2.addWeighted(
            frame,
            1.0,
            road_layer,
            0.20,
            0
        )


        # ==================================================
        # DETECTIONS
        # ==================================================

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
            # OBJECT AREA
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
            # GROUND POINT
            # --------------------------------------------------

            ground_x = int(
                (x1 + x2)
                / 2
            )

            ground_y = y2


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


                box_color = (
                    0,
                    0,
                    255
                )

                status = (
                    "ON ROAD"
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

                status = (
                    "OFF ROAD"
                )


            # --------------------------------------------------
            # DRAW BOX
            # --------------------------------------------------

            cv2.rectangle(
                output_frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                box_color,
                2
            )


            label_text = (
                f"{class_name} "
                f"{confidence:.2f} "
                f"{status}"
            )


            cv2.putText(
                output_frame,
                label_text,
                (
                    x1,
                    max(
                        25,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                box_color,
                2
            )


        # ==================================================
        # DERIVED FEATURES
        # ==================================================

        heavy_vehicle_count = (
            counts["truck"]
            +
            counts["bus"]
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


        latest_features = {

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
                largest_object_area_ratio,

            "average_confidence":
                average_confidence,

            "drivable_area_ratio":
                float(
                    drivable_area_ratio
                )
        }


        # ==================================================
        # TEMPORAL UPDATE
        #
        # Only update LSTM feature history
        # at ~5 FPS.
        # ==================================================

        if (
            frame_index
            % temporal_frame_step
            == 0
        ):

            feature_vector = np.array(
                [
                    latest_features[
                        column
                    ]
                    for column
                    in FEATURE_COLUMNS
                ],
                dtype=np.float32
            )


            feature_buffer.append(
                feature_vector
            )


            # --------------------------------------------------
            # RUN LSTM WHEN 15 OBSERVATIONS EXIST
            # --------------------------------------------------

            if (
                len(feature_buffer)
                == SEQUENCE_LENGTH
            ):

                sequence = np.array(
                    feature_buffer,
                    dtype=np.float32
                )


                # ----------------------------------------------
                # NORMALIZE USING V1 TRAINING STATISTICS
                # ----------------------------------------------

                sequence = (
                    sequence
                    - feature_mean
                ) / feature_std


                sequence_tensor = (
                    torch.tensor(
                        sequence,
                        dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to(device)
                )


                with torch.no_grad():

                    lstm_logits = lstm(
                        sequence_tensor
                    )


                    probabilities = (
                        torch.softmax(
                            lstm_logits,
                            dim=1
                        )
                    )


                    current_risk_probability = float(
                        probabilities[
                            0,
                            1
                        ].item()
                    )


                # ----------------------------------------------
                # RISK ENGINE
                # ----------------------------------------------

                current_risk_result = (
                    risk_engine.assess(
                        risk_probability=(
                            current_risk_probability
                        ),
                        features=(
                            latest_features
                        )
                    )
                )


        # ==================================================
        # DRAW SYSTEM PANEL
        # ==================================================

        panel_x1 = 20
        panel_y1 = 20

        panel_x2 = 500
        panel_y2 = 235


        overlay = (
            output_frame.copy()
        )


        cv2.rectangle(
            overlay,
            (
                panel_x1,
                panel_y1
            ),
            (
                panel_x2,
                panel_y2
            ),
            (
                20,
                20,
                20
            ),
            -1
        )


        output_frame = (
            cv2.addWeighted(
                overlay,
                0.65,
                output_frame,
                0.35,
                0
            )
        )


        # --------------------------------------------------
        # RISK COLOR
        # --------------------------------------------------

        risk_level = (
            current_risk_result[
                "level"
            ]
        )


        if risk_level == "HIGH":

            risk_color = (
                0,
                0,
                255
            )

        elif risk_level == "MEDIUM":

            risk_color = (
                0,
                165,
                255
            )

        elif risk_level == "LOW":

            risk_color = (
                0,
                255,
                0
            )

        else:

            risk_color = (
                255,
                255,
                255
            )


        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        cv2.putText(
            output_frame,
            "ROAD SAFETY MONITOR",
            (
                40,
                55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (
                255,
                255,
                255
            ),
            2
        )


        # --------------------------------------------------
        # RISK LEVEL
        # --------------------------------------------------

        cv2.putText(
            output_frame,
            f"RISK: {risk_level}",
            (
                40,
                95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            risk_color,
            2
        )


        # --------------------------------------------------
        # RISK PROBABILITY
        # --------------------------------------------------

        if risk_level == "BUILDING":

            probability_text = (
                "Temporal context: "
                f"{len(feature_buffer)}/"
                f"{SEQUENCE_LENGTH}"
            )

        else:

            probability_text = (
                f"Risk probability: "
                f"{current_risk_result['risk_percent']:.1f}%"
            )


        cv2.putText(
            output_frame,
            probability_text,
            (
                40,
                130
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                255,
                255,
                255
            ),
            2
        )


        # --------------------------------------------------
        # SCENE COUNTS
        # --------------------------------------------------

        scene_text = (
            f"Cars: "
            f"{counts['car']}  "
            f"Pedestrians: "
            f"{counts['pedestrian']}  "
            f"On-road: "
            f"{on_road_object_count}"
        )


        cv2.putText(
            output_frame,
            scene_text,
            (
                40,
                165
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                255,
                255,
                255
            ),
            2
        )


        # --------------------------------------------------
        # FIRST EXPLANATION
        # --------------------------------------------------

        reasons = (
            current_risk_result[
                "reasons"
            ]
        )


        if reasons:

            reason_text = (
                "Why: "
                + reasons[0]
            )


            cv2.putText(
                output_frame,
                reason_text[:65],
                (
                    40,
                    200
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (
                    255,
                    255,
                    255
                ),
                1
            )


        # ==================================================
        # PERFORMANCE
        # ==================================================

        frame_time = (
            time.perf_counter()
            - frame_start
        )


        processing_fps = (

            1.0
            / frame_time

            if frame_time > 0

            else 0.0
        )


        cv2.putText(
            output_frame,
            f"Processing: "
            f"{processing_fps:.1f} FPS",
            (
                40,
                height - 35
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (
                255,
                255,
                255
            ),
            2
        )


        # ==================================================
        # WRITE FRAME
        # ==================================================

        writer.write(
            output_frame
        )


        processed_frames += 1
        frame_index += 1


        if (
            processed_frames
            % 50
            == 0
        ):

            print(
                f"Processed "
                f"{processed_frames}/"
                f"{total_frames} frames "
                f"| Risk: "
                f"{risk_level}"
            )


finally:

    capture.release()

    writer.release()

    print(
        "\nVideo resources released."
    )


# --------------------------------------------------
# PERFORMANCE SUMMARY
# --------------------------------------------------

total_time = (
    time.perf_counter()
    - start_time
)


average_fps = (

    processed_frames
    / total_time

    if total_time > 0

    else 0.0
)


average_latency = (

    total_time
    / processed_frames
    * 1000

    if processed_frames > 0

    else 0.0
)


print("\n================================")
print("FINAL PIPELINE COMPLETE")
print("================================")

print(
    "Processed frames:",
    processed_frames
)

print(
    "Total time:",
    f"{total_time:.2f} sec"
)

print(
    "Average FPS:",
    f"{average_fps:.2f}"
)

print(
    "Average latency:",
    f"{average_latency:.2f} ms/frame"
)

print(
    "Final risk level:",
    current_risk_result["level"]
)

print(
    "Final risk probability:",
    f"{current_risk_result['risk_percent']:.2f}%"
)

print(
    "Output video:",
    OUTPUT_PATH
)
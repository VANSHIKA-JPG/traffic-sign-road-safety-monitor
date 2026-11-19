import sys
import time
from collections import deque
from pathlib import Path
import subprocess
import imageio_ffmpeg
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# ============================================================
# PROJECT IMPORTS
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.segmentation_model import UNet
from models.lstm_model import RiskLSTM
from integration.risk_engine import RiskEngine


# ============================================================
# MODEL PATHS
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720

SEGMENTATION_SIZE = (256, 256)

CONFIDENCE_THRESHOLD = 0.30

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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEGMENTATION TRANSFORM
# ============================================================

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


# ============================================================
# MODEL CACHE
# ============================================================

_MODELS = None


# ============================================================
# LOAD MODELS ONCE
# ============================================================

def load_models():

    global _MODELS

    if _MODELS is not None:
        return _MODELS


    print(
        "Loading Road Safety models..."
    )

    print(
        "Device:",
        device
    )


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    yolo = YOLO(
        YOLO_MODEL_PATH
    )


    # --------------------------------------------------------
    # U-NET
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LSTM V1
    # --------------------------------------------------------

    checkpoint = torch.load(
        LSTM_MODEL_PATH,
        map_location=device
    )


    lstm = RiskLSTM(
        input_size=checkpoint[
            "input_size"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
        num_classes=checkpoint[
            "num_classes"
        ],
        dropout=0.3
    ).to(device)


    lstm.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    lstm.eval()


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    normalization = torch.load(
        NORMALIZATION_PATH,
        map_location="cpu"
    )


    feature_mean = (
        normalization["mean"]
        .numpy()
    )


    feature_std = (
        normalization["std"]
        .numpy()
    )


    # --------------------------------------------------------
    # RISK ENGINE
    # --------------------------------------------------------

    risk_engine = RiskEngine(
        medium_threshold=0.35,
        high_threshold=0.65
    )


    _MODELS = {
        "yolo": yolo,
        "unet": unet,
        "lstm": lstm,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "risk_engine": risk_engine,
        "lstm_epoch": checkpoint["epoch"],
    }


    print(
        "All models loaded."
    )


    return _MODELS
def convert_to_browser_video(input_path, output_path):

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),

        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",

        "-pix_fmt", "yuv420p",

        "-movflags", "+faststart",

        "-an",

        str(output_path)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return output_path

# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(
    input_path,
    output_path,
    progress_callback=None
):

    models = load_models()

    yolo = models["yolo"]
    unet = models["unet"]
    lstm = models["lstm"]

    feature_mean = models[
        "feature_mean"
    ]

    feature_std = models[
        "feature_std"
    ]

    risk_engine = models[
        "risk_engine"
    ]


    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # OPEN INPUT
    # ========================================================

    capture = cv2.VideoCapture(
        str(input_path)
    )


    if not capture.isOpened():

        raise FileNotFoundError(
            f"Could not open video: "
            f"{input_path}"
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


    temporal_frame_step = max(
        1,
        round(
            source_fps
            / TARGET_TEMPORAL_FPS
        )
    )


    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(output_path),
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


    # ========================================================
    # TEMPORAL STATE
    # ========================================================

    feature_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )


    risk_history = []


    current_risk_result = {
        "level": "BUILDING",
        "risk_probability": 0.0,
        "risk_percent": 0.0,
        "reasons": [
            "Building temporal context"
        ]
    }


    # Keep final scene information
    latest_scene = {
        "vehicles": 0,
        "pedestrians": 0,
        "on_road_objects": 0,
        "heavy_vehicles": 0,
    }


    # ========================================================
    # PROCESSING
    # ========================================================

    frame_index = 0

    processed_frames = 0

    start_time = (
        time.perf_counter()
    )


    try:

        while True:

            success, frame = (
                capture.read()
            )


            if not success:
                break


            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

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


            # =================================================
            # U-NET
            # =================================================

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

                seg_logits = unet(
                    seg_input
                )


                road_probability = (
                    torch.sigmoid(
                        seg_logits
                    )
                )


                road_mask = (
                    road_probability
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


            # =================================================
            # YOLO
            # =================================================

            results = yolo.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                device=0,
                verbose=False
            )


            result = results[0]


            # =================================================
            # FEATURE STORAGE
            # =================================================

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


            # =================================================
            # ROAD OVERLAY
            # =================================================

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


            output_frame = (
                cv2.addWeighted(
                    frame,
                    1.0,
                    road_layer,
                    0.20,
                    0
                )
            )


            # =================================================
            # DETECTIONS
            # =================================================

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


                # ---------------------------------------------
                # OBJECT AREA
                # ---------------------------------------------

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


                # ---------------------------------------------
                # GROUND POINT
                # ---------------------------------------------

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


                # ---------------------------------------------
                # DRAW DETECTION
                # ---------------------------------------------

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


            # =================================================
            # DERIVED FEATURES
            # =================================================

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


            # =================================================
            # SCENE INFORMATION
            # =================================================

            latest_scene = {

                "vehicles":
                    (
                        counts["car"]
                        + counts["truck"]
                        + counts["bus"]
                        + counts[
                            "motorcycle"
                        ]
                        + counts[
                            "bicycle"
                        ]
                    ),

                "pedestrians":
                    counts[
                        "pedestrian"
                    ],

                "on_road_objects":
                    on_road_object_count,

                "heavy_vehicles":
                    heavy_vehicle_count,
            }


            # =================================================
            # TEMPORAL UPDATE
            # =================================================

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


                # ---------------------------------------------
                # LSTM
                # ---------------------------------------------

                if (
                    len(feature_buffer)
                    == SEQUENCE_LENGTH
                ):

                    sequence = np.array(
                        feature_buffer,
                        dtype=np.float32
                    )


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

                        logits = lstm(
                            sequence_tensor
                        )


                        probabilities = (
                            torch.softmax(
                                logits,
                                dim=1
                            )
                        )


                        risk_probability = float(
                            probabilities[
                                0,
                                1
                            ].item()
                        )


                    # -----------------------------------------
                    # RISK ENGINE
                    # -----------------------------------------

                    current_risk_result = (
                        risk_engine.assess(
                            risk_probability=(
                                risk_probability
                            ),
                            features=(
                                latest_features
                            )
                        )
                    )


                    # -----------------------------------------
                    # RISK HISTORY
                    # -----------------------------------------

                    timestamp_sec = (
                        frame_index
                        / source_fps
                    )


                    risk_history.append(
                        {
                            "time":
                                round(
                                    timestamp_sec,
                                    3
                                ),

                            "risk":
                                risk_probability
                        }
                    )


            # =================================================
            # SMALL VIDEO OVERLAY
            # =================================================

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


            cv2.rectangle(
                output_frame,
                (20, 20),
                (430, 125),
                (20, 20, 20),
                -1
            )


            cv2.putText(
                output_frame,
                "ROAD INTELLIGENCE",
                (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            cv2.putText(
                output_frame,
                f"RISK: {risk_level}",
                (40, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                risk_color,
                2
            )


            # =================================================
            # WRITE FRAME
            # =================================================

            writer.write(
                output_frame
            )


            processed_frames += 1

            frame_index += 1


            # =================================================
            # PROGRESS CALLBACK
            # =================================================

            if (
                progress_callback
                is not None
                and total_frames > 0
            ):

                progress_callback(
                    min(
                        processed_frames
                        / total_frames,
                        1.0
                    )
                )


    finally:

        capture.release()

        writer.release()


    # ========================================================
    # PERFORMANCE
    # ========================================================

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


    latency_ms = (

        total_time
        / processed_frames
        * 1000

        if processed_frames > 0

        else 0.0
    )
    # ========================================================
    # BROWSER-COMPATIBLE VIDEO
    # ========================================================

    browser_output_path = (
        output_path.parent
        / f"{output_path.stem}_web.mp4"
    )

    print(
        "Converting output video to browser-compatible H.264..."
    )

    convert_to_browser_video(
        output_path,
        browser_output_path
    )

    print(
        "Browser video created:",
        browser_output_path
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "output_video":
            str(output_path),

        "browser_output_video":
            str(browser_output_path),

        "risk_level":
            current_risk_result[
                "level"
            ],

        "risk_probability":
            current_risk_result[
                "risk_probability"
            ],

        "risk_percent":
            current_risk_result[
                "risk_percent"
            ],

        "risk_history":
            risk_history,

        "scene":
            latest_scene,

        "reasons":
            current_risk_result[
                "reasons"
            ],

        "processed_frames":
            processed_frames,

        "fps":
            average_fps,

        "latency_ms":
            latency_ms,

        "device":
            str(device),

        "lstm_epoch":
            models[
                "lstm_epoch"
            ],
    }


    return result
import sys
import time
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

OUTPUT_DIR = Path("outputs/video")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "road_test_10sec_720p.mp4"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

TEST_SECONDS = 10

SEGMENTATION_SIZE = (256, 256)

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
# LOAD YOLO
# --------------------------------------------------

print("Loading YOLO...")

yolo = YOLO(
    YOLO_MODEL_PATH
)

print("YOLO loaded.")


# --------------------------------------------------
# LOAD U-NET
# --------------------------------------------------

print("Loading U-Net...")

unet = UNet().to(device)

unet.load_state_dict(
    torch.load(
        UNET_MODEL_PATH,
        map_location=device
    )
)

unet.eval()

print("U-Net loaded.")


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
# OPEN INPUT VIDEO
# --------------------------------------------------

capture = cv2.VideoCapture(
    VIDEO_PATH
)

if not capture.isOpened():

    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )


input_fps = capture.get(
    cv2.CAP_PROP_FPS
)

input_width = int(
    capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

input_height = int(
    capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_video_frames = int(
    capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


MAX_FRAMES = int(
    input_fps * TEST_SECONDS
)


print("\nInput Video")
print("--------------------------------")

print(
    "Original resolution:",
    input_width,
    "x",
    input_height
)

print(
    "Input FPS:",
    input_fps
)

print(
    "Total video frames:",
    total_video_frames
)

print(
    "Test duration:",
    TEST_SECONDS,
    "seconds"
)

print(
    "Frames to process:",
    MAX_FRAMES
)

print(
    "Output resolution:",
    OUTPUT_WIDTH,
    "x",
    OUTPUT_HEIGHT
)


# --------------------------------------------------
# CREATE OUTPUT VIDEO
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    input_fps,
    (
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT
    )
)


if not writer.isOpened():

    capture.release()

    raise RuntimeError(
        "Could not create output video."
    )


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

processed_frames = 0

start_time = time.perf_counter()


try:

    while True:

        # Stop after 10 seconds
        if processed_frames >= MAX_FRAMES:
            break


        success, frame = capture.read()


        if not success:
            break


        frame_start = time.perf_counter()


        # ------------------------------------------
        # RESIZE 4K FRAME → 720P
        # ------------------------------------------

        frame = cv2.resize(
            frame,
            (
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )


        height, width = frame.shape[:2]


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

            probability = torch.sigmoid(
                logits
            )

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


        # Resize segmentation mask back
        # to 1280 x 720
        road_mask = cv2.resize(
            road_mask,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
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
            0.25,

            0
        )


        # ==================================================
        # OBJECT DETECTION + ROAD ANALYSIS
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


            # Ignore classes that are not
            # currently used for spatial risk
            if class_name not in RISK_CLASSES:
                continue


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


            # ------------------------------------------
            # GROUND-CONTACT POINT
            # ------------------------------------------

            ground_x = int(
                (x1 + x2) / 2
            )

            ground_y = y2


            # ------------------------------------------
            # CHECK WHETHER OBJECT IS ON ROAD
            # ------------------------------------------

            is_on_road = (

                road_mask[
                    ground_y,
                    ground_x
                ] == 1
            )


            if is_on_road:

                status = "ON ROAD"

                # Red
                color = (
                    0,
                    0,
                    255
                )

            else:

                status = "OFF ROAD"

                # Green
                color = (
                    0,
                    255,
                    0
                )


            # ------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------

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

                color,

                2
            )


            # ------------------------------------------
            # LABEL
            # ------------------------------------------

            label = (

                f"{class_name} "

                f"{confidence:.2f} "

                f"{status}"
            )


            cv2.putText(

                output_frame,

                label,

                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2
            )


            # ------------------------------------------
            # DRAW GROUND POINT
            # ------------------------------------------

            cv2.circle(

                output_frame,

                (
                    ground_x,
                    ground_y
                ),

                5,

                color,

                -1
            )


        # ==================================================
        # FRAME PERFORMANCE
        # ==================================================

        frame_time = (

            time.perf_counter()
            - frame_start
        )


        processing_fps = (

            1.0 / frame_time

            if frame_time > 0

            else 0.0
        )


        latency_ms = (
            frame_time * 1000
        )


        # ------------------------------------------
        # DISPLAY FPS
        # ------------------------------------------

        cv2.putText(

            output_frame,

            (
                f"Processing FPS: "
                f"{processing_fps:.1f}"
            ),

            (
                30,
                45
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (
                255,
                255,
                255
            ),

            2
        )


        # ------------------------------------------
        # DISPLAY LATENCY
        # ------------------------------------------

        cv2.putText(

            output_frame,

            (
                f"Latency: "
                f"{latency_ms:.1f} ms"
            ),

            (
                30,
                85
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (
                255,
                255,
                255
            ),

            2
        )


        # ==================================================
        # WRITE OUTPUT FRAME
        # ==================================================

        writer.write(
            output_frame
        )


        processed_frames += 1


        # ------------------------------------------
        # PROGRESS
        # ------------------------------------------

        if processed_frames % 25 == 0:

            print(

                f"Processed "
                f"{processed_frames}/"
                f"{MAX_FRAMES} frames"
            )


finally:

    # --------------------------------------------------
    # ALWAYS RELEASE VIDEO FILES
    # --------------------------------------------------

    capture.release()

    writer.release()

    print(
        "\nVideo resources released."
    )


# --------------------------------------------------
# PERFORMANCE RESULTS
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


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print(
    "\nVideo Processing Complete"
)

print(
    "--------------------------------"
)

print(
    "Processed frames:",
    processed_frames
)

print(
    "Total processing time:",
    f"{total_time:.2f} seconds"
)

print(
    "Average processing FPS:",
    f"{average_fps:.2f}"
)

print(
    "Average latency:",
    f"{average_latency:.2f} ms/frame"
)

print(
    "Output video:",
    OUTPUT_PATH
)
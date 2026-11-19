from pathlib import Path

import cv2
import pandas as pd


METADATA_PATH = Path(
    "data/nexar/experiment1_metadata.csv"
)

VIDEO_ROOT = Path(
    "data/nexar/experiment1"
)


# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)


positive = metadata[
    metadata["label"] == 1
].iloc[0]

negative = metadata[
    metadata["label"] == 0
].iloc[0]


# --------------------------------------------------
# INSPECTION FUNCTION
# --------------------------------------------------

def inspect_video(row):

    category = row["category"]
    file_name = row["file_name"]

    video_path = (
        VIDEO_ROOT
        / category
        / file_name
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        print(
            "Could not open:",
            video_path
        )

        return


    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        frames / fps
        if fps > 0
        else 0
    )


    print("\n--------------------------------")
    print("Video:", file_name)
    print("Category:", category)
    print("Label:", row["label"])

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "FPS:",
        fps
    )

    print(
        "Frames:",
        frames
    )

    print(
        "Duration:",
        f"{duration:.2f} sec"
    )


    if row["label"] == 1:

        alert_time = row[
            "time_of_alert"
        ]

        event_time = row[
            "time_of_event"
        ]

        warning_time = (
            event_time
            - alert_time
        )

        print(
            "Alert time:",
            f"{alert_time:.3f} sec"
        )

        print(
            "Event time:",
            f"{event_time:.3f} sec"
        )

        print(
            "Alert → Event:",
            f"{warning_time:.3f} sec"
        )


    print(
        "Scene:",
        row["scene"]
    )

    print(
        "Weather:",
        row["weather"]
    )

    print(
        "Lighting:",
        row["light_conditions"]
    )


    capture.release()


# --------------------------------------------------
# INSPECT ONE OF EACH
# --------------------------------------------------

print("POSITIVE SAMPLE")
inspect_video(positive)

print("\nNEGATIVE SAMPLE")
inspect_video(negative)
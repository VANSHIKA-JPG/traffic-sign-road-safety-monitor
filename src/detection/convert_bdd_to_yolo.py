import json
import random
import shutil
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE_ROOT = Path("data/bdd100k")
ANNOTATION_FILE = SOURCE_ROOT / "samples.json"

OUTPUT_ROOT = Path("data/bdd100k_yolo")

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# Classes used by our Road Safety Monitor
CLASS_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "motorcycle": 5,
    "bicycle": 6,
    "traffic light": 7,
    "traffic sign": 8,
}


# Defensive cleanup for inconsistent labels
LABEL_ALIASES = {
    "trafficsign": "traffic sign",
    "trafficlight": "traffic light",
}


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

for split in ["train", "val"]:

    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# LOAD BDD100K ANNOTATIONS
# --------------------------------------------------

print("Loading BDD100K annotations...")

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


samples = data["samples"]

print("Total samples:", len(samples))


# --------------------------------------------------
# REPRODUCIBLE TRAIN / VALIDATION SPLIT
# --------------------------------------------------

random.seed(RANDOM_SEED)

indices = list(range(len(samples)))

random.shuffle(indices)


train_count = int(
    len(indices) * TRAIN_RATIO
)

train_indices = set(
    indices[:train_count]
)


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

converted_images = {
    "train": 0,
    "val": 0
}

converted_objects = {
    "train": 0,
    "val": 0
}

skipped_objects = 0


# --------------------------------------------------
# CONVERT DATASET
# --------------------------------------------------

for index, sample in enumerate(samples):

    split = (
        "train"
        if index in train_indices
        else "val"
    )


    # --------------------------------------------------
    # SOURCE IMAGE
    # --------------------------------------------------

    relative_image_path = Path(
        sample["filepath"]
    )

    source_image = (
        SOURCE_ROOT
        / relative_image_path
    )


    if not source_image.exists():

        print(
            "Missing image:",
            source_image
        )

        continue


    # --------------------------------------------------
    # DESTINATION IMAGE
    # --------------------------------------------------

    destination_image = (
        OUTPUT_ROOT
        / "images"
        / split
        / source_image.name
    )


    shutil.copy2(
        source_image,
        destination_image
    )


    # --------------------------------------------------
    # YOLO LABEL FILE
    # --------------------------------------------------

    label_path = (
        OUTPUT_ROOT
        / "labels"
        / split
        / f"{source_image.stem}.txt"
    )


    yolo_lines = []


    detection_data = sample.get(
        "detections"
    )


    if detection_data:

        detections = detection_data.get(
            "detections",
            []
        )


        for detection in detections:

            label = detection[
                "label"
            ].strip().lower()


            # Clean aliases
            label = LABEL_ALIASES.get(
                label,
                label
            )


            # Skip classes not used
            if label not in CLASS_MAP:

                skipped_objects += 1
                continue


            class_id = CLASS_MAP[
                label
            ]


            # FiftyOne format:
            # x_top_left,
            # y_top_left,
            # width,
            # height

            x, y, width, height = detection[
                "bounding_box"
            ]


            # Convert to YOLO:
            # x_center,
            # y_center,
            # width,
            # height

            x_center = (
                x + width / 2
            )

            y_center = (
                y + height / 2
            )


            # Safety clamp
            x_center = max(
                0.0,
                min(1.0, x_center)
            )

            y_center = max(
                0.0,
                min(1.0, y_center)
            )

            width = max(
                0.0,
                min(1.0, width)
            )

            height = max(
                0.0,
                min(1.0, height)
            )


            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )


    # Create label file even if image
    # contains no selected objects
    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as label_file:

        label_file.write(
            "\n".join(yolo_lines)
        )


    converted_images[split] += 1

    converted_objects[split] += len(
        yolo_lines
    )


# --------------------------------------------------
# CREATE YOLO YAML
# --------------------------------------------------

yaml_path = (
    OUTPUT_ROOT
    / "bdd100k.yaml"
)


yaml_content = f"""path: {OUTPUT_ROOT.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: motorcycle
  6: bicycle
  7: traffic light
  8: traffic sign
"""


with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as yaml_file:

    yaml_file.write(
        yaml_content
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nConversion complete.")
print("--------------------------------")

print(
    "Training images:",
    converted_images["train"]
)

print(
    "Validation images:",
    converted_images["val"]
)

print(
    "Training objects:",
    converted_objects["train"]
)

print(
    "Validation objects:",
    converted_objects["val"]
)

print(
    "Skipped objects:",
    skipped_objects
)

print(
    "\nDataset YAML:",
    yaml_path
)
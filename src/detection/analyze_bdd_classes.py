import json
from collections import Counter


ANNOTATION_PATH = "data/bdd100k/samples.json"


# --------------------------------------------------
# LOAD ANNOTATIONS
# --------------------------------------------------

print("Loading BDD100K annotations...")

with open(
    ANNOTATION_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


samples = data["samples"]

print("Total images:", len(samples))


# --------------------------------------------------
# COUNT OBJECT CLASSES
# --------------------------------------------------

class_counts = Counter()

total_objects = 0


for sample in samples:

    detection_data = sample.get("detections")

    if not detection_data:
        continue

    detections = detection_data.get(
        "detections",
        []
    )

    for detection in detections:

        label = detection["label"]

        class_counts[label] += 1

        total_objects += 1


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nTotal detected objects:", total_objects)

print("\nObject Class Distribution")
print("----------------------------------------")

for label, count in class_counts.most_common():

    percentage = (
        100 * count / total_objects
    )

    print(
        f"{label:20s} "
        f"{count:8d} "
        f"{percentage:6.2f}%"
    )


print("\nUnique labels:", len(class_counts))
from huggingface_hub import hf_hub_download
import pandas as pd
from pathlib import Path


REPO_ID = "empiricearth/nexar_collision_prediction"

OUTPUT_DIR = Path("data/nexar")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUBSET_SIZE = 20
RANDOM_SEED = 42


# --------------------------------------------------
# DOWNLOAD METADATA
# --------------------------------------------------

positive_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename="train/positive/metadata.csv"
)

negative_path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename="train/negative/metadata.csv"
)


# --------------------------------------------------
# READ
# --------------------------------------------------

positive_df = pd.read_csv(
    positive_path
)

negative_df = pd.read_csv(
    negative_path
)


# --------------------------------------------------
# ADD LABELS
# --------------------------------------------------

positive_df["label"] = 1
positive_df["category"] = "positive"

negative_df["label"] = 0
negative_df["category"] = "negative"


# --------------------------------------------------
# RANDOM REPRODUCIBLE SAMPLE
# --------------------------------------------------

positive_subset = positive_df.sample(
    n=SUBSET_SIZE,
    random_state=RANDOM_SEED
)

negative_subset = negative_df.sample(
    n=SUBSET_SIZE,
    random_state=RANDOM_SEED
)


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

subset = pd.concat(
    [
        positive_subset,
        negative_subset
    ],
    ignore_index=True
)


# Shuffle final order
subset = subset.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = (
    OUTPUT_DIR
    / "experiment1_metadata.csv"
)

subset.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print(
    "Experiment-1 subset created."
)

print(
    "Total videos:",
    len(subset)
)

print(
    "Positive:",
    (subset["label"] == 1).sum()
)

print(
    "Negative:",
    (subset["label"] == 0).sum()
)


print("\nScene distribution:")
print(
    subset["scene"].value_counts()
)


print("\nWeather distribution:")
print(
    subset["weather"].value_counts(
        dropna=False
    )
)


print("\nSelected videos")
print("--------------------------------")

print(
    subset[
        [
            "file_name",
            "category",
            "label",
            "time_of_alert",
            "time_of_event",
            "scene",
            "weather"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nMetadata saved to:",
    output_path
)
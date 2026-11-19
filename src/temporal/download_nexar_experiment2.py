from pathlib import Path
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPO_ID = "empiricearth/nexar_collision_prediction"

METADATA_PATH = Path(
    "data/nexar/experiment2_metadata.csv"
)

EXP1_ROOT = Path(
    "data/nexar/experiment1"
)

EXP2_ROOT = Path(
    "data/nexar/experiment2"
)

POSITIVE_DIR = EXP2_ROOT / "positive"
NEGATIVE_DIR = EXP2_ROOT / "negative"

POSITIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NEGATIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "Experiment-2 videos:",
    len(metadata)
)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------

reused = 0
downloaded = 0
already_exists = 0
failed = 0


# --------------------------------------------------
# PROCESS EACH VIDEO
# --------------------------------------------------

for index, row in metadata.iterrows():

    file_name = row["file_name"]
    category = row["category"]

    destination_dir = (
        POSITIVE_DIR
        if category == "positive"
        else NEGATIVE_DIR
    )

    destination_path = (
        destination_dir / file_name
    )


    print(
        f"\n[{index + 1}/{len(metadata)}] "
        f"{category}: {file_name}"
    )


    # --------------------------------------------------
    # ALREADY IN EXPERIMENT 2
    # --------------------------------------------------

    if destination_path.exists():

        print("Already exists.")

        already_exists += 1

        continue


    # --------------------------------------------------
    # CHECK EXPERIMENT 1
    # --------------------------------------------------

    exp1_path = (
        EXP1_ROOT
        / category
        / file_name
    )


    if exp1_path.exists():

        shutil.copy2(
            exp1_path,
            destination_path
        )

        reused += 1

        print(
            "Reused from Experiment 1."
        )

        continue


    # --------------------------------------------------
    # DOWNLOAD FROM HUGGING FACE
    # --------------------------------------------------

    repo_filename = (
        f"train/{category}/{file_name}"
    )


    try:

        print("Downloading...")


        downloaded_path = hf_hub_download(

            repo_id=REPO_ID,

            repo_type="dataset",

            filename=repo_filename
        )


        shutil.copy2(
            downloaded_path,
            destination_path
        )


        downloaded += 1


        print(
            "Saved:",
            destination_path
        )


    except Exception as error:

        failed += 1

        print(
            "FAILED:",
            file_name
        )

        print(
            "Reason:",
            error
        )


# --------------------------------------------------
# FINAL COUNTS
# --------------------------------------------------

positive_count = len(
    list(
        POSITIVE_DIR.glob("*.mp4")
    )
)

negative_count = len(
    list(
        NEGATIVE_DIR.glob("*.mp4")
    )
)


print("\n================================")
print("EXPERIMENT 2 DOWNLOAD COMPLETE")
print("================================")

print(
    "Reused from Experiment 1:",
    reused
)

print(
    "Newly downloaded:",
    downloaded
)

print(
    "Already existed:",
    already_exists
)

print(
    "Failed:",
    failed
)

print(
    "\nPositive videos:",
    positive_count
)

print(
    "Negative videos:",
    negative_count
)

print(
    "Total videos:",
    positive_count
    + negative_count
)
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REPO_ID = "empiricearth/nexar_collision_prediction"

METADATA_PATH = Path(
    "data/nexar/experiment1_metadata.csv"
)

OUTPUT_ROOT = Path(
    "data/nexar/experiment1"
)

POSITIVE_DIR = OUTPUT_ROOT / "positive"
NEGATIVE_DIR = OUTPUT_ROOT / "negative"

POSITIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NEGATIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD OUR FIXED 40-VIDEO SUBSET
# --------------------------------------------------

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "Videos to download:",
    len(metadata)
)


# --------------------------------------------------
# DOWNLOAD VIDEOS
# --------------------------------------------------

successful = 0
failed = 0


for index, row in metadata.iterrows():

    file_name = row["file_name"]
    category = row["category"]


    if category == "positive":

        repo_filename = (
            f"train/positive/{file_name}"
        )

        destination_dir = POSITIVE_DIR

    else:

        repo_filename = (
            f"train/negative/{file_name}"
        )

        destination_dir = NEGATIVE_DIR


    destination_path = (
        destination_dir
        / file_name
    )


    # Skip videos already downloaded
    if destination_path.exists():

        print(
            f"[{index + 1}/"
            f"{len(metadata)}] "
            f"Already exists: "
            f"{file_name}"
        )

        successful += 1

        continue


    print(
        f"[{index + 1}/"
        f"{len(metadata)}] "
        f"Downloading "
        f"{category}: "
        f"{file_name}"
    )


    try:

        downloaded_path = hf_hub_download(

            repo_id=REPO_ID,

            repo_type="dataset",

            filename=repo_filename
        )


        # --------------------------------------------------
        # COPY FROM HF CACHE INTO PROJECT
        # --------------------------------------------------

        with open(
            downloaded_path,
            "rb"
        ) as source:

            with open(
                destination_path,
                "wb"
            ) as destination:

                while True:

                    chunk = source.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    destination.write(
                        chunk
                    )


        successful += 1


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
# SUMMARY
# --------------------------------------------------

print("\nDownload Complete")
print("--------------------------------")

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Positive folder:",
    POSITIVE_DIR
)

print(
    "Negative folder:",
    NEGATIVE_DIR
)
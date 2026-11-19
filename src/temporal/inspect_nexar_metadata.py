from huggingface_hub import hf_hub_download
import pandas as pd


REPO_ID = "empiricearth/nexar_collision_prediction"


# --------------------------------------------------
# DOWNLOAD ONLY METADATA FILES
# --------------------------------------------------

print("Downloading training metadata...")


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
# READ METADATA
# --------------------------------------------------

positive_df = pd.read_csv(
    positive_path
)

negative_df = pd.read_csv(
    negative_path
)


# --------------------------------------------------
# POSITIVE DATA
# --------------------------------------------------

print("\nPOSITIVE METADATA")
print("--------------------------------")

print(
    "Rows:",
    len(positive_df)
)

print(
    "Columns:",
    positive_df.columns.tolist()
)

print("\nFirst 5 positive samples:")

print(
    positive_df.head().to_string(
        index=False
    )
)


# --------------------------------------------------
# NEGATIVE DATA
# --------------------------------------------------

print("\n\nNEGATIVE METADATA")
print("--------------------------------")

print(
    "Rows:",
    len(negative_df)
)

print(
    "Columns:",
    negative_df.columns.tolist()
)

print("\nFirst 5 negative samples:")

print(
    negative_df.head().to_string(
        index=False
    )
)


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\n\nPOSITIVE MISSING VALUES")
print("--------------------------------")

print(
    positive_df.isnull().sum()
)


print("\nNEGATIVE MISSING VALUES")
print("--------------------------------")

print(
    negative_df.isnull().sum()
)
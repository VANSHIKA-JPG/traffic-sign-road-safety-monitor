from huggingface_hub import snapshot_download


print("Downloading BDD100K drivable-area masks...")


snapshot_download(
    repo_id="dgural/bdd100k",
    repo_type="dataset",

    allow_patterns=[
        "fields/drivable/*"
    ],

    local_dir="data/bdd100k"
)


print("Download complete.")
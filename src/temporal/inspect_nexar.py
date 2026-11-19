from huggingface_hub import list_repo_files


REPO_ID = "empiricearth/nexar_collision_prediction"


print("Reading Nexar repository...")

files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)


print("\nTotal repository files:", len(files))


print("\nFirst 50 files")
print("--------------------------------")

for file in files[:50]:
    print(file)


# Find likely metadata files
print("\nPossible metadata / annotation files")
print("--------------------------------")

for file in files:

    lower = file.lower()

    if (
        lower.endswith(".csv")
        or lower.endswith(".json")
        or lower.endswith(".jsonl")
        or "metadata" in lower
        or "annotation" in lower
    ):
        print(file)


# Count video files
video_files = [
    file
    for file in files
    if file.lower().endswith(
        (".mp4", ".avi", ".mov")
    )
]


print("\nVideo files:", len(video_files))


print("\nFirst 20 video files")
print("--------------------------------")

for file in video_files[:20]:
    print(file)
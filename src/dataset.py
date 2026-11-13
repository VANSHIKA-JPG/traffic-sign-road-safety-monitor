import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 64
BATCH_SIZE = 32
RANDOM_SEED = 42


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

full_train_dataset = datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

test_dataset = datasets.GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transform
)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

train_size = int(
    0.8 * len(full_train_dataset)
)

val_size = (
    len(full_train_dataset) - train_size
)


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)


train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)


# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# ONLY RUN WHEN dataset.py IS EXECUTED DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "Total original training images:",
        len(full_train_dataset)
    )

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    images, labels = next(
        iter(train_loader)
    )

    print("\nBatch information:")

    print(
        "Image batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )

    print(
        "First 10 labels:",
        labels[:10]
    )
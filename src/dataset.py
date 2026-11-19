import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 64
BATCH_SIZE = 32
RANDOM_SEED = 42


# --------------------------------------------------
# TRAINING TRANSFORM
# --------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Slight random rotation
    transforms.RandomRotation(
        degrees=10
    ),

    # Random brightness and contrast variation
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.ToTensor()
])


# --------------------------------------------------
# VALIDATION / TEST TRANSFORM
# --------------------------------------------------

eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()
])


# --------------------------------------------------
# LOAD TWO COPIES OF TRAINING DATA
# --------------------------------------------------

train_source = datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=train_transform
)

eval_source = datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=eval_transform
)


# --------------------------------------------------
# CREATE REPRODUCIBLE INDICES
# --------------------------------------------------

dataset_size = len(train_source)

train_size = int(
    0.8 * dataset_size
)

val_size = (
    dataset_size - train_size
)


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()


train_indices = indices[:train_size]

val_indices = indices[train_size:]


# --------------------------------------------------
# CREATE TRAIN / VALIDATION SUBSETS
# --------------------------------------------------

train_dataset = Subset(
    train_source,
    train_indices
)

val_dataset = Subset(
    eval_source,
    val_indices
)


# --------------------------------------------------
# OFFICIAL TEST DATA
# --------------------------------------------------

test_dataset = datasets.GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=eval_transform
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
# TEST DATA PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

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

    print(
        "\nTraining batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 32
RANDOM_SEED = 42


# --------------------------------------------------
# IMAGENET NORMALIZATION
# --------------------------------------------------

weights = ResNet18_Weights.DEFAULT

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# --------------------------------------------------
# TRAINING TRANSFORM
# --------------------------------------------------

train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# --------------------------------------------------
# VALIDATION / TEST TRANSFORM
# --------------------------------------------------

eval_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=mean,
        std=std
    )
])


# --------------------------------------------------
# LOAD TRAINING DATA
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
# SAME REPRODUCIBLE 80 / 20 SPLIT
# --------------------------------------------------

dataset_size = len(train_source)

train_size = int(
    0.8 * dataset_size
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
# SANITY CHECK
# --------------------------------------------------

if __name__ == "__main__":

    images, labels = next(
        iter(train_loader)
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

    print(
        "Image batch shape:",
        images.shape
    )

    print(
        "Label batch shape:",
        labels.shape
    )
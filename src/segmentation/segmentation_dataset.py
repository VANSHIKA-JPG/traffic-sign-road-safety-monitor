from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_DIR = Path("data/bdd100k/data")
MASK_DIR = Path("data/bdd100k/fields/drivable")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
RANDOM_SEED = 42


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class DrivableAreaDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_files = sorted(
            self.image_dir.glob("*.jpg")
        )

        # Keep only images that have a matching mask
        self.image_files = [
            image_path
            for image_path in self.image_files
            if (
                self.mask_dir
                / f"{image_path.stem}.png"
            ).exists()
        ]

        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        # ------------------------------------------
        # IMAGE
        # ------------------------------------------

        image_path = self.image_files[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.image_transform(
            image
        )

        # ------------------------------------------
        # MASK
        # ------------------------------------------

        mask_path = (
            self.mask_dir
            / f"{image_path.stem}.png"
        )

        mask = Image.open(
            mask_path
        )

        # Important:
        # nearest-neighbour interpolation preserves
        # segmentation class values.
        mask = TF.resize(
            mask,
            IMAGE_SIZE,
            interpolation=transforms.InterpolationMode.NEAREST
        )

        mask = np.array(
            mask,
            dtype=np.uint8
        )

        # Original:
        # 0 = non-drivable
        # 1/2 = drivable categories
        #
        # Our binary target:
        # 0 = non-drivable
        # 1 = drivable

        mask = (
            mask > 0
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# --------------------------------------------------
# CREATE DATASET
# --------------------------------------------------

dataset = DrivableAreaDataset(
    IMAGE_DIR,
    MASK_DIR
)

print(
    "Total image-mask pairs:",
    len(dataset)
)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# VERIFY ONE BATCH
# --------------------------------------------------

images, masks = next(
    iter(train_loader)
)


print(
    "Image batch shape:",
    images.shape
)

print(
    "Mask batch shape:",
    masks.shape
)

print(
    "Unique mask values:",
    torch.unique(masks)
)
import torch
import torch.nn as nn
from torchvision import models


class TrafficSignResNet(nn.Module):

    def __init__(self, num_classes=43, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet-18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # --------------------------------------------------
        # FREEZE PRETRAINED BACKBONE
        # --------------------------------------------------

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        # --------------------------------------------------
        # REPLACE FINAL CLASSIFIER
        # --------------------------------------------------

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            num_classes
        )


    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# SANITY CHECK
# --------------------------------------------------

if __name__ == "__main__":

    model = TrafficSignResNet(
        num_classes=43,
        freeze_backbone=True
    )

    sample = torch.randn(
        4,
        3,
        128,
        128
    )

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    print(
        "Total parameters:",
        total_parameters
    )
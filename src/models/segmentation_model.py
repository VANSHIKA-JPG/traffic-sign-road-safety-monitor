import torch
import torch.nn as nn


# --------------------------------------------------
# DOUBLE CONVOLUTION BLOCK
# --------------------------------------------------

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------
# U-NET
# --------------------------------------------------

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # ---------------- ENCODER ----------------

        self.encoder1 = DoubleConv(
            3, 32
        )

        self.pool1 = nn.MaxPool2d(2)


        self.encoder2 = DoubleConv(
            32, 64
        )

        self.pool2 = nn.MaxPool2d(2)


        self.encoder3 = DoubleConv(
            64, 128
        )

        self.pool3 = nn.MaxPool2d(2)


        # ---------------- BOTTLENECK ----------------

        self.bottleneck = DoubleConv(
            128, 256
        )


        # ---------------- DECODER ----------------

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            256, 128
        )


        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            128, 64
        )


        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            64, 32
        )


        # ---------------- OUTPUT ----------------

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder

        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )


        # Bottleneck

        b = self.bottleneck(
            self.pool3(e3)
        )


        # Decoder + skip connections

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.decoder3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.decoder2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.decoder1(d1)


        return self.output(d1)


# --------------------------------------------------
# TEST MODEL
# --------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)


    model = UNet().to(device)


    # Fake batch matching our dataset
    x = torch.randn(
        8,
        3,
        256,
        256
    ).to(device)


    with torch.no_grad():

        output = model(x)


    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        output.shape
    )


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )
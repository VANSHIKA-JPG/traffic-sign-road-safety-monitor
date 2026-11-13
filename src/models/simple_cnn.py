import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Block 2
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # After two pooling operations:
        # 64x64 -> 32x32 -> 16x16
        #
        # Final feature maps:
        # 32 x 16 x 16 = 8192

        self.fc1 = nn.Linear(
            32 * 16 * 16,
            128
        )

        # GTSRB has 43 classes
        self.fc2 = nn.Linear(
            128,
            43
        )


    def forward(self, x):

        # Input:
        # [batch, 3, 64, 64]

        x = self.conv1(x)
        # [batch, 16, 64, 64]

        x = self.relu(x)

        x = self.pool(x)
        # [batch, 16, 32, 32]


        x = self.conv2(x)
        # [batch, 32, 32, 32]

        x = self.relu(x)

        x = self.pool(x)
        # [batch, 32, 16, 16]


        x = torch.flatten(x, 1)
        # [batch, 8192]


        x = self.fc1(x)
        # [batch, 128]

        x = self.relu(x)


        x = self.fc2(x)
        # [batch, 43]

        return x
if __name__ == "__main__":

    model = SimpleCNN()

    # Fake batch with exactly the same shape
    # as our real DataLoader batch
    sample_batch = torch.randn(32, 3, 64, 64)

    output = model(sample_batch)

    print("Input shape:", sample_batch.shape)
    print("Output shape:", output.shape)
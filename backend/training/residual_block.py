import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)

        out += identity

        out = self.relu(out)

        return out


if __name__ == "__main__":

    x = torch.randn(2, 128, 16, 32)

    model = ResidualBlock(128)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

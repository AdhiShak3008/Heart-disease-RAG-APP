import torch
import torch.nn as nn


class DilatedBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=4,
                dilation=4,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)


if __name__ == "__main__":

    x = torch.randn(2, 128, 16, 32)

    model = DilatedBlock(128)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

import torch
import torch.nn as nn


class SplitSelfAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.shape

        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attention = torch.softmax(
            torch.bmm(q, k),
            dim=-1,
        )

        out = torch.bmm(
            v,
            attention.permute(0, 2, 1),
        )

        out = out.view(B, C, H, W)

        return self.gamma * out + x


if __name__ == "__main__":

    x = torch.randn(2, 128, 16, 32)

    model = SplitSelfAttention(128)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
    print("Gamma :", model.gamma)

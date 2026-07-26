import torch
import torch.nn as nn

from backend.training.cnn_encoder import CNNEncoder
from backend.training.residual_block import ResidualBlock
from backend.training.dilated_block import DilatedBlock
from backend.training.split_self_attention import SplitSelfAttention


class RosaNetBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = CNNEncoder()

        self.residual = ResidualBlock(128)

        self.dilated = DilatedBlock(128)

        self.attention = SplitSelfAttention(128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.encoder(x)

        x = self.residual(x)

        x = self.dilated(x)

        x = self.attention(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x


if __name__ == "__main__":

    model = RosaNetBranch()

    dummy = torch.randn(2, 1, 128, 256)

    output = model(dummy)

    print("Input :", dummy.shape)

    print("Output:", output.shape)

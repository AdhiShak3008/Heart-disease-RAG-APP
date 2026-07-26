import torch
import torch.nn as nn

from backend.training.rosanet_branch import RosaNetBranch


class RosaNet(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.av_branch = RosaNetBranch()
        self.mv_branch = RosaNetBranch()
        self.pv_branch = RosaNetBranch()
        self.tv_branch = RosaNetBranch()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, recordings):

        av = recordings[:, 0].unsqueeze(1)
        mv = recordings[:, 1].unsqueeze(1)
        pv = recordings[:, 2].unsqueeze(1)
        tv = recordings[:, 3].unsqueeze(1)

        av = self.av_branch(av)
        mv = self.mv_branch(mv)
        pv = self.pv_branch(pv)
        tv = self.tv_branch(tv)

        features = torch.cat(
            [av, mv, pv, tv],
            dim=1,
        )

        return self.classifier(features)


if __name__ == "__main__":

    model = RosaNet()

    recordings = torch.randn(2, 4, 128, 256)

    output = model(recordings)

    print("Input :", recordings.shape)
    print("Output:", output.shape)

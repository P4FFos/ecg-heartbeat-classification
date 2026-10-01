import torch.nn as nn


class ECGNet(nn.Module):
    def __init__(self, channels=(16, 32, 64), kernel=5, n_classes=4):
        super().__init__()

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        length = 200 // (2 ** len(channels))
        self.classifier = nn.Linear(channels[-1] * length, n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

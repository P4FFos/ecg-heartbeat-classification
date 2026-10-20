# ECGNet from src/model.py, rebuilt with ai8x layers for the MAX78002

import torch.nn as nn

import ai8x


class ECGNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=1, dimensions=(200, 1), channels=(8, 16, 32),
                 kernel=5, bias=True, **kwargs):
        super().__init__()

        layers = []
        in_ch = num_channels
        for i, out_ch in enumerate(channels):
            layer = ai8x.FusedConv1dBNReLU if i == 0 else ai8x.FusedMaxPoolConv1dBNReLU
            layers += [layer(in_ch, out_ch, kernel, padding=kernel // 2, bias=bias, **kwargs)]
            in_ch = out_ch
        layers += [ai8x.MaxPool1d(2, 2, **kwargs)]

        self.features = nn.Sequential(*layers)

        length = dimensions[0] // (2 ** len(channels))
        self.classifier = ai8x.Linear(channels[-1] * length, num_classes, bias=bias, wide=True,
                                      **kwargs)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


def ecgnet(pretrained=False, **kwargs):
    return ECGNet(**kwargs)


models = [
    {
        "name": "ecgnet",
        "min_input": 1,
        "dim": 1,
    },
]

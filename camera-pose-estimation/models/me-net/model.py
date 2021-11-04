import torch.nn as nn

from typing import OrderedDict
from torchinfo import summary


class MeNet(nn.Module):
    def __init__(self, outputs, batch_size) -> None:
        super().__init__()
        self.conv_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=6,
                            out_channels=16,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("relu3", nn.ReLU()),
                    (
                        "conv4",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu4", nn.ReLU()),
                    (
                        "conv5",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("relu5", nn.ReLU()),
                    (
                        "conv6",
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu6", nn.ReLU()),
                    (
                        "conv7",
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("relu7", nn.ReLU()),
                    (
                        "conv8",
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("relu8", nn.ReLU()),
                    (
                        "conv9",
                        nn.Conv2d(
                            in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                    ),
                    ("relu9", nn.ReLU()),
                ]
            )
        )
        summary(self, input_size=(batch_size, 2, 256, 256))

    def forward(self, input):
        ...

import torch.nn as nn

from typing import OrderedDict


def get_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    return (
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.Dropout(0.2),
        nn.ReLU(),
    )


class MeNet(nn.Module):
    def __init__(self, outputs) -> None:
        super().__init__()

        self.conv_encoder = nn.Sequential(
            *get_conv_block(6, 16, 7, 2, 3),
            *get_conv_block(16, 32, 5, 2, 2),
            *get_conv_block(32, 64, 3, 2, 1),
            *get_conv_block(64, 64, 3, 1, 1),
            *get_conv_block(64, 128, 3, 2, 1),
            *get_conv_block(128, 128, 3, 1, 1),
            *get_conv_block(128, 256, 3, 2, 1),
            *get_conv_block(256, 256, 3, 1, 1),
            *get_conv_block(256, 512, 3, 2, 1),
        )

        self.linear_encoder = nn.Sequential(
            nn.Linear(8192, 4092),
            nn.ReLU(),
            nn.Linear(4092, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, outputs),
        )

    def forward(self, input):
        output = self.conv_encoder(input)
        output = output.view(-1, output.size()[1] * output.size()[2] * output.size()[3])
        output = self.linear_encoder(output)

        return output

import torch
from torch import nn
from torchvision.ops import SqueezeExcitation

class SEBasicBlock(nn.Module):
    def __init__(self,
            in_channels,
            filters,
            kernel_size=3,
            squeeze_channels=None,
            ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=1,
                padding='same',
                padding_mode='replicate')
        self.relu = nn.ReLU()

        if squeeze_channels is None:
            squeeze_channels = in_channels // 4

        self.squeeze_excite = SqueezeExcitation(
                input_channels=self.filters,
                squeeze_channels=squeeze_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # SqueezeExcite contains the residual layer.
        x = self.squeeze_excite(x)
        return x

class SEResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        channels, rows, cols = input_size

        self.architecture = nn.Sequential(
                nn.Conv2d(channels, 64, 7,
                    padding='same', padding_mode='replicate'),
                nn.ReLU(),
                SEBasicBlock(64, 64, kernel_size=3),
                SEBasicBlock(64, 128, kernel_size=3),
                SEBasicBlock(128, 256, kernel_size=3),
                SEBasicBlock(256, 512, kernel_size=3),
                nn.Flatten(),
                nn.Linear(rows * cols * 512, num_classes),
                )

    def forward(self, x):
        x = self.architecture(x)
        return x


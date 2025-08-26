"""
Helper functions for InsightFace IR-SE architecture
Based on the implementation from TreB1eN's InsightFace_Pytorch
"""

import torch
from torch import nn
from collections import namedtuple

# Define the bottleneck configuration
Bottleneck = namedtuple('Bottleneck', ['in_channel', 'depth', 'stride'])


def get_blocks(num_layers):
    """
    Get the block configuration for different ResNet depths
    Based on the original InsightFace IR-SE50 architecture
    """
    if num_layers == 50:
        return [
            [Bottleneck(64, 64, 1)] * 3,
            [Bottleneck(64, 128, 2)] + [Bottleneck(128, 128, 1)] * 3,
            [Bottleneck(128, 256, 2)] + [Bottleneck(256, 256, 1)] * 13,
            [Bottleneck(256, 512, 2)] + [Bottleneck(512, 512, 1)] * 2
        ]
    elif num_layers == 100:
        return [
            [Bottleneck(64, 64, 1)] * 3,
            [Bottleneck(64, 128, 2)] + [Bottleneck(128, 128, 1)] * 13,
            [Bottleneck(128, 256, 2)] + [Bottleneck(256, 256, 1)] * 30,
            [Bottleneck(256, 512, 2)] + [Bottleneck(512, 512, 1)] * 3
        ]
    elif num_layers == 152:
        return [
            [Bottleneck(64, 64, 1)] * 3,
            [Bottleneck(64, 128, 2)] + [Bottleneck(128, 128, 1)] * 8,
            [Bottleneck(128, 256, 2)] + [Bottleneck(256, 256, 1)] * 36,
            [Bottleneck(256, 512, 2)] + [Bottleneck(512, 512, 1)] * 3
        ]
    else:
        raise ValueError(f"Unsupported num_layers: {num_layers}")


class Flatten(nn.Module):
    """
    Flatten layer for flattening tensor dimensions
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = torch.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        return x * y


class bottleneck_IR(nn.Module):
    """
    IR (Improved Residual) bottleneck block
    """
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(nn.Module):
    """
    IR-SE (Improved Residual with Squeeze-and-Excitation) bottleneck block
    """
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def l2_norm(input, axis=1):
    """
    L2 normalization function
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

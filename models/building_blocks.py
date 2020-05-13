from collections import OrderedDict

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, relu=False):
        super().__init__()

        padding = dilation if dilation > 1 else padding
        layers = OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels))
        ])

        if relu:
            layers['relu'] = nn.ReLU(inplace=True)

        self._layers = nn.Sequential(layers)

    def forward(self, features):
        return self._layers(features)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, pad, dilation):
        super().__init__()

        self.block1 = ConvBlock(in_channels, out_channels, 3, stride, pad, dilation, relu=True)
        self.block2 = ConvBlock(out_channels, out_channels, 3, 1, pad, dilation)

        self.stride = stride
        self.downsample = None if self.stride == 1 and in_channels == out_channels else \
            ConvBlock(in_channels, out_channels, 1, self.stride, 0)

    def forward(self, features):
        x = features
        out = self.block1(x)
        out = self.block2(out)

        if self.downsample:
            x = self.downsample(x)
        return out.add_(x)

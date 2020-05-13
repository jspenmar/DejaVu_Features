from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.building_blocks import ConvBlock, ResidualBlock
from utils import ops


class DejaVu(nn.Module):
    def __init__(self, n_dims, norm=True):
        super().__init__()
        self.n_dims = n_dims
        self.norm = norm
        self.in_channels = 32

        self.first_conv = nn.Sequential(OrderedDict([
            ('block1', ConvBlock(3, 32, 3, 2, 1, 1, relu=True)),
            ('block2', ConvBlock(32, 32, 3, 1, 1, 1, relu=True)),
            ('block3', ConvBlock(32, 32, 3, 1, 1, 1, relu=True))
        ]))

        self.layer1 = self._make_layer(ResidualBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(ResidualBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(ResidualBlock, 128, 3, 1, 1, 2)

        self.branch1 = self._make_pooling_branch(32)
        self.branch2 = self._make_pooling_branch(16)
        self.branch3 = self._make_pooling_branch(8)
        self.branch4 = self._make_pooling_branch(4)

        self.last_conv = nn.Sequential(OrderedDict([
            ('block1', ConvBlock(320, 128, 3, 1, 1, 1, relu=True)),
            ('conv1',  nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))
        ]))

        self.final = nn.Conv2d(32, self.n_dims, kernel_size=1, padding=0, stride=1, bias=False)

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.n_dims}, {self.norm})'

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--n-dims', default=3, type=int, help='Number of feature descriptor dimensions.')
        parser.add_argument('--norm', default=False, action='store_true', help='Apply unit length feature norm.')

    def load_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=ops.get_map_location())
        try:
            self.load_state_dict(ckpt['model'])
        except RuntimeError:
            state = {k.replace('branch.', ''): v for k, v in ckpt['model'].items()}
            self.load_state_dict(state)

    @classmethod
    def from_ckpt(cls, ckpt_file, n_dims=None):
        ckpt = torch.load(ckpt_file, map_location=ops.get_map_location())
        try:
            _n_dims = ckpt['model']['branch.final.weight'].shape[0]
            model = cls(_n_dims)
            model.load_state_dict(ckpt['model'])
        except (RuntimeError, KeyError):
            state = {k.replace('branch.', ''): v for k, v in ckpt['model'].items()}
            _n_dims = state['final.weight'].shape[0]
            model = cls(_n_dims)
            model.load_state_dict(state)

        if n_dims:
            assert n_dims == model.n_dims, f'Inconsistent number of feature dimensions ({n_dims} vs. {model.n_dims})'

        return model

    def _make_layer(self, block, out_channels, blocks, stride, pad, dilation):
        layers = OrderedDict([('block1', block(self.in_channels, out_channels, stride, pad, dilation))])
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers[f'block{i+1}'] = block(self.in_channels, out_channels, 1, pad, dilation)
        return nn.Sequential(layers)

    @staticmethod
    def _make_pooling_branch(kernel_size):
        return nn.Sequential(OrderedDict([
            ('pool', nn.AvgPool2d(kernel_size, stride=kernel_size)),
            ('block', ConvBlock(128, 32, 1, 1, 0, 1, relu=True))
        ]))

    def forward(self, x):
        # Encoder
        output = self.first_conv(x)
        skip = output
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        # SPP module
        output_branch1 = ops.upsample_like(self.branch1(output_skip), output_skip)
        output_branch2 = ops.upsample_like(self.branch2(output_skip), output_skip)
        output_branch3 = ops.upsample_like(self.branch3(output_skip), output_skip)
        output_branch4 = ops.upsample_like(self.branch4(output_skip), output_skip)

        # Decoder
        encoder_features = (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1)
        output_features = torch.cat(encoder_features, 1)

        output_features = ops.upsample_like(output_features, skip)
        output_features = self.last_conv(output_features).add_(skip)

        output_features = ops.upsample_like(output_features, x)
        output_features = self.final(output_features)

        return F.normalize(output_features) if self.norm else output_features

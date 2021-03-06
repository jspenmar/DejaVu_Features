import torch
import torch.nn as nn

from losses.contextual import ContextualLoss
from utils import ops


class ContextualTripletLoss(nn.Module):
    """
    Contextual loss triplet loss. Objective is to replace traditional distance measures with contextual similarity.
    NOTE: This means that we need to invert the loss -> positives should have high similarity values, negatives should
    have los values.

    Positive pairs are given by each stacked input feature map.
    Negatives are generated by randomly reordering pairs in the batch.

    Attributes:
        margin (float): Target margin between positive and negative samples
        offset (float): Distance offset
        bandwidth (float): "Hardness" of the similarity margin
        sample_numel (int): Square root of number of features to sample
        is_symmetric (bool): Compute symmetric loss, i.e. loss(source, target) and loss(target, source)

    Methods:
        forward: Compute contextual triplet loss

    """
    def __init__(self, margin=0.5, offset=1.0, bandwidth=0.5, sample_numel=64, is_symmetric=True, scale=1, alpha=1):
        super().__init__()
        self.margin = margin
        self.cx_args = (offset, bandwidth, sample_numel, is_symmetric)
        self.cx_loss = ContextualLoss(*self.cx_args, is_similarity=True)
        self.scale = scale
        self.alpha = alpha

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.margin}, {", ".join(map(str, self.cx_args))})'

    @staticmethod
    def add_parser_args(parser):
        ContextualLoss.create_parser(parser)
        parser.add_argument('--margin', default=0.5, type=float, help='Target margin between anchor similarities.')
        parser.add_argument('--cx-scale', default=1, type=int, help='Downsample cx feature maps.')
        parser.add_argument('--cx-alpha', default=1, type=float, help='Same pair weight.')

    def forward(self, features):
        if self.scale > 1:
            features = [ops.downsample(feat, self.scale) for feat in features]
        anchors, positives, negatives = features

        xseasonal_loss = self.pair_loss(anchors, positives, negatives).mean()

        seasonal_loss = 0
        if self.alpha > 0:
            anc = torch.stack((anchors[0], positives[0], negatives[0]))
            pos = torch.stack((anchors[1], positives[1], negatives[1]))
            neg = torch.stack((negatives[0], negatives[1], anchors[0]))
            seasonal_loss = self.alpha * self.pair_loss(anc, pos, neg).mean()

        loss = xseasonal_loss + seasonal_loss
        return loss

    def pair_loss(self, anchor, positive, negative):
        cx_anchor_pos = self.cx_loss(anchor, positive)
        cx_anchor_neg = self.cx_loss(anchor, negative)

        loss = cx_anchor_neg - cx_anchor_pos + self.margin
        loss.clamp_(min=0.0)
        return loss

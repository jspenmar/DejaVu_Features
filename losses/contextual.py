import torch
import torch.nn as nn

from utils.ops import random_sample


class ContextualLoss(nn.Module):
    """
    Based on https://github.com/roimehrez/contextualLoss
            similarity = exp((b - d) / sigma) where d are the normalised distances
            loss = -log(similarity)

    The objective is to obtain a measure of the similarity between two images without requiring alignment between them.
    Original contextual loss asks for a maximum number of elements, above which we randomly sample from each image.
    This implementation provides an alternative, where we can either sample randomly or use the whole image, regardless
    of its size (typically very slow, recommended downsampling).
    The main benefit from using the whole image is that we get the expected value of '1' when comparing an image to
    itself.

    Hyperparameters have been renamed: "sigma" -> "bandwidth", "b" -> "offset"

    Attributes:
        offset (float): Distance offset
        bandwidth (float): "Hardness" of the similarity margin
        sample_numel (int): Square root of number of features to sample
        is_symmetric (bool): Compute symmetric loss, i.e. loss(source, target) and loss(target, source)
        is_similarity (bool): Return similarity instead of loss

    Methods:
        forward: Compute contextual loss
    """
    def __init__(self, offset=1, bandwidth=0.5, sample_numel=None, is_symmetric=True, is_similarity=False):
        super().__init__()
        self.offset = offset
        self.bandwidth = bandwidth
        self.sample_numel = sample_numel
        self.is_symmetric = is_symmetric
        self.is_similarity = is_similarity

    def __repr__(self):
        params = (self.offset, self.bandwidth, self.sample_numel, self.is_symmetric, self.is_similarity)
        return f'{self.__class__.__qualname__}{params}'

    @staticmethod
    def create_parser(parser):
        parser.add_argument('--bandwidth', default=0.5, type=float, help='Contextual loss band-width.')
        parser.add_argument('--offset', default=1.0, type=float, help='Contextual distance normalization.')
        parser.add_argument('--sample-numel', default=None, type=int, help='Maximum number of elements')
        parser.add_argument('--is-symmetric', default=False, action='store_true', help='Symmetric contextual loss')
        parser.add_argument('--is-similarity', default=False, action='store_true', help='Return similarity')

    def forward(self, source, target):
        fn = self.similarity if self.is_similarity else self.loss

        if self.sample_numel:
            if source.shape[-2]*source.shape[-1] > self.sample_numel**2:
                source = random_sample(source, self.sample_numel)[0]

            if target.shape[-2] * target.shape[-1] > self.sample_numel ** 2:
                target = random_sample(target, self.sample_numel)[0]

        out = fn(source, target)
        if self.is_symmetric:
            out = (out + fn(target, source))/2
        return out

    def loss(self, source, target):
        cx_similarity = self.similarity(source, target)
        loss = -1*cx_similarity.log()
        return loss.mean()

    def similarity(self, source, target):
        b, c, h, w = source.shape
        b2, c2 = target.shape[:2]

        # All feature vectors as  [b, (h x w), c]
        source_vecs = source.reshape(b, c, -1).permute(0, 2, 1)
        target_vecs = target.reshape(b2, c2, -1).permute(0, 2, 1)

        # Norm for a^2 + b^2 - 2ab
        source_norm = (source_vecs ** 2).sum(dim=-1)
        target_norm = (target_vecs ** 2).sum(dim=-1)

        # Step to avoid OOM
        max_numel, target_numel = 64**2**2, h*w
        step = max_numel // target_numel

        # Calculate similarities
        max_similarity = []
        for i in range(b):
            running_max = None
            t_vec, t_sqr = target_vecs[i], target_norm[i].reshape(-1, 1)
            for j in range(0, target_numel, step):
                j_end = min(j+step+1, target_numel)
                s_vec, s_sqr = source_vecs[i, j:j_end], source_norm[i, j:j_end]

                if s_vec.dim() == 1:
                    s_vec.unsqueeze_(0)

                diffs = t_vec @ s_vec.t()
                dist = t_sqr - 2 * diffs + s_sqr
                dist.clamp_(min=0.0).squeeze_()

                min_dist = dist.min(0)[0]
                norm_dist = dist / (min_dist + 1e-5)

                similarity = torch.exp((self.offset - norm_dist) / self.bandwidth)
                norm_similarity = similarity / similarity.sum(0)

                norm_similarity = norm_similarity.max(1)[0]
                running_max = norm_similarity if running_max is None else torch.max(running_max, norm_similarity)
            max_similarity.append(running_max)

        return torch.stack(tuple(max_similarity), dim=0).mean(dim=-1)

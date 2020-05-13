from pathlib import Path
import sys

import torch
from skimage.io import imread

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from models.dejavu import DejaVu
from losses.contextual import ContextualLoss
from utils import ops


if __name__ == '__main__':
    device = ops.get_device()

    ckpt_file = root_path / 'ckpts' / 'alpha_04.pt'

    model = DejaVu.from_ckpt(ckpt_file).to(device)
    model.eval()

    loss = ContextualLoss(offset=1.0, bandwidth=0.5, is_similarity=True).to(device)
    loss.eval()

    imfiles = [
        'image_anchor_0.png', 'image_anchor_1.png',
        'image_positive_0.png', 'image_positive_1.png',
        'image_negative_0.png', 'image_negative_1.png'
    ]
    batch_size = len(imfiles)

    def load_image(file):
        image = imread(root_path / 'images' / file)
        return ops.img2torch(image)

    images = torch.stack([load_image(file) for file in imfiles])

    # Compute features and downsample
    with torch.no_grad():
        images = images.to(device)
        images = ops.downsample(images, 2)
        dense_features = model(images)
        dense_features = ops.downsample(dense_features, 4)  # Downsample for faster processing

    # Compute all pairs of similarities
    # We expect all anchor/positive pairs to be higher than the negatives
    # NOTE: Similarities depend on each model, which might be higher or lower
    sims = torch.zeros(batch_size, batch_size)
    for i, fmap in enumerate(dense_features):
        fmap = fmap[None].repeat(batch_size, 1, 1, 1)
        sims[i] = loss(fmap, dense_features)

    print(sims)


import numpy as np

import torch
import torch.nn.functional as F


def get_device(device=None):
    if isinstance(device, torch.device):
        return device
    return torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_map_location():
    return None if torch.cuda.is_available() else lambda storage, loc: storage


def downsample(tensor, factor, bilinear=True):
    """Downsample tensor by a specified scaling factor."""
    return F.interpolate(tensor, scale_factor=1/factor, mode='bilinear', align_corners=True)


def upsample_like(tensor, ref_tensor):
    """Upsample tensor to match ref_tensor shape."""
    return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='bilinear', align_corners=True)


def extract_kpt_vectors(tensor, kpts, rand_batch=False):
    """
    Pick channel vectors from 2D location in tensor.
    E.g. tensor[b, :, y1, x1]

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts: Tensor with 'n' keypoints (x, y) as [b, n, 2]
    :param rand_batch: Randomize tensor in batch the vector is extracted from
    :return: Tensor entries as [b, n, c]
    """
    batch_size, num_kpts = kpts.shape[:-1]  # [b, n]

    # Reshape as a single batch -> [b*n, 2]
    tmp_idx = kpts.contiguous().view(-1, 2).long()

    # Flatten batch number indexes  -> [b*n] e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b_num = torch.arange(batch_size)
    b_num = b_num.repeat((num_kpts, 1)).view(-1)
    b_num = torch.sort(b_num)[0] if not rand_batch else b_num[torch.randperm(len(b_num))]

    # Perform indexing and reshape to [b, n, c]
    return tensor[b_num, :, tmp_idx[:, 1], tmp_idx[:, 0]].reshape([batch_size, num_kpts, -1])


def reshape_as_vectors(tensor):
    """Reshape from (b, c, h, w) to (b, h*w, c)."""
    b, c = tensor.shape[:2]
    return tensor.reshape(b, c, -1).permute(0, 2, 1)


def random_sample(tensor, n_sqrt):
    """ Randomly sample n**2 vectors from a tensor and arrange into a square tensor."""
    n = n_sqrt ** 2

    b, c, h, w = tensor.shape
    x, y = torch.randint(high=w, size=(b, n), dtype=torch.int), torch.randint(high=h, size=(b, n), dtype=torch.int)
    kpts = torch.stack((x, y), dim=-1)

    entries = extract_kpt_vectors(tensor, kpts)
    return entries.reshape(b, n_sqrt, n_sqrt, c).permute(0, -1, 1, 2), kpts


def np2torch(array, dtype=torch.float):
    """
    Convert a numpy array to torch tensor convention.
    If 4D -> [b, h, w, c] to [b, c, h, w]
    If 3D -> [h, w, c] to [c, h, w]

    :param array: Numpy array
    :param dtype: Target tensor dtype
    :return: Torch tensor
    """
    tensor, d = torch.from_numpy(array), array.ndim
    perm = [0, 3, 1, 2] if d == 4 else [2, 0, 1] if d == 3 else None
    tensor = tensor.permute(perm) if perm else tensor
    return tensor.type(dtype)


def img2torch(img, batched=False):
    """
    Convert single image to torch tensor convention.
    Image is normalized and converted to 4D: [1, 3, h, w]

    :param img: Numpy image
    :param batched: Return as 4D or 3D (default)
    :return: Torch tensor
    """
    img = torch.from_numpy(img.astype(np.float32)).permute([2, 0, 1])
    if img.max() > 1:
        img /= img.max()
    return img[None] if batched else img

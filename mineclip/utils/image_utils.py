from __future__ import annotations

import warnings
import yaml
import torch

from .torch_utils import torch_normalize


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['data_config']
MC_IMAGE_SIZE = config['image_size']
MC_IMAGE_MEAN = config['image_mean']
MC_IMAGE_STD = config['image_std']
CLIP_FRAME_NUM = config['clip_frame']


@torch.no_grad()
def basic_image_tensor_preprocess(
    img,
    mean: tuple[float, float, float] = MC_IMAGE_MEAN,
    std: tuple[float, float, float] = MC_IMAGE_SIZE,
    shape: tuple[int, int] | None = MC_IMAGE_SIZE,
):
    """
    Check for resize, and divide by 255
    """
    import kornia

    assert torch.is_tensor(img)
    assert img.dim() >= 4
    original_shape = list(img.size())
    img = img.float()
    img = img.flatten(0, img.dim() - 4)
    assert img.dim() == 4

    input_size = img.size()[-2:]    # height, width

    if shape and input_size != shape:
        warnings.warn(
            f'{"Down" if shape < input_size else "Up"}sampling image'
            f" from original resolution {input_size}x{input_size}"
            f" to {shape}x{shape}"
        )
        img = kornia.geometry.transform.resize(img, shape).clamp(0.0, 255.0)

    B, C, H, W = img.size()
    assert C % 3 == 0, "channel must divide 3"
    img = img.view(B * C // 3, 3, H, W)
    img = torch_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)

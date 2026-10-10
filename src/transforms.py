"""The one project-specific augmentation missing from MMSegmentation core."""

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from PIL import Image
from torchvision.transforms import ColorJitter


@TRANSFORMS.register_module()
class TorchvisionColorJitter(BaseTransform):
    """Apply torchvision ColorJitter to an MMCV BGR uint8 image only.

    Geometric transforms remain MMSegmentation transforms, so the semantic mask
    is never interpolated by this photometric operation.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def transform(self, results):
        rgb = results['img'][..., ::-1].copy()
        rgb = np.asarray(self.jitter(Image.fromarray(rgb))).copy()
        results['img'] = rgb[..., ::-1].copy()
        return results

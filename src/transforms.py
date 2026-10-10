"""DFM Original-compatible Cityscapes preprocessing."""

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


@TRANSFORMS.register_module()
class DFMCityscapesTransform(BaseTransform):
    """Reproduce the DFM Original fixed-resolution tensor pipeline.

    MMCV loads BGR uint8 HWC arrays. DFM uses RGB float CHW tensors in [0, 1],
    applies flip before torchvision resize, then tensor ColorJitter, clamp and
    ImageNet normalization. The final HWC conversion is value-preserving and
    lets MMSegmentation's standard PackSegInputs create the CHW tensor.
    """

    def __init__(self, size, training=False, flip_probability=0.5,
                 brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.size = tuple(size)
        self.training = training
        self.flip_probability = flip_probability
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def transform(self, results):
        # Match torchvision Cityscapes + pil_to_tensor(...).float() / 255.0.
        rgb = results['img'][..., ::-1].copy()
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(results['gt_seg_map'].copy()).long()

        # DFM Original order: flip -> fixed resize -> jitter -> normalize.
        if self.training and torch.rand(()) < self.flip_probability:
            image = torch.flip(image, (2,))
            mask = torch.flip(mask, (1,))

        image = TF.resize(
            image,
            self.size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True)
        mask = TF.resize(
            mask[None],
            self.size,
            interpolation=TF.InterpolationMode.NEAREST)[0].long()

        if self.training:
            image = self.jitter(image).clamp(0.0, 1.0)

        mean = image.new_tensor(self.mean)[:, None, None]
        std = image.new_tensor(self.std)[:, None, None]
        image = (image - mean) / std

        # PackSegInputs expects an HWC array and transposes it back to CHW.
        results['img'] = image.permute(1, 2, 0).contiguous().numpy()
        results['gt_seg_map'] = mask.numpy()
        results['img_shape'] = self.size
        return results


@TRANSFORMS.register_module()
class SetFixedEvaluationMetadata(BaseTransform):
    """Make the resized resolution the canonical evaluation resolution.

    MMSegmentation post-processes logits back to ``ori_shape``. This transform
    is deliberately placed after the joint image/GT Resize so predictions are
    restored to the fixed evaluation size rather than native Cityscapes size.
    """

    def __init__(self, size=(256, 512)):
        self.size = tuple(size)

    def transform(self, results):
        image_shape = tuple(results['img'].shape[:2])
        gt_shape = tuple(results['gt_seg_map'].shape[:2])
        if image_shape != self.size or gt_shape != self.size:
            raise ValueError(
                'Fixed evaluation resize must produce image and GT shapes '
                f'{self.size}, but got image={image_shape}, GT={gt_shape}.')

        # BaseSegmentor.postprocess_result() resizes logits to ori_shape.
        results['ori_shape'] = self.size
        results['img_shape'] = self.size
        results['pad_shape'] = self.size
        return results

# Cityscapes discriminative semantic-segmentation baselines

This project provides six comparable discriminative baselines for the
Cityscapes 19-class semantic-segmentation task. Model components and
model-specific objectives come directly from the MMSegmentation 1.2.2 and
MMDetection 3.3.0 registries. The local code only supplies the common data
policy, detailed metrics, entry points, and checkpoint aliases.

## Environment

The lock targets Python 3.10, PyTorch 2.1.2 + CUDA 12.1, torchvision 0.16.2,
compiled MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation 1.2.2, and MMDetection
3.3.0. `mmcv`, rather than `mmcv-lite`, is deliberately used because
Mask2Former needs compiled MultiScaleDeformableAttention operations.

```bash
cd /home/igarashi_25/discrimination
uv sync --frozen
```

The NVIDIA driver may advertise a newer CUDA version; CUDA 12.1 PyTorch wheels
are forward-compatible with a sufficiently recent driver. No system CUDA
toolkit is needed for the pinned prebuilt wheels.

## Dataset and common protocol

The dataset root is fixed in the shared config as:

```text
/home/igarashi_25/datasets/cityscapes
├── leftImg8bit/{train,val}
└── gtFine/{train,val}
```

MMSegmentation's `CityscapesDataset` reads `*_gtFine_labelTrainIds.png`.
Official train IDs 0--18 are the only semantic targets. Void pixels retain
label 255 and are excluded by every CE loss and by the 19x19 evaluation
confusion matrix. Mask2Former's official conversion also removes label 255
before Hungarian matching; its internal no-object class remains intact.

Common conditions:

- train/validation splits: `train` / `val`
- resolutions: 256x512 and 128x256 (H x W)
- train augmentation: the tensor implementation and operation order used by
  [DFM-ver2 Original](https://github.com/igaryou/DFM-ver2)
- validation: the same tensor resize and ImageNet normalization, without flip
  or ColorJitter
- batch size: 4; 800 epochs; validation every 100 epochs
- bf16 autocast through `AmpOptimWrapper`
- AdamW, lr 1e-4, weight decay 0.01 for every model
- no warmup; PolyLR power 1.0 from 1e-4 to 1e-6 across 800 epochs
- whole-image inference; final logits/predictions are restored to the selected
  fixed resolution with `align_corners=False`

No random crop, random scale, or PhotoMetricDistortion is used. Backbone
parameter-wise LR multipliers from some upstream recipes are intentionally
removed so optimizer conditions remain common; this does not alter model
architecture or model-specific loss.

### DFM Original-compatible augmentation

`DFMCityscapesTransform` reproduces the fixed-resolution path in DFM-ver2
`src/dataset.py`. MMSegmentation's BGR uint8 array is explicitly converted to
an RGB float32 CHW tensor in `[0, 1]`. Training then applies:

```text
RGB float [0,1]
-> horizontal flip (p=0.5)
-> torchvision.functional.resize(image, bilinear, antialias=True)
-> torchvision.functional.resize(mask, nearest)
-> torchvision ColorJitter(0.2, 0.2, 0.2, 0.1) on the float tensor
-> clamp(0, 1)
-> (image - ImageNet mean) / ImageNet std
-> PackSegInputs
```

Validation omits flip and ColorJitter, using RGB float conversion, the same
torchvision image/mask resize, and ImageNet normalization. The model
`SegDataPreProcessor` has `mean=None`, `std=None`, and `bgr_to_rgb=False`, so it
continues to move, pad and batch samples without applying normalization or
channel conversion a second time.

For validation/test, the selected fixed size is recorded as `ori_shape`, so
MMSegmentation post-processing never restores native Cityscapes resolution.
The metric enforces matching GT/prediction shapes: 256x512 for the standard
configs and 128x256 for the additional low-resolution configs.

## Architectures and provenance

| Config | Official MMSegmentation basis | Backbone pretraining | Approx. params |
|---|---|---|---:|
| DeepLabV3+ R101 | `DepthwiseSeparableASPPHead`, R101-D8 Cityscapes config; dilations 1/12/24/36, C1=48, FCN auxiliary loss 0.4 | `open-mmlab://resnet101_v1c` (ImageNet) | ~63M |
| PSPNet R101 | `PSPHead`, R101-D8 Cityscapes config; pools 1/2/3/6, FCN auxiliary loss 0.4 | `open-mmlab://resnet101_v1c` (ImageNet) | ~68M |
| OCRNet HRNet-W48 | `CascadeEncoderDecoder`, `HRNet`, official FCN stage + `OCRHead` | `open-mmlab://msra/hrnetv2_w48` (ImageNet) | ~70M |
| Mask2Former Swin-S | official `Mask2FormerHead`, MSDeformAttn pixel decoder, 9-layer transformer decoder, 100 queries, Hungarian assignment, classification/mask/Dice losses and deep supervision | official Swin-S ImageNet-1K checkpoint | ~69M |
| UPerNet Swin-S | official `UPerHead` PPM+FPN and FCN auxiliary head; Swin depths 2/2/18/2 | official Swin-S ImageNet-1K checkpoint | ~81M |
| SegFormer MiT-B5 | official `MixVisionTransformer` B5 and `SegformerHead`; depths 3/6/40/3 | official MiT-B5 ImageNet-1K checkpoint | ~85M |

The parameter counts are expected values for the official-sized
architectures, not measurements from this repository (runtime model
construction was intentionally not performed). All decoders start from
random initialization; no Cityscapes-trained segmentation checkpoint is used.

Changes from upstream experiment configs are limited to the requested common
dataset resolution/augmentation, 19 classes, 800-epoch optimizer/schedule,
batch size, bf16 precision, validation/checkpoint cadence, metrics, and W&B.
The official decoder widths, backbone topology, auxiliary heads and loss
weights are retained. For fair optimization, warmups and upstream
parameter-wise LR/weight-decay multipliers are omitted.

## Training

Authenticate once with `uv run wandb login`. For 256x512, run one of:

```bash
scripts/train_deeplabv3plus_r101.sh
scripts/train_pspnet_r101.sh
scripts/train_ocrnet_hrnet_w48.sh
scripts/train_mask2former_swin_s.sh
scripts/train_upernet_swin_s.sh
scripts/train_segformer_mit_b5.sh
```

For 128x256, run one of:

```bash
scripts/train_deeplabv3plus_r101_128x256.sh
scripts/train_pspnet_r101_128x256.sh
scripts/train_ocrnet_hrnet_w48_128x256.sh
scripts/train_mask2former_swin_s_128x256.sh
scripts/train_upernet_swin_s_128x256.sh
scripts/train_segformer_mit_b5_128x256.sh
```

Each script forwards extra CLI options. Examples:

```bash
scripts/train_upernet_swin_s.sh --resume
scripts/train_segformer_mit_b5.sh --no-wandb
scripts/train_pspnet_r101.sh --work-dir results/pspnet_trial
```

W&B project `cityscapes-discriminative-baselines` uses an explicit model run
name. MMEngine records training loss (including named auxiliary or
Mask2Former components), learning rate, epoch, and validation metrics.

## Evaluation

256x512 example:

```bash
uv run python -m src.evaluate \
  configs/deeplabv3plus_r101.py \
  results/deeplabv3plus_r101_256x512_800ep/best_miou.pth
```

128x256 example:

```bash
uv run python -m src.evaluate \
  configs/deeplabv3plus_r101_128x256.py \
  results/deeplabv3plus_r101_128x256_800ep/best_miou.pth
```

Replace the config/checkpoint pair for another model. Evaluation returns pixel
accuracy, mIoU, mAcc, and all 19 class IoU/accuracy values. It also writes
`latest_metrics.json` under the config's `evaluation/` directory, including
the full 19x19 confusion matrix. GT and prediction are evaluated at the fixed
config resolution; native-resolution Cityscapes evaluation is not performed.

## Checkpoints and outputs

Each run writes to `results/<run-name>/`:

- `latest.pth`: overwritten after every epoch and fully resumable
- `best_miou.pth`: overwritten only when validation mIoU improves
- `epoch_0100.pth` ... `epoch_0800.pth`: periodic snapshots
- optimizer, parameter-scheduler, epoch, iteration, config, and RNG metadata
  are stored in every checkpoint
- logs and local visualization backend artifacts
- `validation/latest_metrics.json` and `evaluation/latest_metrics.json`

Use `--resume` to discover `latest.pth`, or pass an explicit path:

```bash
scripts/train_deeplabv3plus_r101.sh \
  --resume results/deeplabv3plus_r101_256x512_800ep/latest.pth
```

## Scope of verification

The configs and source were checked statically for model/head names, config
inheritance, class count, ignore index, data paths, schedule, and script
arguments. Per the task requirements, training, validation, model
forward/backward, GPU-memory tests, checkpoint downloads, and unit tests were
not run.

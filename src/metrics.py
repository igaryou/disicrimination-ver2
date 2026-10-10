"""Detailed 19-class Cityscapes metrics."""

import json
from pathlib import Path

import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


CITYSCAPES_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle')


@METRICS.register_module()
class CityscapesDetailedMetric(BaseMetric):
    """Compute a 19x19 confusion matrix and all requested semantic metrics."""

    default_prefix = 'val'

    def __init__(self, ignore_index=255, output_dir=None,
                 target_size=(256, 512), collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index
        self.output_dir = output_dir
        self.target_size = tuple(target_size)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
            target = sample.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
            if pred.shape != self.target_size or target.shape != self.target_size:
                raise ValueError(
                    'Cityscapes evaluation requires prediction and GT shapes '
                    f'{self.target_size}, but got prediction={pred.shape}, '
                    f'GT={target.shape}.')
            valid = ((target != self.ignore_index) & (target >= 0) &
                     (target < 19) & (pred >= 0) & (pred < 19))
            hist = np.bincount(
                19 * target[valid] + pred[valid], minlength=19 * 19)
            self.results.append(hist.reshape(19, 19))

    def compute_metrics(self, results):
        confusion = np.sum(results, axis=0, dtype=np.int64)
        true_count = confusion.sum(axis=1)
        pred_count = confusion.sum(axis=0)
        correct = np.diag(confusion)
        union = true_count + pred_count - correct
        iou = np.divide(correct, union, out=np.full(19, np.nan), where=union > 0)
        acc = np.divide(
            correct, true_count, out=np.full(19, np.nan), where=true_count > 0)

        metrics = {
            'pixel_acc': float(correct.sum() / max(true_count.sum(), 1) * 100),
            'mIoU': float(np.nanmean(iou) * 100),
            'mAcc': float(np.nanmean(acc) * 100),
        }
        for index, name in enumerate(CITYSCAPES_CLASSES):
            metrics[f'IoU/{name}'] = float(iou[index] * 100)
            metrics[f'Acc/{name}'] = float(acc[index] * 100)

        if self.output_dir and self.dataset_meta.get('classes'):
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                'classes': list(self.dataset_meta['classes']),
                'confusion_matrix': confusion.tolist(),
                'metrics_percent': metrics,
            }
            (output_dir / 'latest_metrics.json').write_text(
                json.dumps(report, indent=2), encoding='utf-8')
        return metrics

"""Stable checkpoint aliases requested by the experiment protocol."""

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class LatestCheckpointHook(Hook):
    """Overwrite latest.pth after every epoch for interruption-safe resume."""

    priority = 'VERY_LOW'

    @master_only
    def after_train_epoch(self, runner):
        runner.save_checkpoint(
            runner.work_dir, 'latest.pth', save_optimizer=True,
            save_param_scheduler=True, by_epoch=True)


@HOOKS.register_module()
class BestMiouCheckpointHook(Hook):
    """Maintain the exact best_miou.pth alias using validation mIoU."""

    priority = 'VERY_LOW'

    @master_only
    def after_val_epoch(self, runner, metrics=None):
        metrics = metrics or {}
        value = metrics.get('val/mIoU', metrics.get('mIoU'))
        best = runner.message_hub.get_info('best_miou', float('-inf'))
        if value is not None and value > best:
            best = float(value)
            runner.message_hub.update_info('best_miou', best, resumed=True)
            runner.save_checkpoint(
                runner.work_dir, 'best_miou.pth', save_optimizer=True,
                save_param_scheduler=True,
                meta={'best_miou': best}, by_epoch=True)

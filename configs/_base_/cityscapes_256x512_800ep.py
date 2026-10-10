"""Shared data, optimizer, schedule, runtime, logging and checkpoint policy."""

custom_imports = dict(
    imports=['src.transforms', 'src.metrics', 'src.hooks'],
    allow_failed_imports=False)

default_scope = 'mmseg'
dataset_type = 'CityscapesDataset'
data_root = '/home/igarashi_25/datasets/cityscapes'
crop_size = (256, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 256), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='TorchvisionColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 256), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CityscapesDetailedMetric', ignore_index=255, prefix='val')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(
        type='AdamW', lr=1.0e-4, betas=(0.9, 0.999),
        weight_decay=0.01))
param_scheduler = [
    dict(
        type='PolyLR', eta_min=1.0e-6, power=1.0,
        begin=0, end=800, by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=100,
        filename='epoch_{epoch:04d}.pth', save_optimizer=True,
        save_param_scheduler=True, max_keep_ckpts=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
custom_hooks = [
    dict(type='LatestCheckpointHook'),
    dict(type='BestMiouCheckpointHook'),
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(project='cityscapes-discriminative-baselines')),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
launcher = 'none'

auto_scale_lr = dict(enable=False, base_batch_size=4)

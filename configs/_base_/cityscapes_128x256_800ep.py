"""128x256 variant of the shared Cityscapes experiment policy."""

_base_ = './cityscapes_256x512_800ep.py'

crop_size = (128, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='DFMCityscapesTransform',
        size=(128, 256),
        training=True,
        flip_probability=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='DFMCityscapesTransform', size=(128, 256), training=False),
    dict(type='SetFixedEvaluationMetadata', size=(128, 256)),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(target_size=(128, 256))
test_evaluator = val_evaluator

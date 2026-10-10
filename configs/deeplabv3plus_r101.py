_base_ = './_base_/cityscapes_256x512_800ep.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c', depth=101, num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4), strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg, norm_eval=False, style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead', in_channels=2048, in_index=3,
        channels=512, dilations=(1, 12, 24, 36),
        c1_in_channels=256, c1_channels=48, dropout_ratio=0.1,
        num_classes=19, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=1.0, ignore_index=255)),
    auxiliary_head=dict(
        type='FCNHead', in_channels=1024, in_index=2, channels=256,
        num_convs=1, concat_input=False, dropout_ratio=0.1,
        num_classes=19, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.4, ignore_index=255)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

work_dir = 'results/deeplabv3plus_r101_256x512_800ep'
val_evaluator = dict(output_dir=f'{work_dir}/validation')
test_evaluator = dict(output_dir=f'{work_dir}/evaluation')
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='cityscapes-discriminative-baselines',
        name='deeplabv3plus_r101_256x512_800ep')),
])

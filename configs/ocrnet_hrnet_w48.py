_base_ = './_base_/cityscapes_256x512_800ep.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
channels = [48, 96, 192, 384]
data_preprocessor = dict(
    type='SegDataPreProcessor', mean=None, std=None, bgr_to_rgb=False,
    pad_val=0, seg_pad_val=255, size=(256, 512))
model = dict(
    type='CascadeEncoderDecoder', data_preprocessor=data_preprocessor,
    num_stages=2, pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNet', norm_cfg=norm_cfg, norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1, num_branches=1, block='BOTTLENECK',
                num_blocks=(4,), num_channels=(64,)),
            stage2=dict(
                num_modules=1, num_branches=2, block='BASIC',
                num_blocks=(4, 4), num_channels=(48, 96)),
            stage3=dict(
                num_modules=4, num_branches=3, block='BASIC',
                num_blocks=(4, 4, 4), num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3, num_branches=4, block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead', in_channels=channels, channels=sum(channels),
            in_index=(0, 1, 2, 3), input_transform='resize_concat',
            kernel_size=1, num_convs=1, concat_input=False,
            dropout_ratio=-1, num_classes=19, norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False,
                loss_weight=0.4, ignore_index=255)),
        dict(
            type='OCRHead', in_channels=channels,
            in_index=(0, 1, 2, 3), input_transform='resize_concat',
            channels=512, ocr_channels=256, dropout_ratio=-1,
            num_classes=19, norm_cfg=norm_cfg, align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False,
                loss_weight=1.0, ignore_index=255)),
    ],
    train_cfg=dict(), test_cfg=dict(mode='whole'))

work_dir = 'results/ocrnet_hrnet_w48_256x512_800ep'
val_evaluator = dict(output_dir=f'{work_dir}/validation')
test_evaluator = dict(output_dir=f'{work_dir}/evaluation')
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='cityscapes-discriminative-baselines',
        name='ocrnet_hrnet_w48_256x512_800ep')),
])

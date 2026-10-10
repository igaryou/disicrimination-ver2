_base_ = './_base_/cityscapes_256x512_800ep.py'

checkpoint = ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
              'segformer/mit_b5_20220624-658746d9.pth')
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
    pad_val=0, seg_pad_val=255, size=(256, 512))
model = dict(
    type='EncoderDecoder', data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer', in_channels=3, embed_dims=64,
        num_stages=4, num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8], patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1], out_indices=(0, 1, 2, 3),
        mlp_ratio=4, qkv_bias=True, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='SegformerHead', in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3], channels=256, dropout_ratio=0.1,
        num_classes=19, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=1.0, ignore_index=255)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

work_dir = 'results/segformer_mit_b5_256x512_800ep'
val_evaluator = dict(output_dir=f'{work_dir}/validation')
test_evaluator = dict(output_dir=f'{work_dir}/evaluation')
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='cityscapes-discriminative-baselines',
        name='segformer_mit_b5_256x512_800ep')),
])

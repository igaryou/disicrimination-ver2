_base_ = './_base_/cityscapes_256x512_800ep.py'

checkpoint = ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
              'swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth')
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor', mean=None, std=None, bgr_to_rgb=False,
    pad_val=0, seg_pad_val=255, size=(256, 512))
model = dict(
    type='EncoderDecoder', data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer', pretrain_img_size=224, embed_dims=96,
        patch_size=4, window_size=7, mlp_ratio=4,
        depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3),
        qkv_bias=True, qk_scale=None, patch_norm=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.3,
        use_abs_pos_embed=False, act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='UPerHead', in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
        channels=512, dropout_ratio=0.1, num_classes=19,
        norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=1.0, ignore_index=255)),
    auxiliary_head=dict(
        type='FCNHead', in_channels=384, in_index=2, channels=256,
        num_convs=1, concat_input=False, dropout_ratio=0.1,
        num_classes=19, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.4, ignore_index=255)),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

work_dir = 'results/upernet_swin_s_256x512_800ep'
val_evaluator = dict(output_dir=f'{work_dir}/validation')
test_evaluator = dict(output_dir=f'{work_dir}/evaluation')
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='cityscapes-discriminative-baselines',
        name='upernet_swin_s_256x512_800ep')),
])

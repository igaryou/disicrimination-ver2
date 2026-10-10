_base_ = './_base_/cityscapes_256x512_800ep.py'

checkpoint = ('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
              'swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth')
num_classes = 19
data_preprocessor = dict(
    type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
    pad_val=0, seg_pad_val=255, size=(256, 512),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder', data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer', pretrain_img_size=224, embed_dims=96,
        patch_size=4, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4,
        qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.3, patch_norm=True,
        out_indices=(0, 1, 2, 3), with_cp=False, frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768], strides=[4, 8, 16, 32],
        feat_channels=256, out_channels=256, num_classes=num_classes,
        num_queries=100, num_transformer_feat_level=3,
        align_corners=False, ignore_index=255,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder', num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3,
                        num_points=4, im2col_step=64, dropout=0.0,
                        batch_first=True, norm_cfg=None, init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True, num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0,
                    proj_drop=0.0, dropout_layer=None, batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0,
                    proj_drop=0.0, dropout_layer=None, batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0,
                    dropout_layer=None, add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=2.0, reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True,
            reduction='mean', loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss', use_sigmoid=True, activate=True,
            reduction='mean', naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(
            num_points=12544, oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0,
                         use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0,
                         pred_act=True, eps=1.0),
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(), test_cfg=dict(mode='whole'))

work_dir = 'results/mask2former_swin_s_256x512_800ep'
val_evaluator = dict(output_dir=f'{work_dir}/validation')
test_evaluator = dict(output_dir=f'{work_dir}/evaluation')
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='cityscapes-discriminative-baselines',
        name='mask2former_swin_s_256x512_800ep')),
])

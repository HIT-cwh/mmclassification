from mmcv import ConfigDict

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='T2T_ViT',
        t2t_module=dict(
            img_size=224,
            tokens_type='transformer',
            in_chans=3,
            embed_dim=384,
            token_dim=64),
        encoder=ConfigDict(
            type='T2TTransformerEncoder',
            num_layers=14,
            transformerlayers=dict(
                type='T2TTransformerEncoderLayer',
                attn_cfgs=ConfigDict(
                    type='T2TBlockAttention',
                    embed_dims=384,
                    num_heads=6,
                    attn_drop=0.,
                    proj_drop=0.,
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                ffn_cfgs=dict(
                    embed_dims=384,
                    feedforward_channels=3 * 384,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'))),
        drop_path_rate=0.1),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
        topk=(1, 5),
    ),
    train_cfg=dict(mixup=dict(alpha=0.2, num_classes=1000)))

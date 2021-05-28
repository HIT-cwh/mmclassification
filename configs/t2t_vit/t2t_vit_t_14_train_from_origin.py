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
        encoder=dict(
            type='T2TTransformerEncoder',
            num_layers=14,
            transformerlayers=dict(
                type='T2TTransformerEncoderLayer',
                attn_cfgs=dict(
                    type='T2TBlockAttention',
                    embed_dims=384,
                    num_heads=6,
                    attn_drop=0.,
                    proj_drop=0.,
                    dropout_layer=dict(type='DropPath')),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=384,
                    feedforward_channels=3 * 384,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath')),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                batch_first=True),
            drop_path_rate=0.1),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1.)
        ]),
    neck=None,
    head=dict(
        type='T2THead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
        topk=(1, 5),
    ),
    train_cfg=dict(
        cutmixup=dict(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            num_classes=1000)))

# pipeline
policies = [
    [
        dict(type='Posterize', bits=4, prob=0.4),
        dict(type='Rotate', angle=30., prob=0.6)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.5)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
    [
        dict(type='Posterize', bits=5, prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 6, prob=0.6),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Posterize', bits=6, prob=0.8),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='Rotate', angle=10., prob=0.2),
        dict(type='Solarize', thr=256 / 9, prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.6),
        dict(type='Posterize', bits=5, prob=0.6)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0., prob=0.4)
    ],
    [
        dict(type='Rotate', angle=30., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [dict(type='Equalize', prob=0.0),
     dict(type='Equalize', prob=0.8)],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [
        dict(type='Rotate', angle=30 / 9 * 8, prob=0.8),
        dict(type='ColorTransform', magnitude=0.2, prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0.8, prob=0.8),
        dict(type='Solarize', thr=256 / 9 * 2, prob=0.8)
    ],
    [
        dict(type='Sharpness', magnitude=0.7, prob=0.4),
        dict(type='Invert', prob=0.6)
    ],
    [
        dict(
            type='Shear',
            magnitude=0.3 / 9 * 5,
            prob=0.6,
            direction='horizontal'),
        dict(type='Equalize', prob=1.)
    ],
    [
        dict(type='ColorTransform', magnitude=0., prob=0.4),
        dict(type='Equalize', prob=0.6)
    ],
    [
        dict(type='Equalize', prob=0.4),
        dict(type='Solarize', thr=256 / 9 * 5, prob=0.2)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.6)
    ],
    [dict(type='Invert', prob=0.6),
     dict(type='Equalize', prob=1.)],
    [
        dict(type='ColorTransform', magnitude=0.4, prob=0.6),
        dict(type='Contrast', magnitude=0.8, prob=1.)
    ],
    [dict(type='Equalize', prob=0.8),
     dict(type='Equalize', prob=0.6)],
]
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='memcached',
            server_list_cfg='/mnt/lustre/share/memcached_client'
            '/server_list.conf',
            client_cfg='/mnt/lustre/share/memcached_client/client.conf',
            sys_path='/mnt/lustre/share/pymc/py3')),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='AutoAugment', policies=policies),
    dict(type='RandomErasing', max_area_ratio=1 / 3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(248, -1),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
fp16 = dict()

paramwise_cfg = dict(custom_keys={'.backbone.cls_token': dict(decay_mult=0.0)})
# learning policy
# FIXME: lr in the first 300 epochs conforms to the CosineAnnealing and
#  the lr in the last 10 epoch equals to min_lr
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=310)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='EMAHook', momentum=0.00004)]  # warm_up may be 0

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoint/t2t/t2t_vit_t_14_origin.pth'
resume_from = None
workflow = [('train', 1)]

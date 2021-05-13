# accuracy_top-1 : 81.39 accuracy_top-5 : 95.73
_base_ = [
    '../_base_/models/t2t_vit_t_14.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

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

dataset_type = 'ImageNet'
data = dict(
    samples_per_gpu=64, workers_per_gpu=2, test=dict(pipeline=test_pipeline))

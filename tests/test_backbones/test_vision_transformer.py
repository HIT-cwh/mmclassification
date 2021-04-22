import pytest
import torch
from mmcv import Config
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import VGG, VisionTransformer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_vit_backbone():

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = VisionTransformer()
        model.init_weights(pretrained=0)

    model = dict(
        embed_dim=768,
        img_size=224,
        patch_size=16,
        in_channels=3,
        drop_rate=0.,
        hybrid_backbone=None,
        norm_cfg=dict(type='LN'),
        encoder=dict(
            type='TransformerEncoder',
            num_layers=12,
            transformerlayers=dict(
                type='VitTransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=768,
                        num_heads=12,
                        dropout=0.1)
                ],
                feedforward_channels=3072,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'ffn')),
            init_cfg=[
                dict(type='Xavier', layer='Linear', distribution='normal')
            ]),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ])
    cfg = Config(model)

    # Test ViT base model with input size of 224
    # and patch size of 16
    model = VisionTransformer(**cfg)
    print(model)
    model.init_weights()
    # print(model.encoder.layers[0].ffns[0]._is_init)
    # print(model.encoder.layers[0].ffns[0].layers[0][0].bias.std())
    print(model.state_dict().keys())
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((3, 768))


def test_vit_hybrid_backbone():

    # Test VGG11+ViT-B/16 hybrid model
    backbone = VGG(11, norm_eval=True)
    backbone.init_weights()

    model = dict(
        embed_dim=768,
        img_size=224,
        patch_size=16,
        in_channels=3,
        drop_rate=0.,
        hybrid_backbone=backbone,
        norm_cfg=dict(type='LN'),
        encoder=dict(
            type='TransformerEncoder',
            num_layers=12,
            transformerlayers=dict(
                type='TransformerEncoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=768,
                        num_heads=12,
                        dropout=0.1)
                ],
                feedforward_channels=3072,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'ffn'))))
    cfg = Config(model)

    model = VisionTransformer(**cfg)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((1, 768))

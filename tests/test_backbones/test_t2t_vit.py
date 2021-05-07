# import pytest
# import torch
# from mmcv import Config
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

# from mmcls.models.backbones import VGG, VisionTransformer


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


def test_t2t_backbone():

    transformerlayers = dict(
        type='TokenTransformerLayer',
        attn_cfgs=[
            dict(
                type='TokenTransformerAttention',
                embed_dims=768,
                num_heads=12,
                attn_drop=0.,
                dropout_layer=dict(type='DropOut', drop_prob=0.1))
        ],
        ffn_cfgs=dict(
            embed_dims=768,
            feedforward_channels=3072,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='GELU')),
        operation_order=('norm', 'self_attn', 'norm', 'ffn'))
    print(transformerlayers)

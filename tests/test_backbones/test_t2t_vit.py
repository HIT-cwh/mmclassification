import pytest
import torch
from mmcv import Config, ConfigDict
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import T2T_ViT


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

    model_cfg = dict(
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
            drop_path_rate=0.1))
    cfg = Config(model_cfg)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = T2T_ViT(**cfg)
        model.init_weights(pretrained=0)

    model = T2T_ViT(**cfg)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.rand(3, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((3, 384))

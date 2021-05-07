import copy

import numpy as np
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_dropout,
                                         build_transformer_layer)
from mmcv.runner.base_module import BaseModule, ModuleList

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES


@ATTENTION.register_module()
class TokenTransformerAttention(BaseModule):

    def __init__(self,
                 dim,
                 num_heads=8,
                 embed_dims=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_dropout_layer=dict(type='DropOut', drop_prob=0.),
                 proj_dropout_layer=dict(type='DropOut', drop_prob=0.),
                 init_cfg=None):
        super(TokenTransformerAttention, self).__init__(init_cfg)
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = Linear(dim, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = build_dropout(
            attn_dropout_layer) if attn_dropout_layer else nn.Identity()
        self.proj = Linear(embed_dims, embed_dims)
        self.proj_drop = build_dropout(
            proj_dropout_layer) if proj_dropout_layer else nn.Identity()

    def forward(self, query, key, value, *args, **kwargs):
        assert \
            (query is key or torch.equal(query, key)) and \
            (key is value or torch.equal(key, value)), \
            'In self-attn, query == key == value should be satistied.'
        B, N, C = query.shape

        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads,
                                      self.embed_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        # because the original x has different size with current x,
        # use v to do skip connection
        x = v.squeeze(1) + x

        return x


@TRANSFORMER_LAYER.register_module()
class TokenTransformerLayer(BaseTransformerLayer):

    def __init__(self,
                 attn_cfgs=None,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 *args,
                 **kwargs):
        super(TokenTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            operation_order=operation_order,
            *args,
            **kwargs)

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for i in range(num_norms):
            if i == 0:
                self.norms.append(
                    build_norm_layer(norm_cfg, attn_cfgs['dim'])[1])
            else:
                self.norms.append(
                    build_norm_layer(norm_cfg, attn_cfgs['embed_dims'])[1])

    def forward(self, *args, **kwargs):
        x = super(TokenTransformerLayer, self).forward(*args, **kwargs)
        return x


class T2T_module(BaseModule):
    """Tokens-to-Token encoding module."""

    def __init__(self,
                 img_size=224,
                 tokens_type='transformer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64,
                 init_cfg=None):
        super(T2T_module, self).__init__(init_cfg)

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            tokentransformerlayer1 = dict(
                type='TokenTransformerLayer',
                attn_cfgs=ConfigDict(
                    type='TokenTransformerAttention',
                    dim=in_chans * 7 * 7,
                    embed_dims=token_dim,
                    num_heads=1),
                ffn_cfgs=dict(
                    embed_dims=token_dim,
                    feedforward_channels=token_dim,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'))
            self.attention1 = build_transformer_layer(tokentransformerlayer1)
            tokentransformerlayer2 = copy.deepcopy(tokentransformerlayer1)
            tokentransformerlayer2['attn_cfgs']['dim'] = token_dim * 3 * 3
            self.attention2 = build_transformer_layer(tokentransformerlayer2)

            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        # there are 3 soft split, stride are 4,2,2 seperately
        self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2))

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(query=x, key=None, value=None)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)),
                                      int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(query=x, key=None, value=None)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)),
                                      int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


@TRANSFORMER_LAYER.register_module()
class T2TTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in Tokens-to-Token vision transformer."""

    def __init__(self, *args, **kwargs):
        super(T2TTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class T2TTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of Tokens-to-Token vision transformer.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
            self,
            *args,
            coder_norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(T2TTransformerEncoder, self).__init__(*args, **kwargs)
        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(
                coder_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify coder_norm_cfg'
            self.coder_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(T2TTransformerEncoder, self).forward(*args, **kwargs)
        if self.coder_norm is not None:
            x = self.coder_norm(x)
        return x


@BACKBONES.register_module()
class T2T_ViT(BaseBackbone):

    def __init__(self,
                 t2t_module=dict(
                     img_size=224,
                     tokens_type='transformer',
                     in_chans=3,
                     embed_dim=768,
                     token_dim=64),
                 encoder=dict(
                     type='T2TTransformerEncoder',
                     transformerlayers=None,
                     num_encoder_layers=12,
                     coder_norm_cfg=None,
                 ),
                 init_cfg=None):
        super(T2T_ViT, self).__init__(init_cfg)


if __name__ == '__main__':
    import torch
    t2t = T2T_module()
    imgs = torch.rand(1, 3, 224, 224)
    print(t2t(imgs).shape)

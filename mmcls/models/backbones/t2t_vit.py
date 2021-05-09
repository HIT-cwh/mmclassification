import copy

import numpy as np
import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import (ATTENTION, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_dropout,
                                         build_positional_encoding,
                                         build_transformer_layer,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@ATTENTION.register_module()
class T2TModuleAttention(BaseModule):

    def __init__(self,
                 in_dim,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):
        super(T2TModuleAttention, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(in_dim, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

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


@ATTENTION.register_module()
class T2TBlockAttention(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super(T2TBlockAttention, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.init_cfg = init_cfg

    def forward(self, query, key, value, residual=None, *args, **kwargs):
        assert \
            (query is key or torch.equal(query, key)) and \
            (key is value or torch.equal(key, value)), \
            'In self-attn, query == key == value should be satistied.'

        if residual is None:
            residual = query

        B, N, C = query.shape
        qkv = self.qkv(query).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(
                                          2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # out = out.permute(1, 0, 2)

        return residual + self.dropout_layer(out)


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
                    build_norm_layer(norm_cfg, attn_cfgs['in_dim'])[1])
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

        self.embed_dim = embed_dim

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(
                kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(
                kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            tokentransformer_layer1 = dict(
                type='TokenTransformerLayer',
                attn_cfgs=ConfigDict(
                    type='T2TModuleAttention',
                    in_dim=in_chans * 7 * 7,
                    embed_dims=token_dim,
                    num_heads=1),
                ffn_cfgs=dict(
                    embed_dims=token_dim,
                    feedforward_channels=token_dim,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'))
            self.attention1 = build_transformer_layer(tokentransformer_layer1)
            tokentransformer_layer2 = copy.deepcopy(tokentransformer_layer1)
            tokentransformer_layer2['attn_cfgs']['in_dim'] = token_dim * 3 * 3
            self.attention2 = build_transformer_layer(tokentransformer_layer2)

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

    def __init__(self, qkv_bias=False, *args, **kwargs):
        super(T2TTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])

        # print(self.attentions[0].attn.in_proj_bias.shape)
        # self.attentions[0].attn.in_proj_bias = None


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


@POSITIONAL_ENCODING.register_module()
class SinusoidEncoding(object):

    def __init__(self):
        super(SinusoidEncoding, self).__init__()

    def __call__(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)


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
                     num_layers=12,
                     coder_norm_cfg=None),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_cfg=None):
        super(T2T_ViT, self).__init__(init_cfg)

        self.tokens_to_token = T2T_module(**t2t_module)
        num_patches = self.tokens_to_token.num_patches
        embed_dim = self.tokens_to_token.embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        sinusoid_encoding = build_positional_encoding(
            dict(type='SinusoidEncoding'))
        self.pos_embed = nn.Parameter(
            data=sinusoid_encoding(
                n_position=num_patches + 1, d_hid=embed_dim),
            requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if drop_path_rate:
            assert encoder['transformerlayers']['attn_cfgs']['dropout_layer'][
                       'type'] == 'DropPath' and \
                   encoder['transformerlayers']['ffn_cfgs']['dropout_layer'][
                       'type'] == 'DropPath'
            depth = encoder['num_layers']
            # stochastic depth decay rule
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            transformerlayers = [
                copy.deepcopy(encoder['transformerlayers'])
                for _ in range(depth)
            ]
            for i in range(depth):
                transformerlayers[i]['attn_cfgs']['dropout_layer'][
                    'drop_prob'] = dpr[i]
            encoder['transformerlayers'] = transformerlayers

        self.encoder = build_transformer_layer_sequence(encoder)

    def forward(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(query=x, key=None, value=None)

        return x[:, 0]

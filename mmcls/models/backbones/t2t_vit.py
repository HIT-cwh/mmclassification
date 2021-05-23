import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import (ATTENTION, POSITIONAL_ENCODING,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer, DropPath,
                                         TransformerLayerSequence,
                                         build_dropout,
                                         build_positional_encoding,
                                         build_transformer_layer,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@ATTENTION.register_module()
class TokenTransformerAttention(BaseModule):
    """MultiHead self-attention in T2T Transformer.

    Args:
        in_dim (int): Dimension of input tokens.
        embed_dims (int): Embedding dimension
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        qkv_bias (bool): Add bias as qkv Linear module parameter.
            Default: False.
        qk_scale (float, optional): scale of the dot products. Default: None.
        attn_drop (float): A Dropout layer on attn output weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer on out. Default: 0.0.
        init_cfg (dict, optional): Initialization config dict.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Add batch_first
            to synchronize with MultiheadAttention in transformer.py mmcv.
            batch_first should be True in T2TModuleAttention.
    """

    def __init__(self,
                 in_dim,
                 embed_dims,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None,
                 batch_first=True):
        super(TokenTransformerAttention, self).__init__(init_cfg)
        assert batch_first is True, \
            'batch_first should be True when using T2TModuleAttention'
        self.batch_first = batch_first
        assert embed_dims % num_heads == 0, \
            'embed_dims should be divisible by num_heads.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims / num_heads
        self.scale = qk_scale or (in_dim // num_heads)**-0.5

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
                                      self.head_dims).permute(2, 0, 3, 1, 4)
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
    """MultiHead self-attention in T2T-ViT backbone.

    Args:
        embed_dims (int): Embedding dimension
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        qkv_bias (bool): Add bias as qkv Linear module parameter.
            Default: False.
        qk_scale (float, optional): scale of the dot products. Default: None.
        attn_drop (float): A Dropout layer on attn output weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer on out. Default: 0.0.
        dropout_layer (dict): The dropout_layer used when adding the shortcut.
        init_cfg (dict, optional): Initialization config dict.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim) or (n, batch, embed_dim). Add batch_first
            to synchronize with MultiheadAttention in transformer.py mmcv.
            batch_first should be True in T2TBlockAttention.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True):
        super(T2TBlockAttention, self).__init__(init_cfg)
        assert batch_first is True, \
            'batch_first should be True when using T2TBlockAttention'
        self.batch_first = batch_first
        assert embed_dims % num_heads == 0, \
            'embed_dims should be divisible by num_heads.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        self.scale = qk_scale or self.head_dims**-0.5

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
                                      self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return residual + self.dropout_layer(out)


@TRANSFORMER_LAYER.register_module()
class TokenTransformerLayer(BaseTransformerLayer):
    """Take the standard Transformer as T2T Transformer."""

    def __init__(self,
                 norm_cfg=dict(type='LN'),
                 attn_cfgs=None,
                 *args,
                 **kwargs):
        super(TokenTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs, *args, **kwargs)

        self.norms = ModuleList()
        num_norms = self.operation_order.count('norm')
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


@ATTENTION.register_module()
class TokenPerformerAttention(BaseModule):

    def __init__(self,
                 in_dim,
                 embed_dims,
                 num_heads=1,
                 kernel_ratio=0.5,
                 proj_drop=0.1,
                 init_cfg=None):
        super(TokenPerformerAttention, self).__init__(init_cfg=init_cfg)
        assert embed_dims % num_heads == 0, \
            'embed_dims should be divisible by num_heads.'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        self.qkv = nn.Linear(in_dim, embed_dims * 3)
        self.proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.epsilon = 1e-8  # for stable in division

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(
            nn.init.orthogonal_(self.w) * math.sqrt(self.m),
            requires_grad=False)

    def performer_exp(self, x):
        # part of the function is borrow from
        # https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def forward(self, x):
        q, k, v = torch.split(self.qkv(x), self.emb, dim=-1)

        # (B, T, m), (B, T, m)
        kp, qp = self.performer_exp(k), self.performer_exp(q)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(
            dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / \
            (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # same as token_transformer in T2T layer, use v as skip connection
        y = v + self.dp(self.proj(y))

        return y


# @TRANSFORMER_LAYER.register_module()
# class TokenPerformerLayer(BaseModule):
#     """Take Performer as T2T Transformer."""
#
#     def __init__(self,
#                  in_dim,
#                  embed_dims,
#                  num_heads=1,
#                  kernel_ratio=0.5,
#                  proj_drop=0.1,
#                  ffn_drop=0.1,
#                  init_cfg=None):
#         super(TokenPerformerLayer, self).__init__(init_cfg=init_cfg)
#         assert embed_dims % num_heads == 0, \
#             'embed_dims should be divisible by num_heads.'
#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#         self.head_dims = embed_dims // num_heads
#
#         self.qkv = nn.Linear(in_dim, embed_dims * 3)
#         self.proj = nn.Linear(self.embed_dims, self.embed_dims)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(self.emb)


class T2T_module(BaseModule):
    """Tokens-to-Token module.

    A layer-wise “Tokens-to-Token module” (T2T_module) to model the local
    structure information of images and reduce the length of tokens
    progressively.

    Args:
        img_size (int): Input image size
        tokens_type (str): Transformer type used in T2T_module,
            transformer or performer.
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
        token_dim (int): Tokens dimension in T2TModuleAttention.
            To overcome the limitations, in T2T module, the channel dimension
            of T2T layer is set small (32 or 64) to reduce MACs
        init_cfg (dict, optional): Initialization config dict.
    """

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
                    type='TokenTransformerAttention',
                    in_dim=in_chans * 7 * 7,
                    embed_dims=token_dim,
                    num_heads=1),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=token_dim,
                    feedforward_channels=token_dim,
                    num_fcs=2,
                    act_cfg=dict(type='GELU'),
                    dropout_layer=dict(type='DropPath', drop_prob=0.)),
                operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                batch_first=True)
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
    """Implements transformer layer in T2T-ViT backbone."""

    def __init__(self, *args, **kwargs):
        super(T2TTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class T2TTransformerEncoder(TransformerLayerSequence):
    """Transformer layers of T2T-ViT backbone.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
        drop_path_rate (float): Drop path probability of a drop path layer.
            The drop path probabilities in encoder layers are evenly spaced
            from 0 to drop_path_rate, inclusive. Default: 0.0
    """

    def __init__(
            self,
            coder_norm_cfg=dict(type='LN'),
            drop_path_rate=0.,
            *args,
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

        self.drop_path_rate = drop_path_rate
        if drop_path_rate:
            self.set_droppath_rate()

    def set_droppath_rate(self):
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, self.num_layers)
        ]
        for i, layer in enumerate(self.layers):
            for module in layer.modules():
                if isinstance(module, DropPath):
                    module.drop_prob = dpr[i]

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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official
    # releases - RW
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        cdfl = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * cdfl - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


@BACKBONES.register_module()
class T2T_ViT(BaseBackbone):
    """Tokens-to-Token Vision Transformers (T2T-ViT)

    A PyTorch impl of : `Tokens-to-Token ViT: Training Vision Transformers
    from Scratch on ImageNet` - https://arxiv.org/abs/2101.11986

    Args:
        t2t_module (dict): Config of Tokens-to-Token module
        encoder (dict): Config of T2T-ViT backbone
        drop_rate (float): Probability of an element to be zeroed. Default 0.0.
        init_cfg (dict, optional): Initialization config dict.
    """

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
                     coder_norm_cfg=None,
                     drop_path_rate=0.),
                 drop_rate=0.,
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

        self.encoder = build_transformer_layer_sequence(encoder)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    # TODO: wait for https://github.com/open-mmlab/mmcv/pull/935 merge
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(query=x, key=None, value=None)

        return x[:, 0]

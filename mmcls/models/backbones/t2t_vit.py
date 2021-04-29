# import numpy as np
# import torch
# import torch.nn as nn
# from mmcv.runner.base_module import BaseModule
#
# from ..builder import BACKBONES
# from .base_backbone import BaseBackbone
#
#
# class Token_transformer(nn.Module):
#
#     def __init__(self,
#                  dim,
#                  in_dim,
#                  num_heads,
#                  mlp_ratio=1.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             in_dim=in_dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop)
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(in_dim)
#         self.mlp = Mlp(
#             in_features=in_dim,
#             hidden_features=int(in_dim * mlp_ratio),
#             out_features=in_dim,
#             act_layer=act_layer,
#             drop=drop)
#
#     def forward(self, x):
#         x = self.attn(self.norm1(x))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class T2T_module(BaseModule):
#     """Tokens-to-Token encoding module."""
#
#     def __init__(self,
#                  img_size=224,
#                  tokens_type='transformer',
#                  in_chans=3,
#                  embed_dim=768,
#                  token_dim=64,
#                  init_cfg=None):
#         super(T2T_module, self).__init__(init_cfg)
#
#         if tokens_type == 'transformer':
#             print('adopt transformer encoder for tokens-to-token')
#             self.soft_split0 = nn.Unfold(
#                 kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
#             self.soft_split1 = nn.Unfold(
#                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#             self.soft_split2 = nn.Unfold(
#                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#
#             self.attention1 = Token_transformer(
#                 dim=in_chans * 7 * 7,
#                 in_dim=token_dim,
#                 num_heads=1,
#                 mlp_ratio=1.0)
#             self.attention2 = Token_transformer(
#                 dim=token_dim * 3 * 3,
#                 in_dim=token_dim,
#                 num_heads=1,
#                 mlp_ratio=1.0)
#             self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

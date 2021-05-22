import torch.nn as nn

from ..backbones.t2t_vit import trunc_normal_
from ..builder import HEADS
from .linear_head import LinearClsHead


@HEADS.register_module()
class T2THead(LinearClsHead):

    def __init__(self, *args, **kwargs):
        super(T2THead, self).__init__(*args, **kwargs)

    def init_weights(self):
        trunc_normal_(self.fc.weight, std=.02)
        if isinstance(self.fc, nn.Linear) and self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

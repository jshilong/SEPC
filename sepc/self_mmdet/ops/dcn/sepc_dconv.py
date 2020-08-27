import torch
import torch.nn as nn
from mmdet.ops.dcn import DeformConv, deform_conv
from torch.nn.modules.utils import _pair


class sepc_conv(DeformConv):
    def __init__(self, *args, part_deform=False, **kwargs):
        super(sepc_conv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if self.part_deform:
            self.conv_offset = nn.Conv2d(self.in_channels,
                                         self.deformable_groups * 2 *
                                         self.kernel_size[0] *
                                         self.kernel_size[1],
                                         kernel_size=self.kernel_size,
                                         stride=_pair(self.stride),
                                         padding=_pair(self.padding),
                                         bias=True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(x,
                                              self.weight,
                                              bias=self.bias,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              groups=self.groups)

        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups,
                           self.deformable_groups) + self.bias.unsqueeze(
                               0).unsqueeze(-1).unsqueeze(-1)

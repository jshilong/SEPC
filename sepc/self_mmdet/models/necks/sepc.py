import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import auto_fp16
from mmdet.models.registry import NECKS
from torch.nn import init as init

from sepc.self_mmdet.ops.dcn.sepc_dconv import sepc_conv


@NECKS.register_module
class SEPC(nn.Module):
    def __init__(
        self,
        in_channels=[256] * 5,
        out_channels=256,
        num_outs=5,
        pconv_deform=False,
        lcconv_deform=False,
        iBN=False,
        Pconv_num=4,
    ):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.iBN = iBN
        self.Pconvs = nn.ModuleList()

        for i in range(Pconv_num):
            self.Pconvs.append(
                PConvModule(in_channels[i],
                            out_channels,
                            iBN=self.iBN,
                            part_deform=pconv_deform))

        self.lconv = sepc_conv(256,
                               256,
                               kernel_size=3,
                               dilation=1,
                               part_deform=lcconv_deform)
        self.cconv = sepc_conv(256,
                               256,
                               kernel_size=3,
                               dilation=1,
                               part_deform=lcconv_deform)
        self.relu = nn.ReLU()
        if self.iBN:
            self.lbn = nn.BatchNorm2d(256)
            self.cbn = nn.BatchNorm2d(256)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for str in ['l', 'c']:
            m = getattr(self, str + 'conv')
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = inputs
        for pconv in self.Pconvs:
            x = pconv(x)
        cls = [self.cconv(level, item) for level, item in enumerate(x)]
        loc = [self.lconv(level, item) for level, item in enumerate(x)]
        if self.iBN:
            cls = iBN(cls, self.cbn)
            loc = iBN(loc, self.lbn)
        outs = [[self.relu(s), self.relu(l)] for s, l in zip(cls, loc)]
        return tuple(outs)


class PConvModule(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        kernel_size=[3, 3, 3],
        dilation=[1, 1, 1],
        groups=[1, 1, 1],
        iBN=False,
        part_deform=False,
    ):
        super(PConvModule, self).__init__()

        #     assert not (bias and iBN)
        self.iBN = iBN
        self.Pconv = nn.ModuleList()
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[0],
                      dilation=dilation[0],
                      groups=groups[0],
                      padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,
                      part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[1],
                      dilation=dilation[1],
                      groups=groups[1],
                      padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2,
                      part_deform=part_deform))
        self.Pconv.append(
            sepc_conv(in_channels,
                      out_channels,
                      kernel_size=kernel_size[2],
                      dilation=dilation[2],
                      groups=groups[2],
                      padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                      stride=2,
                      part_deform=part_deform))

        if self.iBN:
            self.bn = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.Pconv:
            init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):

            temp_fea = self.Pconv[1](level, feature)
            if level > 0:
                temp_fea += self.Pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.upsample_bilinear(
                    self.Pconv[0](level, x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3)])
            next_x.append(temp_fea)
        if self.iBN:
            next_x = iBN(next_x, self.bn)
        next_x = [self.relu(item) for item in next_x]
        return next_x


def iBN(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]

import torch
import torch.nn as nn

from models.base_block import SameBlock2d, DownBlock2d, UpBlock2d, ResDownBlock2d
from models.base_block import UpBlock2d, ResUpBlock2d, AttnUpBlock2d, ResAttnUpBlock2d, QCUpBlock2d


class BaseDecoder(nn.Module):
    def __init__(self, in_channels, num_up_blocks):
        super(BaseDecoder, self).__init__()
        blocks = []
        self.in_channels = in_channels
        self.num_up_blocks = num_up_blocks
        out_channels = in_channels // 2
        for i in range(num_up_blocks):
            blocks.append(UpBlock2d(in_channels, out_channels))
            in_channels = out_channels
            out_channels = in_channels // 2
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(in_channels, 3, kernel_size=7, padding=3)

    def forward(self, x):
        for i in range(self.num_up_blocks):
            x = self.blocks[i](x)
        x = self.final(x)
        x = nn.functional.sigmoid(x)
        return x


class ResDecoder(nn.Module):
    def __init__(self, in_channels, num_up_blocks, repeat):
        super(ResDecoder, self).__init__()
        up_blocks = []
        out_channels = in_channels // 2
        for i in range(num_up_blocks):
            up_blocks.append(ResUpBlock2d(in_channels, out_channels, repeat=repeat, dropout=0.1,
                                              norm_type='group', activate_type='relu', up_type='conv'))
            in_channels = out_channels
            out_channels = in_channels // 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(in_channels, 3, kernel_size=7, padding=3)

    def forward(self, x):
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)

        x = self.final(x)
        x = nn.functional.sigmoid(x)
        return x


class AttnDecoder(nn.Module):
    def __init__(self, in_channels, num_up_blocks, repeat):
        super(AttnDecoder, self).__init__()
        up_blocks = []
        out_channels = in_channels // 2
        for i in range(num_up_blocks):
            up_blocks.append(AttnUpBlock2d(in_channels, out_channels, repeat=repeat,
                                            norm_type='group', up_type='conv'))
            in_channels = out_channels
            out_channels = in_channels // 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(in_channels, 3, kernel_size=7, padding=3)

    def forward(self, x):
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)

        x = self.final(x)
        x = nn.functional.sigmoid(x)
        return x


class ResAttnDecoder(nn.Module):
    def __init__(self, in_channels, num_up_blocks, repeat):
        super(ResAttnDecoder, self).__init__()
        up_blocks = []
        out_channels = in_channels // 2
        for i in range(num_up_blocks):
            up_blocks.append(ResAttnUpBlock2d(in_channels, out_channels, repeat=repeat, res_repeat=2, attn_repeat=1,
                                            norm_type='group', up_type='conv'))
            in_channels = out_channels
            out_channels = in_channels // 2
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(in_channels, 3, kernel_size=7, padding=3)

    def forward(self, x):
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)

        x = self.final(x)
        x = nn.functional.sigmoid(x)
        return x


class PoseAwareDecoder(nn.Module):
    def __init__(self, in_channels, num_up_blocks, kp_nums, proj_nums=3, activate_type='relu', up_type='conv'):
        super(PoseAwareDecoder, self).__init__()
        self.num_up_blocks = num_up_blocks
        out_channels = in_channels // 2
        self.blocks = []
        for i in range(num_up_blocks):
            self.blocks.append(QCUpBlock2d(in_channels, out_channels, kp_nums, proj_nums, activate_type=activate_type,
                                      up_type=up_type))
            in_channels = out_channels
            out_channels = in_channels // 2

        self.final = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, feature_maps, sparse_motions):
        for i in range(self.num_up_blocks):
            feature_maps = self.blocks[i](feature_maps, sparse_motions)
        x = self.final(feature_maps)
        x = nn.functional.sigmoid(x)
        return x




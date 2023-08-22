import torch
import torch.nn as nn
import torch.nn.functional as F
# from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from models.base_block import SameBlock2d, DownBlock2d, UpBlock2d, ResDownBlock2d
from models.base_block import DownBlock2d, ResDownBlock2d, AttnDownBlock2d, ResAttnDownBlock2d


class BaseEncoder(nn.Module):
    def __init__(self, num_base_channels, num_down_blocks):
        super(BaseEncoder, self).__init__()
        self.first = SameBlock2d(3, num_base_channels, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        in_channel = num_base_channels
        for i in range(num_down_blocks):
            down_blocks.append(DownBlock2d(in_channel, in_channel * 2))
            in_channel = in_channel * 2
        self.down_blocks = nn.ModuleList(down_blocks)

    # x is image with 3 * H * W
    def forward(self, x):
        x = self.first(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        return x


class ResEncoder(nn.Module):
    def __init__(self, num_base_channels, num_down_blocks, repeat):
        super(ResEncoder, self).__init__()
        self.first = SameBlock2d(3, num_base_channels, kernel_size=(3, 3), padding=1)

        down_blocks = []
        in_channel = num_base_channels
        for i in range(num_down_blocks):
            down_blocks.append(ResDownBlock2d(in_channel, in_channel * 2, repeat=repeat, dropout=0.1,
                                              norm_type='group', activate_type='relu', down_type='conv'))
            in_channel = in_channel * 2
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        x = self.first(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        return x


class AttnEncoder(nn.Module):
    def __init__(self, num_base_channels, num_down_blocks, repeat):
        super(AttnEncoder, self).__init__()
        self.first = SameBlock2d(3, num_base_channels, kernel_size=3, padding=1)

        down_blocks = []
        in_channel = num_base_channels
        for i in range(num_down_blocks):
            down_blocks.append(AttnDownBlock2d(in_channel, in_channel * 2, repeat=repeat,
                                              norm_type='group', down_type='conv'))
            in_channel = in_channel * 2
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        x = self.first(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        return x


class ResAttnEncoder(nn.Module):
    def __init__(self, num_base_channels, num_down_blocks, repeat):
        super(ResAttnEncoder, self).__init__()
        self.first = SameBlock2d(3, num_base_channels, kernel_size=3, padding=1)

        down_blocks = []
        in_channel = num_base_channels
        for i in range(num_down_blocks):
            down_blocks.append(ResAttnDownBlock2d(in_channel, in_channel * 2, res_repeat=2,
                                                  attn_repeat=2, repeat=repeat,
                                              norm_type='group', down_type='conv'))
            in_channel = in_channel * 2
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        x = self.first(x)
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x)
        return x










import os
import time

from models.base_block import ResBlock2d, Normalize, Activate
import torch
import torch.nn as nn



def dwt(x, data_format='channels_first'):
    if data_format == 'channels_last':
        x1 = x[:, 0::2, 0::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, :]
    elif data_format == 'channels_first':
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2]
        x4 = x[:, :, 1::2, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4

    if data_format == 'channels_last':
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=-1)
    elif data_format == 'channels_first':
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)


# using dwt to replace pooling in torch
# channel_nums will plus 4
class DWT_Pooling(nn.Module):
    def __init__(self, data_format='channels_first'):
        super(DWT_Pooling, self).__init__()
        self.data_format = data_format
        self.requires_grad = False

    def forward(self, x):
        return dwt(x, self.data_format)

    def get_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*4, input_shape[2]//2, input_shape[3]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]*4)


def iwt(x, data_format='channels_first', device='cuda:0'):
    B, C, H, W = x.shape
    device = x.device
    if data_format == 'channels_last':
        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH) / 4
        x2 = (x_LL - x_LH + x_HL - x_HH) / 4
        x3 = (x_LL + x_LH - x_HL - x_HH) / 4
        x4 = (x_LL + x_LH + x_HL + x_HH) / 4

        y1 = torch.stack([x1, x3], dim=2)
        y2 = torch.stack([x2, x4], dim=2)
        shape = x.shape
        return torch.reshape(torch.cat([y1, y2], dim=-1), [shape[0], shape[1]*2, shape[2]*2, shape[3]//4]).to(device)

    elif data_format == 'channels_first':
        x_LL = x[:, 0:x.shape[1]//4, :, :]
        x_LH = x[:, x.shape[1]//4:x.shape[1]//4*2, :, :]
        x_HL = x[:, x.shape[1]//4*2:x.shape[1]//4*3, :, :]
        x_HH = x[:, x.shape[1]//4*3:, :, :]

        out = torch.zeros(B, C//4, H * 2, W * 2)

        out[:, :, 0::2, 0::2] = (x_LL - x_LH - x_HL + x_HH) / 4
        out[:, :, 1::2, 0::2] = (x_LL - x_LH + x_HL - x_HH) / 4
        out[:, :, 0::2, 1::2] = (x_LL + x_LH - x_HL - x_HH) / 4
        out[:, :, 1::2, 1::2] = (x_LL + x_LH + x_HL + x_HH) / 4

        return out.to(device)


# using dwt to replace pooling in torch
# channel_nums will plus 4
class IWT_UpSampling(nn.Module):
    def __init__(self, data_format='channels_first'):
        super(IWT_UpSampling, self).__init__()
        self.data_format = data_format

    def forward(self, x):
        return iwt(x, self.data_format)

    def get_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1]*4, input_shape[2]//2, input_shape[3]//2)
        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, input_shape[3]*4)


class WTResBlock2D(nn.Module):
    def __init__(self, in_channels, conv_number, activ_type='relu', norm_type='batch', res_type='spatial_res'):
        super(WTResBlock2D, self).__init__()
        assert res_type in ['spatial_res', 'wavelet_res']
        self.wavelet_pooling = DWT_Pooling()
        self.wavelet_upsampling = IWT_UpSampling()
        self.res_type = res_type
        blocks = []
        for i in range(conv_number):
            if norm_type is not None:
                blocks.append(Normalize(norm_type, in_channels * 4))
            if activ_type is not None:
                blocks.append(Activate(activ_type))
            blocks.append(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        if self.res_type == 'spatial_res':
            identity = x
            x = self.wavelet_pooling(x)
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = self.wavelet_upsampling(x)
            x = identity + x
        else:
            x = self.wavelet_pooling(x)
            identity = x
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = identity + x
            x = self.wavelet_upsampling(x)
        return x


# fusion_type: HF means 3/4 belongs to HF
#              LF means 3/4 belongs to LF
class WTFusionBlock2D(nn.Module):
    def __init__(self, in_channels, conv_number, activ_type='relu', norm_type='batch', fusion_type='HF'):
        super(WTFusionBlock2D, self).__init__()
        assert fusion_type in ['HF', 'LF']
        self.wavelet_pooling = DWT_Pooling()
        self.wavelet_upsampling = IWT_UpSampling()
        self.fusion_type = fusion_type
        self.in_channels = in_channels
        blocks = []
        for i in range(conv_number):
            if norm_type is not None:
                blocks.append(Normalize(norm_type, in_channels * 4))
            if activ_type is not None:
                blocks.append(Activate(activ_type))
            blocks.append(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, LF, HF):
        LF = self.wavelet_pooling(LF)
        HF = self.wavelet_pooling(HF)
        if self.fusion_type == 'HF':
            fusions = torch.cat([LF[:, :self.in_channels, :, :], HF[:, self.in_channels:, :, :]], dim=1)
        else:
            fusions = torch.cat([LF[:, :3 * self.in_channels, :, :], HF[:, 3 * self.in_channels:, :, :]], dim=1)
        for i in range(len(self.blocks)):
            fusions = self.blocks[i](fusions)
        fusions = self.wavelet_upsampling(fusions)
        return fusions


class WTFusionResBlock2D(nn.Module):
    def __init__(self, in_channels, conv_number, activ_type='relu', norm_type='batch', fusion_type='HF', res_type='spatial_res'):
        super(WTFusionResBlock2D, self).__init__()
        assert fusion_type in ['HF', 'LF']
        assert res_type in ['spatial_res', 'wavelet_res']
        self.wavelet_pooling = DWT_Pooling()
        self.wavelet_upsampling = IWT_UpSampling()
        self.fusion_type = fusion_type
        self.in_channels = in_channels
        self.res_type = res_type
        blocks = []
        for i in range(conv_number):
            if norm_type is not None:
                blocks.append(Normalize(norm_type, in_channels * 4))
            if activ_type is not None:
                blocks.append(Activate(activ_type))
            blocks.append(nn.Conv2d(in_channels * 4, in_channels * 4, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, LF, HF):
        if self.res_type == 'spatial_res':
            identity = LF
            if self.fusion_type == 'HF':
                x = torch.cat([self.wavelet_pooling(LF)[:, :self.in_channels, :, :],
                           self.wavelet_pooling(HF)[:, self.in_channels:, :, :]],
                          dim=1)
            else:
                x = torch.cat([self.wavelet_pooling(LF)[:, :3 * self.in_channels, :, :],
                           self.wavelet_pooling(HF)[:, 3 * self.in_channels:, :, :]],
                          dim=1)
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = self.wavelet_upsampling(x)
            x = identity + x
        else:
            if self.fusion_type == 'HF':
                x = torch.cat([self.wavelet_pooling(LF)[:, :self.in_channels, :, :],
                           self.wavelet_pooling(HF)[:, self.in_channels:, :, :]],
                          dim=1)
            else:
                x = torch.cat([self.wavelet_pooling(LF)[:, :3 * self.in_channels, :, :],
                           self.wavelet_pooling(HF)[:, 3 * self.in_channels:, :, :]],
                          dim=1)
            identity = x
            for i in range(len(self.blocks)):
                x = self.blocks[i](x)
            x = identity + x
            x = self.wavelet_upsampling(x)
        return x


class ResBottleneck(nn.Module):
    def __init__(self, in_channels, block_nums, dropout=0, conv_shortcut=False, norm_type='batch'):
        super(ResBottleneck, self).__init__()
        self.block_numbers = block_nums
        blocks = []
        for i in range(block_nums):
            blocks.append(ResBlock2d(in_channels, in_channels, dropout=dropout, conv_shortcut=conv_shortcut, norm_type=norm_type,
                                     activate_type='relu'))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for i in range(self.block_numbers):
            x = self.blocks[i](x)
        return x


class WTBottleneck(nn.Module):
    def __init__(self, in_channels, conv_nums, block_nums, activ_type='swish', norm_type='batch', res_type='spatial_res'):
        super(WTBottleneck, self).__init__()
        blocks = []
        for i in range(block_nums):
            blocks.append(WTResBlock2D(in_channels, conv_nums, activ_type=activ_type, norm_type=norm_type, res_type=res_type))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x


class WTFusion(nn.Module):
    def __init__(self, in_channels, conv_nums, block_nums, activ_type='swish', norm_type='batch', fusion_type='HF'):
        super(WTFusion, self).__init__()
        blocks = []
        for i in range(block_nums):
            blocks.append(WTFusionBlock2D(in_channels, conv_nums, activ_type=activ_type, norm_type=norm_type, fusion_type=fusion_type))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, LF, HF):
        for i in range(len(self.blocks)):
            LF = self.blocks[i](LF, HF)
        return LF


class WTResFusion(nn.Module):
    def __init__(self, in_channels, conv_nums, block_nums, activ_type='swish',
                 norm_type='batch', fusion_type='HF', res_type='spatial_res'):
        super(WTResFusion, self).__init__()
        blocks = []
        for i in range(block_nums):
            blocks.append(WTFusionResBlock2D(in_channels, conv_nums, activ_type=activ_type, norm_type=norm_type,
                                             fusion_type=fusion_type, res_type=res_type))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, LF, HF):
        for i in range(len(self.blocks)):
            LF = self.blocks[i](LF, HF)
        return LF
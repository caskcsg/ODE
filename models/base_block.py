import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from torchvision.transforms import Resize
from utils.wavelet import DWT_Pooling, IWT_UpSampling


def Normalize(norm_type, in_channels):
    assert norm_type in ['group', 'layer', 'batch', 'instance']
    if norm_type == 'group':
        norm_layer = nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm(in_channels, eps=1e-6)
    elif norm_type == 'batch':
        norm_layer = BatchNorm2d(in_channels, eps=1e-6)
    else:
        norm_layer = nn.InstanceNorm2d(in_channels, eps=1e-6)
    return norm_layer


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        pass

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


def Activate(activ_type):
    assert activ_type in ['relu', 'leakyrelu', 'swish']
    if activ_type == 'relu':
        activ = nn.ReLU()
    elif activ_type == 'leakyrelu':
        activ = nn.LeakyReLU(negative_slope=0.1)
    else:
        activ = Swish()
    return activ


class DownSample(nn.Module):
    def __init__(self, type, in_channels=None):
        super(DownSample, self).__init__()
        assert type in ['ave', 'max', 'conv']
        if type == 'ave':
            self.down = nn.AvgPool2d(2)
        elif type == 'max':
            self.down = nn.MaxPool2d(2)
        else:
            self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, type, in_channels):
        super(UpSample, self).__init__()
        assert type in ['nearest', 'bilinear', 'linear', 'conv']
        if type == 'conv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=type)

    def forward(self, x):
        return self.up(x)


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, conv_shortcut=False, norm_type='batch',
                 activate_type='swish'):
        super(ResBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(norm_type, in_channels)
        self.norm2 = Normalize(norm_type, out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.activ = Activate(activate_type)

        # use conv to align channels between residual and identity
        if in_channels != out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            identity = self.conv_shortcut(identity)
        return x + identity


class AttnBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super(AttnBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = Normalize(norm_type, in_channels)

        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.proj_out = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        identity = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(B, C, H*W).permute(0, 2, 1)
        k = k.reshape(B, C, H*W)
        alpha = torch.bmm(q, k)
        alpha = alpha * (C**(-0.5))
        alpha = nn.functional.softmax(alpha, dim=-1)
        v = v.reshape(B, C, H*W)
        alpha = alpha.permute(0, 2, 1)
        out = torch.bmm(v, alpha)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        if self.in_channels != self.out_channels:
            return self.conv_shortcut(identity) + out

        return identity + out


class ResAttnBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, res_repeat, attn_repeat, conv_shortcut=False,
                 dropout=None, norm_type='batch', activate_type='swish'):
        super(ResAttnBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_repeat = res_repeat
        self.attn_repeat = attn_repeat
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.norm_type = norm_type
        self.activate_type = activate_type

        blocks = []
        for i in range(self.res_repeat):
            if i == 0:
                blocks.append(ResBlock2d(in_channels, out_channels, conv_shortcut, dropout,
                                                 norm_type, activate_type))
            else:
                blocks.append(ResBlock2d(out_channels, out_channels, conv_shortcut, dropout,
                                                 norm_type, activate_type))
        for i in range(self.attn_repeat):
            blocks.append(AttnBlock2d(out_channels, out_channels, norm_type))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for i in range(self.res_repeat + self.attn_repeat):
            x = self.blocks[i](x)
        return x


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class ResDownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, dropout, conv_shortcut=False, norm_type='batch',
                 activate_type='swish', down_type='max'):
        super(ResDownBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.repeat = repeat

        res_blocks = []
        for i in range(self.repeat):
            if i == 0:
                res_blocks.append(ResBlock2d(in_channels, out_channels, conv_shortcut=conv_shortcut, dropout=dropout,
                                             norm_type=norm_type, activate_type=activate_type))
            else:
                res_blocks.append(ResBlock2d(out_channels, out_channels, conv_shortcut=conv_shortcut, dropout=dropout,
                                             norm_type=norm_type, activate_type=activate_type))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.down = DownSample(down_type, out_channels)

    def forward(self, x):
        for i in range(self.repeat):
            x = self.res_blocks[i](x)

        x = self.down(x)

        return x


class AttnDownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, norm_type='batch', down_type='conv'):
        super(AttnDownBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = repeat
        self.norm_type = norm_type
        self.down_type = down_type

        attn_blocks = []
        for i in range(repeat):
            if i == 0:
                attn_blocks.append(AttnBlock2d(in_channels, out_channels, norm_type))
            else:
                attn_blocks.append(AttnBlock2d(out_channels, out_channels, norm_type))
        self.attn_blocks = nn.ModuleList(attn_blocks)

        self.down = DownSample(down_type, out_channels)

    def forward(self, x):
        for i in range(self.repeat):
            x = self.attn_blocks[i](x)
        x = self.down(x)
        return x


class ResAttnDownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, res_repeat, attn_repeat, conv_shortcut=False,
                 dropout=None, norm_type='batch', activate_type='swish', down_type='max'):
        super(ResAttnDownBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_repeat = res_repeat
        self.attn_repeat = attn_repeat
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.norm_type = norm_type
        self.activate_type = activate_type
        self.down_type = down_type
        self.repeat = repeat

        blocks = []
        for i in range(repeat):
            if i == 0:
                blocks.append(ResAttnBlock2d(in_channels, out_channels, res_repeat, attn_repeat, conv_shortcut,
                                         dropout, norm_type, activate_type))
            else:
                blocks.append(ResAttnBlock2d(out_channels, out_channels, res_repeat, attn_repeat, conv_shortcut,
                                         dropout, norm_type, activate_type))
        self.blocks = nn.ModuleList(blocks)

        self.down = DownSample(down_type, out_channels)

    def forward(self, x):
        for i in range(self.repeat):
            x = self.blocks[i](x)
        x = self.down(x)
        return x


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class ResUpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, dropout, conv_shortcut=False, norm_type='batch',
                 activate_type='swish', up_type='conv'):
        super(ResUpBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.repeat = repeat

        res_blocks = []
        for i in range(self.repeat):
            if i == 0:
                res_blocks.append(ResBlock2d(in_channels, out_channels, conv_shortcut=conv_shortcut,
                                             dropout=dropout, norm_type=norm_type, activate_type=activate_type))
            else:
                res_blocks.append(ResBlock2d(out_channels, out_channels, conv_shortcut=conv_shortcut,
                                             dropout=dropout, norm_type=norm_type, activate_type=activate_type))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.up = UpSample(type=up_type, in_channels=in_channels)

    def forward(self, x):
        x = self.up(x)

        for i in range(self.repeat):
            x = self.res_blocks[i](x)

        return x


class AttnUpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, norm_type='batch', up_type='conv'):
        super(AttnUpBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = repeat

        attn_blocks = []
        for i in range(repeat):
            if i == 0:
                attn_blocks.append(AttnBlock2d(in_channels, out_channels, norm_type))
            else:
                attn_blocks.append(AttnBlock2d(out_channels, out_channels, norm_type))
        self.attn_blocks = nn.ModuleList(attn_blocks)

        self.up = UpSample(type=up_type, in_channels=in_channels)

    def forward(self, x):
        x = self.up(x)

        for i in range(self.repeat):
            x = self.attn_blocks[i](x)

        return x


class ResAttnUpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, repeat, res_repeat, attn_repeat, conv_shortcut=False,
                 dropout=None, norm_type='batch', activate_type='swish', up_type='conv'):
        super(ResAttnUpBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = repeat

        blocks = []
        for i in range(repeat):
            if i == 0:
                blocks.append(ResAttnBlock2d(in_channels, out_channels, res_repeat, attn_repeat, conv_shortcut,
                                         dropout, norm_type, activate_type))
            else:
                blocks.append(ResAttnBlock2d(out_channels, out_channels, res_repeat, attn_repeat, conv_shortcut,
                                         dropout, norm_type, activate_type))
        self.blocks = nn.ModuleList(blocks)

        self.up = UpSample(type=up_type, in_channels=in_channels)

    def forward(self, x):
        x = self.up(x)

        for i in range(self.repeat):
            x = self.blocks[i](x)

        return x


# QC-StyleGAN block
# inter_channels = kp_nums * in_channels
class DegradeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kp_nums, proj_nums, activate_type='relu'):
        super(DegradeBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channles = out_channels
        self.up_channel = nn.Conv2d(in_channels, out_channels*kp_nums, kernel_size=3, padding=1)
        blocks = []
        self.activ = Activate(activate_type)
        for i in range(proj_nums):
            blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            blocks.append(self.activ)
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feature_maps, sparse_motions):
        B, C, H, W = feature_maps.shape
        # B, N, H, W, 2 --> B, N, H, W
        sparse_motions = torch.norm(sparse_motions, p=2, dim=-1, keepdim=False)
        if feature_maps.shape[-2:] != sparse_motions.shape[-2:]:
            sparse_motions = Resize(feature_maps.shape[-2:])(sparse_motions)

        # B, C, H, W --> B, CN, H, W --> B, C, N, H, W
        feature_maps = self.up_channel(feature_maps).view(B, self.out_channles, -1, H, W)
        # B, N, H, W --> B, C, N, H, W
        sparse_motions = sparse_motions.unsqueeze(1).repeat(1, self.out_channles, 1, 1, 1)

        feature_maps = torch.sum(feature_maps * sparse_motions, dim=-3)
        for i in range(len(self.blocks)):
            feature_maps = self.blocks[i](feature_maps)
        feature_maps = self.final(feature_maps)
        return feature_maps


class QCUpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kp_nums, proj_nums, activate_type='relu', up_type='conv'):
        super(QCUpBlock2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.degrade = DegradeBlock(in_channels=in_channels, out_channels=out_channels,
                                    kp_nums=kp_nums, proj_nums=proj_nums,activate_type=activate_type)
        self.up = UpSample(type=up_type, in_channels=out_channels)

    def forward(self, feature_maps, sparse_motion):
        identity = self.conv(feature_maps)
        res = self.degrade(feature_maps, sparse_motion)
        return self.up(identity+res)


# texture pose fusion
class TPFusion(nn.Module):
    def __init__(self, input_channels):
        super(TPFusion, self).__init__()
        self.input_channels = input_channels
        self.norm = nn.BatchNorm2d(input_channels)
        self.hwt = DWT_Pooling()
        self.iwt = IWT_UpSampling()
        self.conv1 = nn.Conv2d(input_channels*4, input_channels*4 * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels*4 * 4, input_channels*4, kernel_size=3, padding=1)
        self.activ = nn.LeakyReLU()

    def forward(self, texture_features, pose_features):
        id = pose_features
        texture_features = self.hwt(self.norm(texture_features))
        pose_features = self.hwt(self.norm(pose_features))
        fused_features = torch.cat([pose_features[:,:self.input_channels,:,:], texture_features[:,self.input_channels:,:,:]], dim=1)
        fused_features = self.conv2(self.conv1(fused_features))
        fused_features = self.activ(fused_features)
        fused_features = self.iwt(fused_features)
        return fused_features + id










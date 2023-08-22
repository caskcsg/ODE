import pywt
import pywt.data
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


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


def iwt(x, data_format='channels_first'):
    B, C, H, W = x.shape
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
        return torch.reshape(torch.cat([y1, y2], dim=-1), [shape[0], shape[1]*2, shape[2]*2, shape[3]//4])

    elif data_format == 'channels_first':
        x_LL = x[:, 0:x.shape[1]//4, :, :]
        x_LH = x[:, x.shape[1]//4:x.shape[1]//4*2, :, :]
        x_HL = x[:, x.shape[1]//4*2:x.shape[1]//4*3, :, :]
        x_HH = x[:, x.shape[1]//4*3:, :, :]

        device = x.device

        out = torch.zeros(B, C//4, H * 2, W * 2).to(device)

        out[:, :, 0::2, 0::2] = (x_LL - x_LH - x_HL + x_HH) / 4
        out[:, :, 1::2, 0::2] = (x_LL - x_LH + x_HL - x_HH) / 4
        out[:, :, 0::2, 1::2] = (x_LL + x_LH - x_HL - x_HH) / 4
        out[:, :, 1::2, 1::2] = (x_LL + x_LH + x_HL + x_HH) / 4

        return out


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







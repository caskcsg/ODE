import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft2, irfft2, rfftfreq, fftfreq, fft2, ifft2
# from torchsummary import summary
from utils.wavelet import *
# from utils.wavelet import DWT_Pooling, IWT_UpSampling
# the model for input animation models


# true/false classifier and identity classifier
class FeatureClassifier(nn.Module):
    def __init__(self, config = [512, 512, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512], in_channel=256, identity_embedding_dim=2048, norm=True):
        super(FeatureClassifier, self).__init__()
        self.config = config
        layers = []
        for v in self.config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, v, kernel_size=3, padding=1)
                if norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(0.1)]
                else:
                    layers += [conv2d, nn.LeakyReLU(0.1)]
                in_channel = v
        self.layers = layers
        self.feature = nn.Sequential(*self.layers)
        self.final_pooling = nn.MaxPool2d(kernel_size=4)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, identity_embedding_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(identity_embedding_dim, 2)
        )

    def forward(self, corrupted_feature_map):
        x = self.feature(corrupted_feature_map)
        x = self.final_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# compute the ground truth given source and driving
class PermutationMatrix(nn.Module):
    def __init__(self, config=None):
        super(PermutationMatrix, self).__init__()


    def forward(self, source, driving):
        B, C, H, W = source.shape
        source = source.view(B, C, -1)
        source_transpose = torch.transpose(source, -1, -2)
        driving = driving.view(B, C, -1)
        pinverse = torch.matmul(source_transpose, torch.inverse(torch.matmul(source, source_transpose)))
        permutation_matrix = torch.matmul(pinverse, driving)
        predict = torch.matmul(source, permutation_matrix)
        predict = predict.view(B, C, H, W)
        return permutation_matrix, predict


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=256, wavelet_type='Haar', down_layer_nums=3):
        super(FeatureFusion, self).__init__()
        self.wavelet_type = wavelet_type
        self.down_layer_nums = down_layer_nums
        self.conv_down_layers = []
        self.conv_up_layers = []
        self.dwt_down = DWT_Pooling()
        self.iwt_up = IWT_UpSampling()
        for layer in range(down_layer_nums):
            # wt, iwt = self.get_wavelet_transform(in_channels, in_channels)
            # self.down_layers.append(wt)
            self.conv_down_layers.append(nn.Conv2d(in_channels*4, in_channels, kernel_size=3, padding=1))
            self.conv_down_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            # self.up_layers.append(iwt)
            self.conv_up_layers.append(nn.Conv2d(in_channels//4, in_channels, kernel_size=3, padding=1))
            self.conv_up_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.conv_down_layers = nn.Sequential(*self.conv_down_layers)
        self.conv_up_layers = nn.Sequential(*self.conv_up_layers)
        # self.layers = nn.Sequential(self.layers)

    # def get_wavelet_transform(self, in_channel, out_channel):
    #     dec_filters, rec_filters = create_wavelet_filter(wave=self.wavelet_type, in_size=in_channel, out_size=out_channel)
    #     wt_filters = nn.Parameter(dec_filters, requires_grad=False)
    #     iwt_filters = nn.Parameter(rec_filters, requires_grad=False)
    #     wt = get_transform(wt_filters, in_channel, 2)
    #     iwt = get_inverse_transform(iwt_filters, out_channel , 1)
    #     return wt, iwt

    def split_freq(self, feature_dwt):
        channel = feature_dwt.shape[1]
        feature_single_dwtl = feature_dwt[:, 0:channel//4, :, :]
        feature_single_dwth = feature_dwt[:, channel//4:, :, :]
        return feature_single_dwtl, feature_single_dwth


    def forward(self, feature_single, feature_multi):
        # downsample
        for i in range(self.down_layer_nums):
            # inject concat single high freq and multi low freq
            feature_single_dwt = self.dwt_down(feature_single)
            feature_single_dwtl, feature_single_dwth = self.split_freq(feature_single_dwt)
            feature_multi_dwt = self.dwt_down(feature_multi)
            feature_multi_dwtl, feature_multi_dwth = self.split_freq(feature_multi_dwt)
            feature_new = torch.cat([feature_multi_dwtl, feature_single_dwth], dim=1)
            feature_new = self.conv_down_layers[2*i](feature_new)
            feature_new = self.conv_down_layers[2*i+1](feature_new)
            feature_single = feature_new
            feature_multi = feature_multi_dwtl

        for i in range(self.down_layer_nums):
            feature_new = self.iwt_up(feature_new)
            feature_new = self.conv_up_layers[2*i](feature_new)
            feature_new = self.conv_up_layers[2*i+1](feature_new)

        return feature_new

        # wt, iwt = self.get_wavelet_transform(256, 256)
        # wt_feature_single = wt(feature_single)
        # wt_feature_multi = wt(feature_multi)
        # feature = torch.cat([feature_single, feature_multi], dim=1)
        # heat_map = self.layers(feature)
        # out = feature_single * heat_map + feature_multi * (1 - heat_map)
        # feature_single_fft = self.fft(feature_single)
        # feature_multi_fft = self.fft(feature_multi)
        # pass1 = torch.abs(fftfreq(feature_single.shape[-1])) < 0.4
        # pass2 = torch.abs(fftfreq(feature_single.shape[-2])) < 0.4
        # kernel = torch.outer(pass2, pass1)
        # fft_result = fft2(feature_single)
        # filtered = fft_result * kernel
        # result = torch.abs(ifft2(filtered, s=feature_multi.shape[-2:]))
        # feature_single_fft = rfft2(feature_single, dim=(-2, -1))
        # return feature_single_fft




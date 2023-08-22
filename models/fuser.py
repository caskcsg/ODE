import torch
import torch.nn as nn
from models.base_block import TPFusion


class MLF(nn.Module):
    def __init__(self, block_nums, in_channels):
        super(MLF, self).__init__()
        self.block_nums = block_nums
        fusion_blocks = []
        for i in range(block_nums):
            fusion_blocks.append(TPFusion(input_channels=in_channels))
        self.fusion_blocks = nn.ModuleList(fusion_blocks)

    def forward(self, texture_features, pose_features):
        for i in range(self.block_nums):
            pose_features = self.fusion_blocks[i](texture_features, pose_features)
        return pose_features




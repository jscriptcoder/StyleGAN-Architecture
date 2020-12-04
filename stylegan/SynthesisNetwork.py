import torch
import torch.nn as nn
import torch.nn.functional as F
from stylegan import SynthesisBlock, EqualizedConv2d


class SynthesisNetwork(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.progression = nn.ModuleList([
            SynthesisBlock(512, 512, initial=True), # 4x4, input block
            SynthesisBlock(512, 512, upsample=True), # 8x8
            SynthesisBlock(512, 512, upsample=True), # 16x16
            SynthesisBlock(512, 512, upsample=True), # 32x32
            SynthesisBlock(512, 256, upsample=True), # 64x64
            SynthesisBlock(256, 128, upsample=True), # 128x128
            SynthesisBlock(128, 64, upsample=True), # 256x256
            SynthesisBlock(64, 32, upsample=True), # 512x512
            SynthesisBlock(32, 16, upsample=True), # 1024x1024
        ])

        self.to_image = nn.ModuleList([
            EqualizedConv2d(512, 3, 1),
            EqualizedConv2d(512, 3, 1),
            EqualizedConv2d(512, 3, 1),
            EqualizedConv2d(512, 3, 1),
            EqualizedConv2d(256, 3, 1),
            EqualizedConv2d(128, 3, 1),
            EqualizedConv2d(64, 3, 1),
            EqualizedConv2d(32, 3, 1),
            EqualizedConv2d(16, 3, 1),
        ])
    
    def forward(self, w):
        self.progression(x)
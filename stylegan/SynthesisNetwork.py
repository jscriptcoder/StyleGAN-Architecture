import torch
import torch.nn as nn
import torch.nn.functional as F
from stylegan import SynthesisBlock


class SynthesisNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList([
            SynthesisBlock(512, 512, 4, initial=True),
            SynthesisBlock(512, 512, 8, upsample=True),
            SynthesisBlock(512, 512, 16, upsample=True),
            SynthesisBlock(512, 512, 32, upsample=True),
            SynthesisBlock(512, 256, 64, upsample=True),
            SynthesisBlock(256, 128, 128, upsample=True),
            SynthesisBlock(128, 64, 256, upsample=True),
            SynthesisBlock(64, 32, 512, upsample=True),
            SynthesisBlock(32, 16, 1024, upsample=True),
        ])
    
    def forward(self, w):
        
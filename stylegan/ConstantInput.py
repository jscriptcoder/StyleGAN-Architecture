import torch.nn as nn


class ConstantInput(nn.Module):
    def __init__(self, in_chan, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, in_chan, size, size)) # (batch, channels, height, width)

    def forward(self, x):
        batch = x.shape[0]
        return self.input.repeat(batch, 1, 1, 1)
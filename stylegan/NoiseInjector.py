import torch
import torch.nn as nn


class NoiseInjector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(0, 1, size=(1, channels, 1, 1)))

    def forward(self, x):
        noise_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
        noise = torch.randn(noise_shape, device=x.device)
        return x + self.weight * noise
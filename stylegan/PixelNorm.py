import torch
from torch.nn import Module


class PixelNorm(Module):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
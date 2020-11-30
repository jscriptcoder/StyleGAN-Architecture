import torch.nn as nn
import torch.nn.functional as F


class EqualizedConv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2**0.5,
                 use_wscale=True,
                 lrmul=1,
                 bias=True):

        super().__init__()

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init

        self.kernel_size = kernel_size

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)
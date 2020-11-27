import torch.nn as nn
import torch.nn.functional as F

class EqualizedLinear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2,
                 use_wscale=True,
                 lrmul=1.0,
                 bias=True):
        
        super(, self).__init__()

        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)

        return out
import torch.nn as nn
from stylegan.layers import EqualizedLinear


class MappingNetwork(nn.Module):
    def __init__(self, 
                 z_dim=512, 
                 hidden_dim=512, 
                 w_dim=512, 
                 n_layers=8):
        
        super().__init__()

        layers = []
        for i in n_layers:
            fmaps_in = z_dim if i == 0 else hidden_dim
            fmaps_out = w_dim if i == n_layers - 1 else hidden_dim
            layers.append(EqualizedLinear(fmaps_in, fmaps_out))
            layers.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.Sequential(*layers)

    def forward(self, noise):
        return self.mapping(noise)
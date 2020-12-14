import torch
import torch.nn as nn
import torch.nn.functional as F
from stylegan import MappingNetwork, SynthesisNetwork



class Generator(nn.Module):
    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan):
        super().__init__()

        self.mapping = MappingNetwork(z_dim, map_hidden_dim, w_dim)
        self.synthesis = SynthesisNetwork()

        self.alpha = 0.2

    def forward(self, noise, return_intermediate=False):
        pass
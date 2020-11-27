import torch.nn as nn
from stylegan.layers import EqualizedLinear


class MappingNetwork(nn.Module):
    def __init__(self, 
                 z_dim=512, 
                 hidden_dim=512, 
                 w_dim=512, 
                 n_layers=8):
        
        super().__init__()

        layers = [
            EqualizedLinear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        ]

        for i in n_layers - 2:
            layers.append(EqualizedLinear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        
        layers.append(EqualizedLinear(z_dim, w_dim))

        # NN that takes in tensors of 
        # shape (n_samples, z_dim) and outputs (n_samples, w_dim)
        # with a hidden layer with hidden_dim neurons
        self.mapping = nn.Sequential(*layers)

    def forward(self, noise):
        return self.mapping(noise)
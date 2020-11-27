import torch.nn as nn
from stylegan.layers import EqualizedLinear


class MappingNetwork(nn.Module):
    '''
    Mapping Network Class 𝑧 → 𝑤
    Values:
        z_dim: the dimension of the noise vector, a scalar
        hidden_dim: the inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
    '''
 
    def __init__(self, 
                 z_dim, 
                 hidden_dim, 
                 w_dim, 
                 n_layers=8, 
                 activ='lrelu'):
        
        super().__init__()

        activation = {
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(0.2),
        }[activ]

        layers = [
            EqualizedLinear(z_dim, hidden_dim),
            activation,
        ]

        for i in n_layers - 2:
            layers.append(EqualizedLinear(hidden_dim, hidden_dim))
            layers.append(activation)
        
        layers.append(EqualizedLinear(z_dim, w_dim))

        # NN that takes in tensors of 
        # shape (n_samples, z_dim) and outputs (n_samples, w_dim)
        # with a hidden layer with hidden_dim neurons
        self.mapping = nn.Sequential(*layers)

    def forward(self, noise):
        '''
        Function for completing a forward pass of MappingLayers: 
        Given an initial noise tensor, returns the intermediate noise tensor.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.mapping(noise)
import torch
import torch.nn as nn


class NoiseInjector(nn.Module):
    '''
    Noise Injector Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            # Initiate the weights for the channels from a random normal distribution
            torch.normal(0, 1, size=(1, channels, 1, 1))
        )

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise

        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel
import torch.nn as nn
from stylegan import ConstantInput, NoiseInjector, AdaIN, MappingNetwork, EqualizedConv2d


class SynthesisBlock(nn.Module):
    def __init__(self, in_chan, out_chan, resolution, kernel_size=3, initial=False, upsample=True, w_dim=512):
        super().__init__()
        
        if initial:
            self.first_layer = ConstantInput(in_chan, resolution)
        else:
            if upsample:
                self.first_layer = nn.Sequential(
                    nn.Upsample(size=resolution, mode='bilinear'),
                    EqualizedConv2d(in_chan, out_chan, kernel_size),
                )
            else:
                self.first_layer = EqualizedConv2d(in_chan, out_chan, kernel_size)
        
        self.noise_injector1 = NoiseInjector(out_chan)
        self.activ1 = nn.LeakyReLU(0.2) 
        self.adain1 = AdaIN(out_chan, w_dim)

        self.conv2 = EqualizedConv2d(out_chan, out_chan, kernel_size)
        self.noise_injector2 = NoiseInjector(out_chan)
        self.activ2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaIN(out_chan, w_dim)

    def forward(self, x, w):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        Parameters:
            x: the input into the generator, feature map of shape (n_samples, channels, width, height)
            w: the intermediate noise vector
        '''
        
        x = self.first_layer(x)

        x = self.noise_injector1(x)
        x = self.activ1(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        x = self.noise_injector2(x)
        x = self.activ2(x)
        x = self.adain2(x, w)

        return x
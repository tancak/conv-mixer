import torch
import torch.nn as nn

class ConvMixerBlock(nn.Module):
    '''
    Implementation of the convmixer block that gets repeatadly used for depth in the final network
    Performns depthwise and pointwise convolutions aswell as implements the residual connection between
    input and depthwise output
    '''
    def __init__(self, dim, kernel_size):
        super(ConvMixerBlock, self).__init__()

        self.conv_depth = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.conv_point = nn.Conv2d(dim, dim, kernel_size = 1)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv_depth(x)
        out = self.gelu(out)
        out = self.bn(out)

        out = out + x # Residual connection

        out = self.conv_point(out)
        out = self.gelu(out)
        out = self.bn(out)
        return out
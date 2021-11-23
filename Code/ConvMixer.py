import torch
import torch.nn as nn
from torchsummary import summary

class ConvMixerBlock(nn.Module):
    '''
    Implementation of the convmixer block that gets repeatadly used for depth in the final network
    Performns depthwise and pointwise convolutions aswell as implements the residual connection between
    input and depthwise output
    Params:
        dim - Number of dimensions for each patch
        kernel_size - Kernel size for the depthwise convolution
    '''
    def __init__(self, dim, kernel_size):
        super(ConvMixerBlock, self).__init__()

        self.conv_depth = nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same")
        self.conv_point = nn.Conv2d(dim, dim, kernel_size = 1)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv_depth(x)
        out = self.gelu(out)
        out = self.bn1(out)

        out = out + x # Residual connection

        out = self.conv_point(out)
        out = self.gelu(out)
        out = self.bn2(out)
        return out

class ConvMixer(nn.Module):
    '''
    Module for the ConvMixer network that contains multiple ConvMixerBlocks equivelent to the depth
    Params:
        num_classes - Number of class outputs used for the final fully connected layer
        dim - Dimensions for the patch embedding (default: 256)
        depth - Number of ConvMixerBlocks used before the final output (default: 8)
        kernel_size - Kernel size for the depthwise convolution (default: 5)
        patch_size - Size of each patch (default: 8)
    '''
    def __init__(self, num_classes, dim = 256, depth = 8, kernel_size = 5, patch_size = 8):

        
        super(ConvMixer, self).__init__()
        
        self.conv_embedding = nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.conv_mixer = nn.ModuleList([ConvMixerBlock(dim, kernel_size) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        out = self.conv_embedding(x)
        out = self.gelu(out)
        out = self.bn(out)
        
        for layer in self.conv_mixer:
            out = layer(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)

    net = ConvMixer(10).to(device)
    summary(net, (3, 32, 32))
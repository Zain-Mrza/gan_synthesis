import torch.nn as nn
import torch

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        # halves spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        if self.use_norm:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.act =  nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)

        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        # doubles spatial dimensions
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

        if self.use_norm:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

        self.act =  nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        
        return x

class Same(nn.Module):
    def __init__(self, channels, out_channels=None, groups=8, use_norm=True, act='silu'):
        super().__init__()
        self.use_norm = use_norm
        self.out_channels = out_channels if out_channels else channels

        self.conv = nn.Conv2d(in_channels=channels, out_channels=self.out_channels, kernel_size=3, padding="same", bias=False)
        if self.use_norm:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=self.out_channels)
        self.act =  nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)

        return x

class ChannelDouble(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=8):
        super().__init__()

        self.out_channels = out_channels if out_channels else in_channels*2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.norm = nn.GroupNorm(num_channels=self.out_channels, num_groups=groups)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
    
class Right(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()

        self.right = nn.Sequential(
            Same(in_channels, out_channels=out_channels, act=act),             # Doubles channel count
            Same(channels=out_channels, act=act),
            Same(channels=out_channels, act=act)
        )

    def forward(self, x):
        x = self.right(x)
        return x

class Contract(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.right = Right(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.right(x)
        x = self.pool(skip)

        return x, skip
    
class Expand(nn.Module):
    def __init__(self, in_channels, skip=True, up=True):
        super().__init__()
        
        self.skip = skip
        self.right = Right(in_channels=in_channels, out_channels=int(in_channels/2))
        self.up = Up(in_channels=int(in_channels/2), out_channels=int(in_channels/4))

    def forward(self, x, skip):
        if x.shape[1] != skip.shape[1]:
            print("Different channel dimensions for skip tensor and inptu tensor")
        print(x.shape, skip.shape)
        tensor = torch.cat((x, skip), dim=1) if self.skip else x
        x = self.right(tensor)
        x = self.up(x)
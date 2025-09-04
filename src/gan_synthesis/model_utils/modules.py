import torch.nn as nn

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
    def __init__(self, channels, groups=8, use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same", bias=False)
        if self.use_norm:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.act =  nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)

        return x
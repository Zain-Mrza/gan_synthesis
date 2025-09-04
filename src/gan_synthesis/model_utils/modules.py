import torch.nn as nn

class Down(nn.module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()

        # halves spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.act =  nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class Up(nn.module):
    def __init___(self, in_channels, out_channels, groups=8):
        super().__init__()

        # doubles spatial dimensions
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels),
        self.act =  nn.SiLU(),

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x

class Same(nn.module):
    def __init__(self, channels, groups=8):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same", bias=False),
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels),
        self.act =  nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
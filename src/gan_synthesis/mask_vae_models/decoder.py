import torch.nn as nn
import torch

from gan_synthesis.model_utils.modules import Same, Up


class Decoder(nn.Module):
    def __init__(self, size=96, out_channels=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, int(128 * 12*12))
        self.unflatten = nn.Unflatten(1, (128, 12, 12)) # Starting with 128 channels of 12x12
        self.unflatten_skip = nn.Unflatten(1, (8, 48, 48))

        self.block1 = nn.Sequential(
            Same(128),
            Same(128),
            Up(128, 64),        # 12x12 -> 24x24
        )

        self.block2 = nn.Sequential(
            Up(64, 32),         # 24x24 -> 48x48
            Same(32),
            Same(32),
        )

        self.block3 = nn.Sequential(
            Up(40, 16),         # 48x48 -> 96x96
            Same(16),
            Same(16)
        )
        
        self.out = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0) 

    def forward(self, x):
        x = self.fc(x)
        skip_noise = self.unflatten_skip(x)
        x = self.unflatten(x)

        x = self.block1(x)
        x = self.block2(x)
        x = torch.cat((x, skip_noise), dim=1)
        x = self.block3(x)

        x = self.out(x)
        return x

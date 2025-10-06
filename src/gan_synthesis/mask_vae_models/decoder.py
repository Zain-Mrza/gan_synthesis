import torch.nn as nn

from gan_synthesis.model_utils.modules import Same, Up


class Decoder(nn.Module):
    def __init__(self, size=96, out_channels=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, int(128 * 12*12))
        self.unflatten = nn.Unflatten(1, (128, 12, 12)) # Starting with 128 channels of 12x12

        self.decoder = nn.Sequential(
            Same(128),
            Same(128),
            Up(128, 64),        # 12x12 -> 24x24

            nn.Dropout(p=0.5),

            Up(64, 32),         # 24x24 -> 48x48
            Same(32),
            Same(32),
            Up(32, 16),         # 48x48 -> 96x96
            Same(16),
            Same(16)
        )
        
        self.out = nn.Conv2d(16, out_channels, kernel_size=1, stride=1, padding=0) 

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = self.out(x)
        return x

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, size=96, out_channels=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, int(128 * size / 8 * size / 8))

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, int(size / 8), int(size / 8))),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),
            
            nn.Dropout(p=0.3),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 24 → 48
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 24 → 48
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(num_groups=8, num_channels=16),
            nn.SiLU(),

            nn.Conv2d(
                16, out_channels, kernel_size=1, stride=1, padding=0
            ),  # 48 → 96
        )

    def forward(self, x):
        x = self.fc(x)  # (B, latent_dim) → (B, 128*12*12)
        x = self.decoder(x)  # (B, 128*12*12) → (B, out_channels, 96, 96)
        return x

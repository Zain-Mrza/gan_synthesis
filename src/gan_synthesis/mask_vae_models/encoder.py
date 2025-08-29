import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, size=96, in_channels=1, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 96x96 → 48x48
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 48x48 → 24x24
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24 → 12x12
            nn.SiLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),  # 24x24 → 12x12
            nn.SiLU(),


        )

        # size / 8
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(int(128 * size / 8 * size / 8), latent_dim)
        self.fc_logvar = nn.Linear(int(128 * size / 8 * size / 8), latent_dim)

    def forward(self, x):
        x = self.encoder(x)  # shape: (B, 128, 12, 12)
        x = self.flatten(x)  # shape: (B, 128*12*12)
        mu = self.fc_mu(x)  # shape: (B, latent_dim)
        logvar = self.fc_logvar(x)  # shape: (B, latent_dim)
        return mu, logvar

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=4, stride=2, padding=1
            ),  # 96x96 → 48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 48x48 → 24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24x24 → 12x12
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 12 * 12, latent_dim)
        self.fc_logvar = nn.Linear(128 * 12 * 12, latent_dim)

    def forward(self, x):
        x = self.encoder(x)  # shape: (B, 128, 12, 12)
        x = self.flatten(x)  # shape: (B, 128*12*12)
        mu = self.fc_mu(x)  # shape: (B, latent_dim)
        logvar = self.fc_logvar(x)  # shape: (B, latent_dim)
        return mu, logvar

import torch.nn as nn
from gan_synthesis.model_utils.modules import Down, Same


class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            Down(in_channels, 32),            # Spatial dim halved (96 -> 48)
            Same(32),
            Same(32),
            Down(32, 64),                     # Spatial dim halved (48 -> 24)
            Same(64, 64),
            Down(64, 128, use_norm=False),    # Spatial dim halved (24 -> 12)
            Same(128, 128, use_norm=False)
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(int(128 * 12*12), latent_dim)        # TODO: Consider global pooling instead
        self.fc_logvar = nn.Linear(int(128 * 12*12), latent_dim)

    def forward(self, x):
        x = self.encoder(x)  # shape: (B, 128, 12, 12)
        x = self.flatten(x)  # shape: (B, 128*12*12)
        mu = self.fc_mu(x)  # shape: (B, latent_dim)
        logvar = self.fc_logvar(x)  # shape: (B, latent_dim)
        return mu, logvar

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, out_channels=4, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 12 * 12)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 12, 12)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12 → 24
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 24 → 48
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=4, stride=2, padding=1
            ),  # 48 → 96
        )

    def forward(self, x):
        x = self.fc(x)  # (B, latent_dim) → (B, 128*12*12)
        x = self.decoder(x)  # (B, 128*12*12) → (B, out_channels, 96, 96)
        return x

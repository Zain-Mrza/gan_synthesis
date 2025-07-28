import torch
import torch.nn as nn
from torch.utils.data import random_split


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * logvar

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def split(self, trainp):
        if trainp > 1:
            raise ValueError("Must be a decimal as a percent")
        train_size = int(trainp * len(self))
        test_size = len(self) - train_size

        generator = torch.Generator().manual_seed(42)

        return random_split(self, [train_size, test_size], generator=generator)


def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = encoder.latent_dim
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

    @torch.no_grad()
    def sample(self, num_samples=1, device="cuda"):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z).argmax(dim=1).squeeze()
        return samples


def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar)

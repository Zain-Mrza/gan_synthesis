import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from gan_synthesis.mask_vae_models.decoder import Decoder
from gan_synthesis.mask_vae_models.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    @torch.inference_mode()
    def sample(self, num_samples=1, overlay=None):
        """
        Function to create fake segmentation mask.
        """

        device = next(self.parameters()).device

        # Sample latent z and decode
        z = torch.randn(num_samples, self.latent_dim, device=device)
        logits = self.decoder(z)  # [N, C, H, W]

        
        preds = logits.argmax(dim=1)  # [N, H, W]

        # Move to CPU for plotting
        preds_np = preds.cpu().numpy()

        # Plot each sample
        for i in range(num_samples):
            plt.figure(figsize=(4, 4))
            if overlay is not None:
                img = overlay[i]
                if img.ndim == 3 and img.shape[0] == 1:  # [1, H, W] → [H, W]
                    img = img.squeeze(0)
                plt.imshow(img.cpu().numpy(), cmap="gray")
                plt.imshow(preds_np[i], cmap="tab20", alpha=0.4)  # semi-transparent mask
            else:
                plt.imshow(preds_np[i])  # mask only
            
            plt.title(f"Sample {i+1}")
            plt.axis("off")
            plt.show()


def kl_divergence(mu, logvar):

    # KL term (mean over batch)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1 - logvar, dim=1).mean()

    return kl

def kl_divergence_capacity(mu, logvar, epoch, max_capacity=25.0, capacity_epochs=200, beta=10.0):
    """
    KL with capacity scheduling from Burgess et al. 2018.
    mu, logvar: [N, z_dim]
    """
    # KL per sample
    kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
    kl_mean = kl_per_sample.mean()

    # Target capacity at this epoch
    C = min(max_capacity, max_capacity * epoch / capacity_epochs)

    # Penalize deviation from target
    return beta * torch.abs(kl_mean - C)

def dice_loss_mc_opts(logits, target, eps=1e-6, ignore_bg=False, class_weights=None, ignore_index=None):
    """
    logits: [N, C, H, W]
    target: [N, H, W] (long)
    ignore_bg: if True, exclude class 0 from the Dice average
    class_weights: tensor [C] to weight per-class Dice (after excluding ignored classes)
    ignore_index: int or None; pixels with this label are masked out from Dice
    """
    N, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)                      # [N,C,H,W]
    tgt_1h = F.one_hot(target.clamp_min(0), C).permute(0,3,1,2).float()

    if ignore_index is not None:
        # mask out those pixels from both probs and target
        mask = (target != ignore_index).float().unsqueeze(1)  # [N,1,H,W]
        probs = probs * mask
        tgt_1h = tgt_1h * mask

    dims = (0,2,3)
    inter = (probs * tgt_1h).sum(dims)                        # [C]
    denom = (probs + tgt_1h).sum(dims)                        # [C]
    dice = (2*inter + eps) / (denom + eps)                    # [C]

    # choose classes to average
    start = 1 if ignore_bg else 0
    dice = dice[start:]
    if class_weights is not None:
        w = class_weights[start:].to(dice.device)
        w = w / (w.sum() + 1e-12)
        loss = 1 - (w * dice).sum()
    else:
        loss = 1 - dice.mean()
    return loss

def kl_freebits(mu, logvar, free_bits=0.03, beta=0.5):
    # per-dim free bits
    kl_dim = 0.5*(mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [N,z]
    kl_dim = torch.clamp(kl_dim.mean(0), min=free_bits)      # [z]
    return beta * kl_dim.sum()

def focal_ce_loss(logits, targets, alpha=None, gamma=2.0, reduction="mean"):
    # logits: [N, K, H, W], targets: [N, H, W] (int labels 0..K-1)
    logpt = -F.cross_entropy(logits, targets, reduction="none", weight=alpha)  # = log p_t
    pt = torch.exp(logpt)                              # p_t
    loss = ((1 - pt) ** gamma) * (-logpt)              # (1-p_t)^γ * CE
    return loss.mean() if reduction=="mean" else loss
import torch
import torch.nn as nn


def gn_act(ch, groups=8):  # GN + SiLU
    return nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(inplace=True))

class ResBlock(nn.Module):
    def __init__(self, ch, ch_out=None, groups=8, use_skip=True):
        super().__init__()
        ch_out = ch if ch_out is None else ch_out
        self.same_ch = (ch == ch_out)
        self.conv1 = nn.Conv2d(ch, ch_out, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, ch_out)
        self.act = nn.SiLU(inplace=True)

        self.skip_or_nah = use_skip                             #boolean

        if self.skip_or_nah:
            self.skip = (nn.Identity() if self.same_ch else nn.Conv2d(ch, ch_out, 1, bias=False))

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return self.act(h + self.skip(x)) if self.skip_or_nah else self.act(h)

class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = ResBlock(ch_in, ch_out)
        self.down = nn.Conv2d(ch_out, ch_out, 4, stride=2, padding=1, bias=False)  # strided conv
    def forward(self, x):
        x = self.block(x)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out, groups=8, use_skip=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up = nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False)

        if use_skip:
            self.block = ResBlock(ch_out + ch_skip, ch_out)  # concat skip
        else:
            self.block = ResBlock(ch_out, ch_out, use_skip=False)
        # learnable gate on the skip

        self.use_skip = use_skip

        if self.use_skip:
            self.alpha = nn.Parameter(torch.tensor(-2.0))  # init: moderate skip strength
            self.dropout = nn.Dropout(0.8)

        self.groups = groups

        self.use_skip = use_skip

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_up(x)
        # gated skip
        if self.use_skip:
            x = torch.cat([x, self.dropout(self.alpha.sigmoid() * skip)], dim=1)
        x = self.block(x)
        return x

class BottleneckAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # simple SE (channel attention) — cheap & effective
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch//4, 1), nn.SiLU(), nn.Conv2d(ch//4, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class Encoder(nn.Module):
    def __init__(self, in_ch=1, base=32, latent_dim=128):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1, bias=False)
        self.d1 = Down(base, base)         # 96 -> 48
        self.d2 = Down(base, base*2)       # 48 -> 24
        self.d3 = Down(base*2, base*4)     # 24 -> 12
        self.d4 = Down(base*4, base*8)     # 12 -> 6
        self.post = ResBlock(base*8, base*8)

        # global pooling head for µ, logσ²
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu     = nn.Conv2d(base*8, latent_dim, 1, bias=True)
        self.logvar = nn.Conv2d(base*8, latent_dim, 1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x, s4 = self.d4(x)
        x = self.post(x)
        h = self.pool(x)                   # B, C, 1, 1
        mu = self.mu(h).squeeze(-1).squeeze(-1)        # B, latent_dim
        logvar = self.logvar(h).squeeze(-1).squeeze(-1)
        return (mu, logvar), (s1, s2, s3, s4), x       # return skips + bottleneck feat

class Decoder(nn.Module):
    def __init__(self, out_ch=4, base=32, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base*8*6*6)   # start at 6×6
        self.pre = ResBlock(base*8, base*8)
        self.attn = BottleneckAttn(base*8)

        self.u1 = Up(base*8, base*8, base*4)  # 6 -> 12, skip s4
        self.u2 = Up(base*4, base*4, base*2)  # 12 -> 24, skip s3
        self.u3 = Up(base*2, base*2, base*1, use_skip=False)  # 24 -> 48, skip s2
        self.u4 = Up(base*1, base*1, base*1, use_skip=False)  # 48 -> 96, skip s1

        self.head = nn.Sequential(
            ResBlock(base, base),
            nn.Conv2d(base, out_ch, 1, bias=True)      # logits (no softmax)
        )

    def forward(self, z, skips, bottleneck=None):
        s1, s2, s3, s4 = skips
        x = self.fc(z).view(z.size(0), -1, 6, 6)
        x = self.pre(x)
        x = self.attn(x)

        x = self.u1(x, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)
        return self.head(x)


class VAE(nn.Module):
    """Variational Autoencoder wiring for the provided Encoder/Decoder.

    Forward returns (logits, mu, logvar). Use logits with CE/Dice/Focal.
    """
    def __init__(self, in_ch=1, out_ch=4, base=32, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_ch=in_ch, base=base, latent_dim=latent_dim)
        self.decoder = Decoder(out_ch=out_ch, base=base, latent_dim=latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x)||N(0,I)) per-sample: 0.5 * sum( exp(logvar)+mu^2-1-logvar )."""
        return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=1)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor, skips_bottleneck=None):
        if skips_bottleneck is None:
            base = self.decoder.head[0].conv1.in_channels  # or self.decoder.base if you store it
            B, device = z.size(0), z.device
            s1 = torch.zeros(B, base,     96, 96, device=device)
            s2 = torch.zeros(B, base*2,   48, 48, device=device)
            s3 = torch.zeros(B, base*4,   24, 24, device=device)
            s4 = torch.zeros(B, base*8,   12, 12, device=device)
            return self.decoder(z, (s1, s2, s3, s4))
        else:
            skips, bottleneck = skips_bottleneck
            return self.decoder(z, skips, bottleneck)


    def forward(self, x: torch.Tensor):
        (mu, logvar), skips, bottleneck = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, skips, bottleneck)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        base = self.decoder.head[0].conv1.in_channels  # or self.decoder.base
        s1 = torch.zeros(n, base,     96, 96, device=device)
        s2 = torch.zeros(n, base*2,   48, 48, device=device)
        s3 = torch.zeros(n, base*4,   24, 24, device=device)
        s4 = torch.zeros(n, base*8,   12, 12, device=device)
        return self.decoder(z, (s1, s2, s3, s4))


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two 3x3 convs each followed by GroupNorm + ReLU.
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        # groups must divide channels; we choose groups=min(8, out_ch) but ensure divisibility
        g1 = min(groups, out_ch)
        if out_ch % g1 != 0:
            # fall back to 1 group if not divisible (LayerNorm-like over channels)
            g1 = 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool(2) then DoubleConv.
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d to double H/W, concat with skip, then DoubleConv.
    We define channels explicitly to avoid shape confusion.
      - x1_ch: channels from the deeper feature map (before upsample)
      - x2_ch: channels from the skip connection
      - out_ch: output channels after the DoubleConv
    """
    def __init__(self, x1_ch: int, x2_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        # produce the same channels as the skip so concat makes x1_ch channels total
        self.up = nn.ConvTranspose2d(x1_ch, x2_ch, kernel_size=2, stride=2)
        # After concat we have x2_ch (upsampled) + x2_ch (skip) = x1_ch channels
        self.conv = DoubleConv(x1_ch, out_ch, groups=groups)

    @staticmethod
    def _pad_to(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pad x to match ref's H,W (handles rare off-by-ones)."""
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self._pad_to(x2, x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 projection to class logits."""
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for 96x96 single-channel inputs -> 4-class segmentation logits.

    Shapes (H=W=96) with 4 downsamples:
        96 -> 48 -> 24 -> 12 -> 6  (bottleneck)
        6  -> 12 -> 24 -> 48 -> 96 (upsampling back)

    Args:
        in_channels:   Input channels (default 1)
        num_classes:   Number of output classes (default 4)
        base_channels: Base width; network widths are
                       [base, 2*base, 4*base, 8*base, 16*base]
        groups:        GroupNorm groups (defaults to 8; must divide channels)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        groups: int = 8,
    ):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16  # bottleneck

        # Encoder
        self.inc = DoubleConv(in_channels, c1, groups=groups)
        self.down1 = Down(c1, c2, groups=groups)
        self.down2 = Down(c2, c3, groups=groups)
        self.down3 = Down(c3, c4, groups=groups)
        self.down4 = Down(c4, c5, groups=groups)

        # Decoder
        self.up1 = Up(c5, c4, c4, groups=groups)
        self.up2 = Up(c4, c3, c3, groups=groups)
        self.up3 = Up(c3, c2, c2, groups=groups)
        self.up4 = Up(c2, c1, c1, groups=groups)

        # Head
        self.outc = OutConv(c1, num_classes)

        self._init_weights()

    @staticmethod
    def _kaiming_init(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_weights(self) -> None:
        self.apply(self._kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Return raw logits (apply nn.CrossEntropyLoss during training).
        """
        x1 = self.inc(x)   # (B, c1, 96, 96)
        x2 = self.down1(x1)  # (B, c2, 48, 48)
        x3 = self.down2(x2)  # (B, c3, 24, 24)
        x4 = self.down3(x3)  # (B, c4, 12, 12)
        x5 = self.down4(x4)  # (B, c5, 6, 6)

        y = self.up1(x5, x4)  # -> (B, c4, 12, 12)
        y = self.up2(y, x3)   # -> (B, c3, 24, 24)
        y = self.up3(y, x2)   # -> (B, c2, 48, 48)
        y = self.up4(y, x1)   # -> (B, c1, 96, 96)

        logits = self.outc(y)  # (B, num_classes, 96, 96)
        return logits


# Quick self-test
if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=4, base_channels=32)
    x = torch.randn(2, 1, 96, 96)
    with torch.no_grad():
        y = model(x)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))  # should be (2, 4, 96, 96)

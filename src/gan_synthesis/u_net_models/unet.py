import torch.nn as nn

from gan_synthesis.model_utils.modules import Right, Up
from gan_synthesis.u_net_models.contracting import Contracting
from gan_synthesis.u_net_models.expanding import Expanding


class UNet(nn.Module):
    def __init__(self, anchor=16):
        super().__init__()
        self.contract = Contracting(anchor=anchor)
        self.expand = Expanding(anchor=anchor)
        self.bottleneck = nn.Sequential(
            Right(in_channels=anchor*4, out_channels=anchor*8),
            Up(in_channels=anchor*8, out_channels=anchor*4)
        )
        self.head = nn.Conv2d(in_channels=anchor, out_channels=4, kernel_size=1, padding=0)

    def forward(self, x):
        x, skips = self.contract(x)
        x = self.bottleneck(x)
        x = self.expand(x, skips)

        return self.head(x)

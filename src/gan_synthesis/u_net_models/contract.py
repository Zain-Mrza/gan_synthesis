import torch.nn as nn
from gan_synthesis.model_utils.modules import Contract

class UNet(nn.Module):
    def __init__(self, anchor=16):
        super().__init__()

        self.contract1 = Contract(in_channels=1, out_channels=anchor)                     # channels are set to anchor
        self.contract2 = Contract(in_channels=anchor, out_channels=anchor*2)              # channels are doubled
        self.contract3 = Contract(in_channels=anchor*2, out_channels=anchor*4)            # channels are doubled
        self.contract4 = Contract(in_channels=anchor*4, out_channels=anchor*8)            # channels are doubled
    
    def forward(self, x):
        x, s0 = self.contract1(x)                          # anchor
        x, s1 = self.contract2(x)                          # anchor * 2
        x, s2 = self.contract3(x)                          # anchor * 4
        x, _ = self.contract4(x)                          # anchor * 8

        return x

import torch.nn as nn

from gan_synthesis.model_utils.modules import Contract


class Contracting(nn.Module):
    def __init__(self, anchor):
        super().__init__()

        self.contract1 = Contract(in_channels=1, out_channels=anchor)                     # channels are set to anchor
        self.contract2 = Contract(in_channels=anchor, out_channels=anchor*2)              # channels are doubled
        self.contract3 = Contract(in_channels=anchor*2, out_channels=anchor*4)            # channels are doubled
    
    def forward(self, x):
        x, top = self.contract1(x)                          # anchor
        x, middle = self.contract2(x)                          # anchor * 2
        x, bottom = self.contract3(x)                          # anchor * 4

        return x, (top, middle, bottom)

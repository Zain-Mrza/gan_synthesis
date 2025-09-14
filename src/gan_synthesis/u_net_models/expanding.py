import torch
import torch.nn as nn

from gan_synthesis.model_utils.modules import Expand, Right


class Expanding(nn.Module):
    def __init__(self, anchor):
        super().__init__()
        self.expand1 = Expand(in_channels=anchor*8)
        self.expand2 = Expand(in_channels=anchor*4)

        self.right = Right(in_channels=anchor*2, out_channels=anchor)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skips):
        top, middle, bottom = skips
    
        x = self.expand1(x, bottom)
        x = self.expand2(x, middle)
        x = torch.cat((x, top), dim=1)
        x = self.right(x)
        return x

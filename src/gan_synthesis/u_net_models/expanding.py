from gan_synthesis.model_utils.modules import Expand, Right
import torch.nn as nn
import torch

class Expanding(nn.Module):
    def __init__(self, anchor):
        super().__init__()

        self.expand1 = Expand(in_channels=anchor*8)
        self.expand2 = Expand(in_channels=anchor*4)

        self.right = Right(in_channels=anchor*2, out_channels=anchor)

    def forward(self, x, skips):
        top, middle, bottom = skips
        x = self.expand1(x, bottom)
        x = self.expand2(x, middle)
        x = torch.cat((x, top))
        x = self.right(x)

        return x


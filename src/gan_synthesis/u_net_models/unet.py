import random

import matplotlib.pyplot as plt
import torch
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

    def compare(self, dataset):
        self.to("cpu")
        index = random.randint(0, len(dataset)-1)
        contrast, seg = dataset[index]

        self.eval()
        with torch.no_grad():
            recon = torch.squeeze(torch.argmax(self(contrast.unsqueeze(0)), dim=1)).numpy()
        
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))
        for ax in axs.ravel():
            ax.axis("off")
        axs[0].imshow(torch.squeeze(contrast), cmap='gray')
        axs[0].set_title("Original Image")

        axs[1].imshow(torch.squeeze(seg))
        axs[1].set_title("Original Segmentation Map")

        axs[2].imshow(recon)
        axs[2].set_title("Reconstrcuted Segmentation Map")

        plt.tight_layout()
        plt.show()

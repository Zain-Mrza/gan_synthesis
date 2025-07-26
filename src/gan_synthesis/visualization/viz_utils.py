import math

import matplotlib.pyplot as plt
import numpy as np

from gan_synthesis.preprocessing.transforms import iter_slices


def show_all_slices(volume, step=1, cmap="gray", cols=6):
    """
    Display axial slices of a 3D volume in a fixed-column grid layout.

    Parameters:
        volume (ndarray): 3D NumPy array (H x W x D).
        step (int): Step size for selecting slices.
        cmap (str): Colormap for displaying the slices.
        cols (int): Number of images per row.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D array.")

    slices = [z for z in iter_slices(volume, step)]
    n = len(slices)

    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < n:
            axes[i].imshow(slices[i], cmap=cmap)
            axes[i].set_title(f"Slice {i * step}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def overlay_mask(image, mask, alpha=0.5, cmap="magma"):
    plt.imshow(image, cmap="gray", alpha=1)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.show()

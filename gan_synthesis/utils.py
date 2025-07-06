import math

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler


def is_acceptable(training_example, im_type):
    acceptable = {"seg": "seg", "contrast": "t1ce"}
    if int(training_example) > 369:
        raise ValueError("No such file, only 369 training files exist")
    if isinstance(training_example, int):
        training_example = str(training_example)
    training_example = training_example.zfill(3)
    if im_type not in acceptable:
        raise ValueError("im_types must be either 'seg' or 'contrast'")
    return acceptable[im_type], training_example


def scale(volume):
    scaler = MinMaxScaler()

    volume_transformed = scaler.fit_transform(volume.reshape(-1, 1))
    volume = volume_transformed.reshape(volume.shape)
    return volume


def read(training_example: str or int, im_type: str):
    name, string_num = is_acceptable(training_example, im_type)

    volume = nib.load(
        rf"C:\Users\zzmir\gan_synthesis\data\train\BraTS20_Training_{string_num}\BraTS20_Training_{string_num}_{name}.nii"
    )
    volume = volume.get_fdata()
    volume = crop(volume)
    if im_type == "seg":
        volume[volume == 4] = 3
        return volume

    return scale(volume)


def crop(volume, square_dim=128):
    """
    Center-crops a 240x240x155 volume to square_dim x square_dim x 155.

    Parameters:
        volume (ndarray): Input 3D volume of shape (240, 240, 155).
        square_dim (int): Desired square size to crop to (must be <= 240).

    Returns:
        ndarray: Cropped volume of shape (square_dim, square_dim, 155).
    """
    if square_dim > 240:
        raise ValueError("square_dim must be less than or equal to 240")

    start = (240 - square_dim) // 2
    end = start + square_dim
    return volume[start:end, start:end, :]


def show_slices(volume, step=1, cmap="gray"):
    for z in iter_slices(volume, step):
        plt.imshow(z, cmap="gray")
        plt.show()


def iter_slices(volume, step=1):
    for depth in range(0, volume.shape[2], step):
        yield volume[:, :, depth]


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


def count_tumor_pixels(seg_slice):
    _, counts = np.unique(seg_slice, return_counts=True)
    return sum(counts[1:])


def overlay_mask(
    image,
    mask,
    alpha=0.5,
):
    cmap = ListedColormap(
        [
            (0, 0, 0, 0),  # class 0: fully transparent
            (1, 0, 0, alpha),  # class 1: red
            (0, 1, 0, alpha),  # class 2: green
            (0, 0, 1, alpha),  # class 3: blue
        ]
    )

    plt.imshow(image, cmap="gray", alpha=1)
    plt.imshow(mask, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.show()

from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess():
    types = ["seg", "contrast"]
    for filenum in range(1, 370):
        depth = None
        for t in types:
            volume = read(filenum, t)
            if t == "seg":
                depth = find_max_tumor_slice(volume)
            matrix = volume[:, :, depth]
            np.save(
                rf"C:\Users\zzmir\gan_synthesis\processed_data\{t}_slice_{filenum - 1}",
                matrix,
            )


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


def iter_slices(volume, step=1):
    for depth in range(0, volume.shape[2], step):
        yield volume[:, :, depth]


def count_tumor_pixels(seg_slice):
    _, counts = np.unique(seg_slice, return_counts=True)
    return sum(counts[1:])


def find_max_tumor_slice(volume):
    """
    Finds the index of the slice (along the z-axis) with the most tumor pixels.

    Parameters:
        volume (ndarray): 3D segmentation volume.

    Returns:
        int: Index of the slice with the highest tumor pixel count.
    """
    max_index = 0
    max_count = 0

    for depth, seg_slice in enumerate(iter_slices(volume)):
        count = count_tumor_pixels(seg_slice)
        if count > max_count:
            max_index = depth
            max_count = count

    return max_index


def find_center(image):
    mask = image > 0
    y_indices = np.where(np.any(mask, axis=1))[0]
    x_indices = np.where(np.any(mask, axis=0))[0]

    firsty, lasty = y_indices[0], y_indices[-1]
    firstx, lastx = x_indices[0], x_indices[-1]

    centerx = (firstx + lastx) // 2
    centery = (firsty + lasty) // 2

    return centerx, centery


def crop_tumor_center(index):
    image = read_data(index, "contrast")
    seg = read_data(index, "seg")
    centerx, centery = find_center(seg)

    minx = centerx - 48
    maxx = centerx + 48
    miny = centery - 48
    maxy = centery + 48

    # Make sure we stay in bounds
    minx = max(minx, 0)
    maxx = min(maxx, image.shape[1])
    miny = max(miny, 0)
    maxy = min(maxy, image.shape[0])

    image = image[miny:maxy, minx:maxx]
    seg = seg[miny:maxy, minx:maxx]

    return image, seg


def read_data(index, mode: str):
    root = find_project_root()
    image = np.load(rf"{root}\processed_data\{mode}_slice_{index}.npy")
    return image


def find_project_root(marker=".git"):
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found.")

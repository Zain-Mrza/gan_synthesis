import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler


def is_acceptable(training_example, type):
    acceptable = {"seg": "seg", "contrast": "t1ce"}
    if int(training_example) > 369:
        raise ValueError("No such file, only 369 training files exist")
        return False
    if isinstance(training_example, int):
        training_example = str(training_example)
    training_example = training_example.zfill(3)
    if type not in acceptable:
        raise ValueError("Types must be either 'seg' or 'contrast'")
        return False
    return acceptable[type], training_example


scaler = MinMaxScaler()


def scale(volume):
    volume_transformed = scaler.fit_transform(volume.reshape(-1, 1))
    volume = volume_transformed.reshape(volume.shape)
    return volume


def read(training_example: str or int, type: str):
    name, string_num = is_acceptable(training_example, type)

    volume = nib.load(
        r"C:\Users\zzmir\MNIST-GAN\data\train\BraTS20_Training_"
        + string_num
        + r"\BraTS20_Training_"
        + string_num
        + "_"
        + name
        + ".nii"
    )
    volume = volume.get_fdata()
    return scale(volume)


def show_slices(training_examples: list, type: str):
    for volume in training_examples:
        volume = read(volume, type)
        slice = volume[:, :, 90]
        plt.imshow(slice, cmap="gray")
        plt.show()
    plt.tight_layout()


examples = [1, 43, 234, 233, 2]
show_slices(examples, "contrast")
